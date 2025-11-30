#![allow(dead_code)]

mod dataformat;

use std::{
    array::{from_mut, from_ref},
    borrow::Cow,
    cmp::Reverse,
    collections::HashMap,
    fmt::{Display, Formatter},
    fs::{self, File},
    hash::Hash,
    io::{BufRead, BufReader, BufWriter, Seek, Write},
    ops::ControlFlow,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Instant,
};

use anyhow::{Context, anyhow, bail};
use bulletformat::ChessBoard;
use dataformat::Filter;
use rand::{Rng, rngs::ThreadRng, seq::IndexedRandom};

use crate::{
    chess::{
        CHESS960,
        board::{Board, DrawType, GameOutcome, WinType},
        chessmove::Move,
        piece::{Colour, PieceType},
        types::Square,
    },
    datagen::dataformat::Game,
    evaluation::{is_decisive, is_mate_score},
    nnue::network::NNUEParams,
    search::{parameters::Config, search_position, static_exchange_eval},
    tablebases::{self, probe::WDL},
    threadlocal::make_thread_data,
    threadpool,
    timemgmt::{SearchLimit, TimeManager},
    transpositiontable::TT,
    uci::{SYZYGY_ENABLED, SYZYGY_PATH},
    util::MEGABYTE,
};

/// Number of attempts to find a good random move before just picking any random move.
const RANDOM_MOVE_ATTEMPTS: usize = 8;
/// Number of random moves to make from the root position in classical startpos or DFRC.
const RANDOM_MOVES_ROOT: usize = 8;
/// Number of random moves to make from a book position.
const RANDOM_MOVES_BOOK: usize = 0;
/// The SEE threshold for random move selection.
const RANDOM_SEE_THRESHOLD: i32 = -1000;

/// Global atomic counter for tracking progress.
static FENS_GENERATED: AtomicU64 = AtomicU64::new(0);

/// Flag to indicate that generation should stop.
static STOP_GENERATION: AtomicBool = AtomicBool::new(false);

/// Configuration options for Viri's self-play data generation.
#[derive(Clone, Debug, Hash)]
struct DataGenOptions {
    // The number of games to generate.
    num_games: usize,
    // The number of threads to use.
    num_threads: usize,
    // The (optional) path to the directory containing syzygy endgame tablebases.
    tablebases_path: Option<PathBuf>,
    // The (optional) path to an EPD format book to use for generating starting positions.
    book: Option<PathBuf>,
    // The node limit for searches.
    nodes: u64,
    // Whether to generate DFRC data.
    generate_dfrc: bool,
}

/// Builder for datagen options.
pub struct DataGenOptionsBuilder {
    // The number of games to generate.
    pub games: usize,
    // The number of threads to use.
    pub threads: usize,
    // The (optional) path to the directory containing syzygy endgame tablebases.
    pub tbs: Option<PathBuf>,
    // The (optional) path to an EPD format book to use for generating starting positions.
    pub book: Option<PathBuf>,
    // The node limit for searches.
    pub nodes: u64,
    // Whether to generate DFRC data.
    pub dfrc: bool,
}

impl DataGenOptionsBuilder {
    fn build(self) -> DataGenOptions {
        DataGenOptions {
            num_games: self.games,
            num_threads: self.threads,
            tablebases_path: self.tbs,
            book: self.book,
            nodes: self.nodes,
            generate_dfrc: self.dfrc,
        }
    }
}

impl DataGenOptions {
    /// Creates a new `DataGenOptions` instance.
    const fn new() -> Self {
        Self {
            num_games: 100,
            num_threads: 1,
            tablebases_path: None,
            book: None,
            nodes: 25_000,
            generate_dfrc: true,
        }
    }

    /// Gives a summarised string representation of the options.
    fn summary(&self) -> String {
        format!(
            "{}g-{}t-{}-{}-n{}{}",
            self.num_games,
            self.num_threads,
            if self.tablebases_path.is_some() {
                format!("tb{}", tablebases::probe::get_max_pieces_count())
            } else {
                "no_tb".into()
            },
            if self.generate_dfrc {
                "dfrc"
            } else {
                "classical"
            },
            self.nodes,
            self.book.as_ref().map_or_else(String::new, |book| format!(
                "-{}",
                book.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .trim_end_matches(".epd")
            ))
        )
    }
}

trait StartposGenerator {
    fn generate(&mut self, board: &mut Board, conf: &Config) -> ControlFlow<(), ()>;
}

/// Make a random move on the board, attempting to pick only
/// moves with a static exchange evaluation above the given threshold.
/// If no such move is found, a random legal move is made.
/// If there are no legal moves, None is returned.
fn make_random_move(
    rng: &mut ThreadRng,
    board: &mut Board,
    conf: &Config,
    see_threshold: i32,
) -> Option<Move> {
    let legal_moves = board.legal_moves();
    for _ in 0..RANDOM_MOVE_ATTEMPTS {
        let m = *legal_moves.choose(rng)?;
        if static_exchange_eval(board, conf, m, see_threshold) {
            assert!(board.is_legal(m));
            board.make_move_simple(m);
            return Some(m);
        }
    }
    let m = *legal_moves.choose(rng)?;
    assert!(board.is_legal(m));
    board.make_move_simple(m);
    Some(m)
}

struct ClassicalStartposGenerator {
    rng: ThreadRng,
}

struct DFRCStartposGenerator {
    rng: ThreadRng,
}

struct BookStartposGenerator<'a> {
    rng: ThreadRng,
    source: &'a [&'a str],
    cursor: &'a AtomicUsize,
}

impl StartposGenerator for ClassicalStartposGenerator {
    fn generate(&mut self, board: &mut Board, conf: &Config) -> ControlFlow<(), ()> {
        board.set_startpos();

        for _ in 0..RANDOM_MOVES_ROOT + usize::from(self.rng.random_bool(0.5)) {
            let res = make_random_move(&mut self.rng, board, conf, RANDOM_SEE_THRESHOLD);
            if res.is_none() {
                return ControlFlow::Break(());
            }
            if board.outcome().is_some() {
                return ControlFlow::Break(());
            }
        }

        ControlFlow::Continue(())
    }
}

impl StartposGenerator for DFRCStartposGenerator {
    fn generate(&mut self, board: &mut Board, conf: &Config) -> ControlFlow<(), ()> {
        board.set_dfrc_idx(rand::Rng::random_range(&mut self.rng, 0..960 * 960));

        for _ in 0..RANDOM_MOVES_ROOT + usize::from(self.rng.random_bool(0.5)) {
            let res = make_random_move(&mut self.rng, board, conf, RANDOM_SEE_THRESHOLD);
            if res.is_none() {
                return ControlFlow::Break(());
            }
            if board.outcome().is_some() {
                return ControlFlow::Break(());
            }
        }

        ControlFlow::Continue(())
    }
}

impl StartposGenerator for BookStartposGenerator<'_> {
    fn generate(&mut self, board: &mut Board, conf: &Config) -> ControlFlow<(), ()> {
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed);
        if idx >= self.source.len() {
            println!("Book exhausted!");
            self.cursor.store(0, Ordering::Relaxed);
            return ControlFlow::Break(());
        }
        let fen = self.source[idx];
        board
            .set_from_fen(fen)
            .with_context(|| format!("Failed to set board from FEN {fen} at index {idx} in book."))
            .unwrap();

        #[allow(clippy::reversed_empty_ranges)]
        for _ in 0..RANDOM_MOVES_BOOK {
            let res = make_random_move(&mut self.rng, board, conf, RANDOM_SEE_THRESHOLD);
            if res.is_none() {
                return ControlFlow::Break(());
            }
            if board.outcome().is_some() {
                return ControlFlow::Break(());
            }
        }

        ControlFlow::Continue(())
    }
}

pub fn gen_data_main(cli_config: DataGenOptionsBuilder) -> anyhow::Result<()> {
    if !cfg!(feature = "datagen") {
        bail!("datagen feature not enabled (compile with --features datagen)");
    }

    ctrlc::set_handler(move || {
        STOP_GENERATION.store(true, Ordering::SeqCst);
        println!("Stopping generation, please don't force quit.");
    })
    .with_context(|| "Failed to set Ctrl-C handler")?;

    let nnue_params = NNUEParams::decompress_and_alloc()?;

    let options: DataGenOptions = cli_config.build();

    CHESS960.store(options.generate_dfrc, Ordering::SeqCst);
    FENS_GENERATED.store(0, Ordering::SeqCst);

    println!("Starting data generation with the following configuration:");
    println!("{options}");
    if !options.num_games.is_multiple_of(options.num_threads) {
        println!("Warning: The number of games is not evenly divisible by the number of threads,");
        println!(
            "this will result in {} games being omitted.",
            options.num_games % options.num_threads
        );
    }
    if let Some(tb_path) = &options.tablebases_path {
        let tb_path = tb_path.to_string_lossy();
        tablebases::probe::init(&tb_path);
        if let Ok(mut lock) = SYZYGY_PATH.lock() {
            *lock = tb_path.to_string();
        } else {
            bail!("Failed to take lock on SYZYGY_PATH");
        }
        SYZYGY_ENABLED.store(true, Ordering::SeqCst);
        println!("Syzygy tablebases enabled.");
    }

    // create a new unique identifier for this generation run
    // this is used to create a unique directory for the data
    // and to name the data files.
    // the ID is formed by taking the current date and time,
    // plus a compressed representation of the options struct.
    let run_id = format!(
        "run_{}_{}",
        chrono::Utc::now().format("%Y-%m-%d_%H-%M-%S"),
        options.summary()
    );
    println!("This run will be saved to the directory \"data/{run_id}\"");
    println!("Each thread will save its data to a separate file in this directory.");

    // create the directory for the data
    let data_dir = PathBuf::from("data").join(run_id);
    std::fs::create_dir_all(&data_dir).with_context(|| "Failed to create data directory")?;

    let mut counters = Vec::new();
    let book_positions = options
        .book
        .as_deref()
        .map(|book| fs::read_to_string(book).with_context(|| "Failed to read book file."))
        .transpose()?;
    let book_positions = book_positions
        .as_deref()
        .map(|book| book.lines().collect::<Vec<_>>());
    let cursor = AtomicUsize::new(0);
    let book_positions = book_positions.as_deref();
    let cursor = &cursor;
    std::thread::scope(|s| {
        let thread_handles = (0..options.num_threads)
            .map(|id| {
                let opt_ref = &options;
                let path_ref = &data_dir;
                let nnue_params_ref = &nnue_params;
                s.spawn(move || {
                    // this rng is different between each thread
                    // (https://rust-random.github.io/book/guide-parallel.html)
                    // so no worries :3
                    let rng = rand::rng();
                    let startpos_src = if let Some(source) = book_positions {
                        Box::new(BookStartposGenerator {
                            rng,
                            source,
                            cursor,
                        }) as Box<_>
                    } else if opt_ref.generate_dfrc {
                        Box::new(DFRCStartposGenerator { rng }) as Box<_>
                    } else {
                        Box::new(ClassicalStartposGenerator { rng }) as Box<_>
                    };
                    generate_on_thread(id, opt_ref, path_ref, nnue_params_ref, startpos_src)
                })
            })
            .collect::<Vec<_>>();
        for handle in thread_handles {
            if let Ok(res) = handle.join() {
                counters.push(res?);
            } else {
                bail!("Thread failed to join!");
            }
        }
        Ok(())
    })?;

    let counters = counters.into_iter().reduce(|mut acc, e| {
        for (key, value) in e {
            *acc.entry(key).or_insert(0) += value;
        }
        acc
    });

    println!("Done!");

    if let Some(counters) = counters {
        print_game_stats(&counters);
    }

    Ok(())
}

fn print_game_stats(counters: &HashMap<GameOutcome, u64>) {
    #![allow(clippy::cast_precision_loss)]
    let total = counters.values().sum::<u64>();
    eprintln!("Total games: {total}");
    let mut counters = counters.iter().collect::<Vec<_>>();
    counters.sort_unstable_by_key(|&(_, value)| Reverse(value));
    for (&key, &value) in counters {
        eprintln!(
            "{key:?}: {value} ({percentage}%)",
            percentage = (value as f64 / total as f64 * 100.0).round()
        );
    }
}

impl From<WDL> for GameOutcome {
    fn from(value: WDL) -> Self {
        match value {
            WDL::Win => Self::WhiteWin(WinType::TB),
            WDL::Loss => Self::BlackWin(WinType::TB),
            WDL::Draw => Self::Draw(DrawType::TB),
        }
    }
}

#[allow(clippy::too_many_lines)]
fn generate_on_thread<'a>(
    id: usize,
    options: &DataGenOptions,
    data_dir: &Path,
    nnue_params: &'static NNUEParams,
    mut startpos_src: Box<dyn StartposGenerator + 'a>,
) -> anyhow::Result<HashMap<GameOutcome, u64>> {
    // Datagen uses the default configuration:
    let conf = Config::default();

    // Whole datagen workers are multiplied across the machine,
    // so any given worker has only one thread for search.
    // This is good, because we don't have to contend with any
    // imperfect Lazy SMP scaling losses, and it makes the whole
    // procedure utterly parallel.
    let worker_thread = threadpool::make_worker_threads(1)
        .into_iter()
        .next()
        .unwrap();

    // We allocate four megabytes of cache for each worker,
    // because search budgets are so small and smaller TTs
    // increase speed. We split all state by colour, so that
    // the two players of each game can't "see into each
    // other's minds" via the TT / histories. Additionally,
    // search is tuned around each move in the game being
    // two ply forward from the previous move, so this is
    // prudent just from a robustness perspective.
    let mut tts = [TT::new(), TT::new()];
    tts[Colour::White].resize(4 * MEGABYTE, from_ref(&worker_thread));
    tts[Colour::Black].resize(4 * MEGABYTE, from_ref(&worker_thread));
    let stopped = AtomicBool::new(false);
    let nodes = AtomicU64::new(0);
    let tbhits = AtomicU64::new(0);
    let mut thread_data = std::array::from_fn::<_, 2, _>(|colour| {
        make_thread_data(
            &Board::default(),
            tts[colour].view(),
            nnue_params,
            &stopped,
            &nodes,
            &tbhits,
            from_ref(&worker_thread),
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
    });

    let time_manager = TimeManager::default_with_limit(SearchLimit::SoftNodes {
        soft_limit: options.nodes,
        hard_limit: options.nodes * 8,
    });
    thread_data[Colour::White].info.clock = time_manager.clone();
    thread_data[Colour::Black].info.clock = time_manager;
    thread_data[Colour::White].info.print_to_stdout = false;
    thread_data[Colour::Black].info.print_to_stdout = false;

    let n_games_to_run = std::cmp::max(options.num_games / options.num_threads, 1);

    let mut output_file = File::create(data_dir.join(format!("thread_{id}.bin")))
        .with_context(|| "Failed to create output file.")?;
    let mut output_buffer = BufWriter::new(&mut output_file);

    let mut counters = HashMap::<GameOutcome, u64>::new();

    let start = Instant::now();
    'generation_main_loop: for game in 0..n_games_to_run {
        // report progress
        if id == 0 && game % 32 == 0 && game > 0 {
            print_progress(n_games_to_run, &counters, start, game)?;
        }
        // reset everything: board, thread data, tt, search info
        for (tt, td) in tts.iter().zip(thread_data.iter_mut()) {
            tt.clear(from_ref(&worker_thread));
            td.info.set_up_for_search();
        }
        // generate game
        // STEP 1: get the next starting position from the callback
        let mut startpos = Board::empty();
        match startpos_src.generate(&mut startpos, &conf) {
            ControlFlow::Break(()) => continue 'generation_main_loop,
            ControlFlow::Continue(()) => {}
        }

        // set up both players with the starting position
        let [white, black] = &mut thread_data;
        white.board = startpos.clone();
        black.board = startpos.clone();
        white.nnue.reinit_from(&startpos, white.nnue_params);
        black.nnue.reinit_from(&startpos, black.nnue_params);

        let mut td = &mut thread_data[startpos.turn()];

        // STEP 2: evaluate the exit position with reasonable
        // effort to make sure that it's worth learning from.
        let temp_limit = td.info.clock.limit().clone();
        td.info.clock.set_limit(SearchLimit::SoftNodes {
            soft_limit: 50_000,
            hard_limit: 50_000 * 8,
        });
        let eval = search_position(from_ref(&worker_thread), from_mut(td)).0;
        td.info.clock.set_limit(temp_limit);
        if eval.abs() > 1000 {
            // if the position is too good or too bad, we don't want it
            continue 'generation_main_loop;
        }
        let mut game = Game::new(&td.board);
        // STEP 3: play out to the end of the game
        let mut win_adj_counter = 0;
        let mut draw_adj_counter = 0;
        let outcome = loop {
            let colour = td.board.turn();

            // set the thread data and TT to the correct colour.
            let tt = &tts[colour];
            let other;
            [td, other] = thread_data
                .get_disjoint_mut([colour as usize, !colour as usize])
                .unwrap();

            // hoik the board state from the other player
            // on the assumption that we always change sides
            // each iteration of this loop.
            td.board = other.board.clone();
            // │ pointlessly reflowing
            // └────────────────────▼
            td.nnue.reinit_from(&td.board, td.nnue_params);

            if let Some(outcome) = td.board.outcome() {
                break outcome;
            }
            if options.tablebases_path.is_some()
                && let Some(wdl) = tablebases::probe::get_wdl_white(&td.board)
            {
                break wdl.into();
            }

            tt.increase_age();
            let (score, best_move) = search_position(from_ref(&worker_thread), from_mut(td));

            let Some(best_move) = best_move else {
                println!("[WARNING!] search returned a null move as the best move!");
                println!("[WARNING!] this occurred in position {}", td.board);
                continue 'generation_main_loop;
            };

            game.add_move(
                best_move,
                score
                    .try_into()
                    .with_context(|| "Failed to convert score into eval.")?,
            );

            let abs_score = score.abs();
            if abs_score >= 2500 {
                win_adj_counter += 1;
                draw_adj_counter = 0;
            } else if abs_score <= 4 {
                draw_adj_counter += 1;
                win_adj_counter = 0;
            } else {
                win_adj_counter = 0;
                draw_adj_counter = 0;
            }

            if win_adj_counter >= 4 {
                break if score > 0 {
                    GameOutcome::WhiteWin(WinType::Adjudication)
                } else {
                    GameOutcome::BlackWin(WinType::Adjudication)
                };
            }
            if draw_adj_counter >= 12 {
                break GameOutcome::Draw(DrawType::Adjudication);
            }
            if is_decisive(score) {
                // if the score is game theoretic, we don't want to play out the rest of the game
                break match (score > 0, is_mate_score(score)) {
                    (true, true) => GameOutcome::WhiteWin(WinType::Mate),
                    (true, false) => GameOutcome::WhiteWin(WinType::TB),
                    (false, true) => GameOutcome::BlackWin(WinType::Mate),
                    (false, false) => GameOutcome::BlackWin(WinType::TB),
                };
            }

            td.board.make_move(best_move, &mut td.nnue);
        };

        // STEP 4: write the game to the output file
        // increment the counter
        FENS_GENERATED.fetch_add(game.len() as u64, Ordering::SeqCst);
        // update with outcome
        game.set_outcome(outcome);

        // write to file
        game.serialise_into(&mut output_buffer)
            .with_context(|| "Failed to serialise game into output buffer.")?;

        // STEP 5: update the game outcome statistics
        *counters.entry(outcome).or_default() += 1;

        // STEP 6: check if we should stop because the STOP_GENERATION signal was set.
        if STOP_GENERATION.load(Ordering::SeqCst) {
            break 'generation_main_loop;
        }
    }

    output_buffer
        .flush()
        .with_context(|| "Failed to flush output buffer to file.")?;

    Ok(counters)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn print_progress(
    n_games_to_run: usize,
    counters: &HashMap<GameOutcome, u64>,
    start: Instant,
    game: usize,
) -> Result<(), anyhow::Error> {
    let percentage = game * 100_000 / n_games_to_run;
    let percentage = percentage as f64 / 1000.0;
    let time_per_game = start.elapsed().as_secs_f64() / game as f64;
    let games_to_go = n_games_to_run as f64 - game as f64;
    let time_remaining = games_to_go * time_per_game;
    eprintln!(
        "[+] Main thread: Generated {game} games ({percentage:.1}%). Time per game: {time_per_game:.2} seconds."
    );
    eprintln!(
        " |> FENs generated: {fens} (FENs/sec = {fps:.2})",
        fens = FENS_GENERATED.load(Ordering::Relaxed),
        fps = FENS_GENERATED.load(Ordering::Relaxed) as f64 / start.elapsed().as_secs_f64()
    );
    eprintln!(" |> Estimated time remaining: {time_remaining:.2} seconds.");
    let est_completion_date = chrono::Local::now()
        .checked_add_signed(
            chrono::Duration::try_seconds(time_remaining as i64)
                .with_context(|| "failed to convert remaining time to seconds")?,
        )
        .with_context(|| "failed to add remaining time to current time")?;
    let time_completion = est_completion_date.format("%Y-%m-%d %H:%M:%S");
    eprintln!(" |> Estimated completion time: {time_completion}");
    std::io::stderr()
        .flush()
        .with_context(|| "Failed to flush stderr.")?;
    if game.is_multiple_of(1024) {
        eprintln!("Game stats for main thread:");
        print_game_stats(counters);
    }

    Ok(())
}

fn show_boot_info(options: &DataGenOptions) {
    println!("Welcome to Viri's data generation tool!");
    println!("This tool will generate self-play data for Viridithas.");
    println!("You can configure the data generation process by setting the following parameters:");
    println!("{options}");
    println!("you can also set positions_limit to limit the number of positions generated.");
    println!(
        "It is recommended that you do not set the number of threads to more than the number of logical cores on your CPU, as performance will suffer."
    );
    println!("(You have {} logical cores on your CPU)", num_cpus::get());
    println!("To set a parameter, type \"set <PARAM> <VALUE>\"");
    println!("To start data generation, type \"start\" or \"go\".");
}

fn config_loop(mut options: DataGenOptions) -> anyhow::Result<DataGenOptions> {
    println!();
    let mut user_input = String::new();
    loop {
        print!(">>> ");
        std::io::stdout()
            .flush()
            .with_context(|| "Failed to flush stdout.")?;
        user_input.clear();
        std::io::stdin()
            .read_line(&mut user_input)
            .with_context(|| "Failed to read line from stdin.")?;
        let mut user_input = user_input.split_whitespace();
        let Some(command) = user_input.next() else {
            continue;
        };
        if matches!(command, "start" | "go") {
            break;
        }
        if command != "set" {
            eprintln!(
                "Invalid command, supported commands are \"set <PARAM> <VALUE>\", \"start\", and \"go\""
            );
            continue;
        }
        let Some(param) = user_input.next() else {
            eprintln!(
                "Invalid command, supported commands are \"set <PARAM> <VALUE>\" and \"start\""
            );
            continue;
        };
        let Some(value) = user_input.next() else {
            eprintln!(
                "Invalid command, supported commands are \"set <PARAM> <VALUE>\" and \"start\""
            );
            continue;
        };
        match param {
            "num_games" => {
                if let Ok(num_games) = value.parse::<usize>() {
                    options.num_games = num_games;
                } else {
                    eprintln!("Invalid value for num_games, must be a positive integer");
                }
            }
            "num_threads" => {
                if let Ok(num_threads) = value.parse::<usize>() {
                    let num_cpus = num_cpus::get();
                    if num_threads > num_cpus {
                        eprintln!(
                            "Warning: The specified number of threads ({num_threads}) is greater than the number of available CPUs ({num_cpus})."
                        );
                    }
                    options.num_threads = num_threads;
                } else {
                    eprintln!("Invalid value for num_threads, must be a positive integer");
                }
            }
            "tablebases_path" => {
                let Ok(tablebases_path) = value.parse::<PathBuf>();
                if !tablebases_path.exists() {
                    eprintln!("Warning: The specified tablebases path does not exist.");
                } else if !tablebases_path.is_dir() {
                    eprintln!("Warning: The specified tablebases path is not a directory.");
                } else {
                    options.tablebases_path = Some(tablebases_path);
                }
            }
            "limit" => {
                let Some(limit_size) = user_input.next() else {
                    eprintln!("Trying to set limit, but only one token was provided");
                    eprintln!("Usage: \"set limit <NUMBER>\"");
                    eprintln!("Example: \"set limit 8000\" (sets the limit to 8000 nodes)");
                    continue;
                };
                let full_limit = value.to_string() + " " + limit_size;
                let limit = match full_limit.parse::<u64>() {
                    Ok(limit) => limit,
                    Err(e) => {
                        eprintln!("{e}");
                        continue;
                    }
                };
                options.nodes = limit;
            }
            "dfrc" => {
                if let Ok(dfrc) = value.parse::<bool>() {
                    options.generate_dfrc = dfrc;
                } else {
                    eprintln!("Invalid value for dfrc, must be a boolean");
                }
            }
            other => {
                eprintln!(
                    "Invalid parameter (\"{other}\"), supported parameters are \"num_games\", \"num_threads\", \"tablebases_path\", \"use_nnue\", and \"nodes\"."
                );
            }
        }
    }

    Ok(options)
}

impl Display for DataGenOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[+] Current data generation configuration:")?;
        writeln!(f, " |> num_games: {}", self.num_games)?;
        writeln!(f, " |> num_threads: {}", self.num_threads)?;
        writeln!(
            f,
            " |> tablebases_path: {}",
            self.tablebases_path
                .as_ref()
                .map_or_else(|| "None".into(), |path| path.to_string_lossy())
        )?;
        writeln!(f, " |> limit: {} nodes", self.nodes)?;
        writeln!(f, " |> dfrc: {}", self.generate_dfrc)?;
        if self.tablebases_path.is_none() {
            writeln!(
                f,
                "    ! Tablebases path not set - this will result in weaker data - are you sure you want to continue?"
            )?;
        }
        Ok(())
    }
}

/// Unpacks the variable-length game format into either bulletformat or marlinformat records,
/// filtering as it goes.
pub fn run_splat(
    input: &Path,
    output: &Path,
    cfg_path: Option<&Path>,
    marlinformat: bool,
    limit: Option<usize>,
) -> anyhow::Result<()> {
    // check that the input file exists
    if !input.try_exists()? {
        bail!("Input file does not exist.");
    }
    // check that the output does not exist
    if output.try_exists()? {
        bail!("Output file already exists.");
    }

    let filter = cfg_path.map_or_else(|| Ok(Filter::default()), Filter::from_path)?;

    // open the input file
    let input_file = File::open(input).with_context(|| "Failed to create input file")?;
    let mut input_buffer = BufReader::new(input_file);

    // open the output file
    let output_file = File::create(output).with_context(|| "Failed to create output file")?;
    let mut output_buffer = BufWriter::new(output_file);

    println!("Splatting...");
    print!("0 games splatted");
    let mut game_count = 0;
    let mut move_buffer = Vec::new();
    while let Ok(game) =
        dataformat::Game::deserialise_from(&mut input_buffer, std::mem::take(&mut move_buffer))
    {
        if marlinformat {
            game.splat_to_marlinformat(
                |packed_board| {
                    output_buffer
                        .write_all(&packed_board.as_bytes())
                        .with_context(|| "Failed to write PackedBoard into buffered writer.")
                },
                &filter,
            )?;
        } else {
            game.splat_to_bulletformat(
                |chess_board| {
                    // SAFETY: ChessBoard is composed entirely of integer types, which are safe to transmute into bytes.
                    let bytes = unsafe { std::mem::transmute::<ChessBoard, [u8; 32]>(chess_board) };
                    output_buffer.write_all(&bytes).with_context(
                        || "Failed to write bulletformat::ChessBoard into buffered writer.",
                    )
                },
                &filter,
            )?;
        }
        move_buffer = game.into_move_buffer();
        game_count += 1;
        if game_count % 2048 == 0 {
            print!("\r{game_count} games splatted");
            std::io::stdout()
                .flush()
                .with_context(|| "Failed to flush stdout.")?;
        }
        if limit.is_some_and(|limit| game_count >= limit) {
            break;
        }
    }
    println!("\r{game_count} games splatted.");

    output_buffer
        .flush()
        .with_context(|| "Failed to flush output buffer to file.")?;

    Ok(())
}

/// Unpacks the variable-length game format into a PGN file.
pub fn run_topgn(input: &Path, output: &Path, limit: Option<usize>) -> anyhow::Result<()> {
    // check that the input file exists
    if !input.try_exists()? {
        bail!("Input file does not exist.");
    }
    // check that the output does not exist
    if output.try_exists()? {
        bail!("Output file already exists.");
    }

    // open the input file
    let input_file = File::open(input).with_context(|| "Failed to create input file")?;
    let mut input_buffer = BufReader::new(input_file);

    // open the output file
    let output_file = File::create(output).with_context(|| "Failed to create output file")?;
    let mut output_buffer = BufWriter::new(output_file);

    let file_name = input
        .file_name()
        .with_context(|| "Failed to get filename.")?
        .to_string_lossy();
    let make_header = |outcome: WDL, fen: String| {
        format!(
            r#"[Event "datagen id {}"]
[Site "NA"]
[Date "NA"]
[White "Viridithas"]
[Black "Viridithas"]
[Result "{}"]
[FEN "{}"]"#,
            file_name,
            match outcome {
                WDL::Win => "1-0",
                WDL::Loss => "0-1",
                WDL::Draw => "1/2-1/2",
            },
            fen
        )
    };

    println!("Converting to PGN...");
    let mut move_buffer = Vec::new();
    let mut game_count = 0;
    while let Ok(game) =
        dataformat::Game::deserialise_from(&mut input_buffer, std::mem::take(&mut move_buffer))
    {
        let outcome = game.outcome();
        let mut board = game.initial_position();
        let header = make_header(outcome, board.to_string());
        writeln!(output_buffer, "{header}").unwrap();
        let mut fullmoves = 0;
        for mv in game.moves() {
            if fullmoves % 12 == 0 && board.turn() == Colour::White {
                writeln!(output_buffer).unwrap();
            }
            let san = board.san(mv).with_context(|| {
                format!(
                    "Failed to create SAN for move {} in position {board:X}.",
                    mv.display(CHESS960.load(Ordering::Relaxed))
                )
            })?;
            if board.turn() == Colour::White {
                write!(output_buffer, "{}. ", board.ply() / 2 + 1).unwrap();
            } else {
                fullmoves += 1;
            }
            write!(output_buffer, "{san} ").unwrap();
            board.make_move_simple(mv);
        }

        write!(
            output_buffer,
            "{}\n\n",
            match outcome {
                WDL::Win => "1-0",
                WDL::Loss => "0-1",
                WDL::Draw => "1/2-1/2",
            }
        )
        .unwrap();

        move_buffer = game.into_move_buffer();
        game_count += 1;
        if limit.is_some_and(|limit| game_count >= limit) {
            break;
        }
    }

    output_buffer
        .flush()
        .with_context(|| "Failed to flush output buffer to file.")?;

    Ok(())
}

/// Take a binpack, and write a new binpack with identical data but rescaled evaluations.
pub fn run_rescale(input: &Path, output: &Path, scale: f64) -> anyhow::Result<()> {
    // check that the input file exists
    if !input.try_exists()? {
        bail!("Input file does not exist.");
    }
    // check that the output does not exist
    if output.try_exists()? {
        bail!("Output file already exists.");
    }

    // open the input file
    let input_file = File::open(input)
        .with_context(|| format!("Failed to create input file: {}", input.display()))?;
    let input_buffer = BufReader::new(input_file);

    // open the output file
    let output_file = File::create(output)
        .with_context(|| format!("Failed to create output file: {}", output.display()))?;
    let output_buffer = BufWriter::new(output_file);

    println!("Rescaling evaluations by a factor of {scale}...");
    rescale_binpacks(scale, input_buffer, output_buffer)?;

    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
fn rescale_binpacks(
    scale: f64,
    mut input_buffer: impl BufRead,
    mut output_buffer: impl Write,
) -> Result<(), anyhow::Error> {
    let mut move_buffer = Vec::new();
    while let Ok(mut game) =
        dataformat::Game::deserialise_from(&mut input_buffer, std::mem::take(&mut move_buffer))
    {
        for (_, slot) in game.buffer_mut() {
            let value = i32::from(slot.get());
            let new_value = if is_decisive(value * 2) {
                value
            } else {
                (f64::from(value) * scale).round() as i32
            };
            if is_decisive(new_value) && !is_decisive(value) {
                eprintln!("[!] a network evaluation became decisive ({value} -> {new_value})");
            }
            let new_value: i16 = new_value.try_into().with_context(|| {
                format!("Failed to convert rescaled evaluation into i16: {new_value}.")
            })?;

            slot.set(new_value);
        }

        game.serialise_into(&mut output_buffer)
            .context("Failed to serialise game into output buffer.")?;

        move_buffer = game.into_move_buffer();
    }

    output_buffer
        .flush()
        .with_context(|| "Failed to flush output buffer to file.")?;

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
struct MaterialConfiguration {
    counts: [u8; 10],
}

impl MaterialConfiguration {
    fn men(&self) -> u8 {
        self.counts.iter().sum::<u8>() + 2 // add 2 for the kings
    }
}

impl Display for MaterialConfiguration {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        static CHARS: [char; 5] = ['P', 'N', 'B', 'R', 'Q'];
        // output strings like KRPPvKRP or KQvKRP
        write!(f, "K")?;
        for (i, pc) in self.counts[..5].iter().enumerate().rev() {
            for _ in 0..*pc {
                write!(f, "{}", CHARS[i])?;
            }
        }
        write!(f, "vK")?;
        for (i, pc) in self.counts[5..].iter().enumerate().rev() {
            for _ in 0..*pc {
                write!(f, "{}", CHARS[i])?;
            }
        }
        Ok(())
    }
}

impl From<&Board> for MaterialConfiguration {
    fn from(board: &Board) -> Self {
        let mut mc = Self::default();
        let white = board.state.bbs.colours[Colour::White];
        let black = board.state.bbs.colours[Colour::Black];
        for piece in PieceType::all().take(5) {
            let pieces = board.state.bbs.pieces[piece];
            let white_pieces = pieces & white;
            let black_pieces = pieces & black;
            mc.counts[piece.index()] = u8::try_from(white_pieces.count()).unwrap_or(u8::MAX);
            mc.counts[piece.index() + 5] = u8::try_from(black_pieces.count()).unwrap_or(u8::MAX);
        }
        // normalize the counts so that the white side has more material than the black side
        let ordering_key = |subslice: &[u8]| -> u64 {
            let count = u64::from(subslice.iter().sum::<u8>());
            let highest_piece = subslice
                .iter()
                .enumerate()
                .filter(|(_, v)| **v > 0)
                .next_back()
                .unwrap_or((0, &0))
                .0;
            count * 10 + highest_piece as u64
        };
        let (white, black) = mc.counts.split_at_mut(5);
        let white_key = ordering_key(white);
        let black_key = ordering_key(black);
        if black_key > white_key {
            // swap the counts
            white.swap_with_slice(black);
        }
        mc
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct DataSetStats {
    /// The total number of games in the dataset.
    games: usize,
    /// A histogram of opening evaluations (the evaluation of the first position in each game).
    opening_eval_counts: HashMap<i32, usize>,
    /// A histogram of game lengths (in ply).
    length_counts: HashMap<usize, usize>,
    /// A histogram of all evaluations in the dataset.
    eval_counts: HashMap<i32, usize>,
    /// A histogram of the number of pieces on the board across all positions in the dataset.
    piece_counts: HashMap<u8, usize>,
    /// A histogram of material configurations across all positions in the dataset.
    material_counts: HashMap<MaterialConfiguration, usize>,
    /// A histogram of the positions of the point-of-view king across all positions in the dataset.
    pov_king_positions: HashMap<Square, usize>,
}

/// Scans a variable-length game format file and prints statistics about it.
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
pub fn dataset_stats(dataset_path: &Path) -> anyhow::Result<()> {
    let mut move_buffer = Vec::new();
    let mut stats = DataSetStats::default();

    println!("Scanning dataset at {}", dataset_path.display());

    // because the format is variable-length, we can't know exactly our progress in terms of game-count.
    // we *can*, however, know how far we are through the file, so we use that as a progress indicator:
    print!("Progress: 0%");
    std::io::stdout()
        .flush()
        .with_context(|| "Failed to flush stdout!")?;
    // get the file size
    let file_size = dataset_path
        .metadata()
        .with_context(|| {
            format!(
                "Failed to get metadata for {}",
                dataset_path.to_string_lossy()
            )
        })?
        .len();

    let mut reader =
        BufReader::new(File::open(dataset_path).with_context(|| "Failed to open dataset.")?);

    while let Ok(game) =
        dataformat::Game::deserialise_from(&mut reader, std::mem::take(&mut move_buffer))
    {
        stats.games += 1;
        *stats.length_counts.entry(game.len()).or_default() += 1;
        let mut idx = 0;
        game.visit_positions(|position, evaluation| {
            if idx == 0 {
                *stats.opening_eval_counts.entry(evaluation).or_default() += 1;
            }
            idx += 1;
            *stats.eval_counts.entry(evaluation).or_default() += 1;
            *stats
                .piece_counts
                .entry(u8::try_from(position.state.bbs.occupied().count()).unwrap_or(u8::MAX))
                .or_default() += 1;
            *stats
                .material_counts
                .entry(MaterialConfiguration::from(position))
                .or_default() += 1;
            *stats
                .pov_king_positions
                .entry(position.state.bbs.king_sq(Colour::White))
                .or_default() += 1;
            *stats
                .pov_king_positions
                .entry(position.state.bbs.king_sq(Colour::Black).flip_rank())
                .or_default() += 1;
        });
        move_buffer = game.into_move_buffer();

        // print progress
        if stats.games % 1024 == 0 {
            let progress = reader
                .stream_position()
                .with_context(|| "Failed to get stream position.")?;
            let percentage = progress * 100 / file_size;
            print!("\rProgress: {percentage}%");
            std::io::stdout()
                .flush()
                .with_context(|| "Failed to flush stdout!")?;
        }
    }
    println!("\rProgress: 100%");

    println!("Statistics for dataset at {}", dataset_path.display());
    println!("Number of games: {}", stats.games);
    println!("Writing length counts to length_counts.csv");
    let mut length_counts = stats.length_counts.iter().collect::<Vec<_>>();
    length_counts.sort_unstable_by_key(|(length, _)| *length);
    let mut length_counts_file = BufWriter::new(File::create("length_counts.csv")?);
    writeln!(length_counts_file, "length,count")?;
    for (length, count) in length_counts {
        writeln!(length_counts_file, "{length},{count}")?;
    }
    length_counts_file.flush()?;
    println!("Writing eval counts to eval_counts.csv");
    let mut eval_counts = stats.eval_counts.into_iter().collect::<Vec<_>>();
    eval_counts.sort_unstable_by_key(|(eval, _)| *eval);
    let mut eval_counts_file = BufWriter::new(File::create("eval_counts.csv")?);
    writeln!(eval_counts_file, "eval,count")?;
    for &(eval, count) in &eval_counts {
        writeln!(eval_counts_file, "{eval},{count}")?;
    }
    eval_counts_file.flush()?;
    println!("Writing opening eval counts to opening_eval_counts.csv");
    let mut opening_eval_counts = stats.opening_eval_counts.into_iter().collect::<Vec<_>>();
    opening_eval_counts.sort_unstable_by_key(|(eval, _)| *eval);
    let mut eval_counts_file = BufWriter::new(File::create("opening_eval_counts.csv")?);
    writeln!(eval_counts_file, "eval,count")?;
    for (eval, count) in opening_eval_counts {
        writeln!(eval_counts_file, "{eval},{count}")?;
    }
    eval_counts_file.flush()?;
    println!("Writing piece counts to piece_counts.csv");
    let mut piece_counts = stats.piece_counts.into_iter().collect::<Vec<_>>();
    piece_counts.sort_unstable_by_key(|(count, _)| *count);
    let mut piece_counts_file = BufWriter::new(File::create("piece_counts.csv")?);
    writeln!(piece_counts_file, "men,count")?;
    for (men, count) in piece_counts {
        writeln!(piece_counts_file, "{men},{count}")?;
    }
    piece_counts_file.flush()?;
    println!("Writing material counts to material_counts.csv");
    let material_counts = stats.material_counts.into_iter().collect::<Vec<_>>();
    let mut material_counts_file = BufWriter::new(File::create("material_counts.csv")?);
    writeln!(material_counts_file, "material,count")?;
    for (mc, count) in material_counts {
        writeln!(material_counts_file, "{mc},{count}")?;
    }
    material_counts_file.flush()?;
    println!("Writing PoV king positions to pov_king_positions.csv");
    let pov_king_positions = stats.pov_king_positions.into_iter().collect::<Vec<_>>();
    let mut pov_king_positions_file = BufWriter::new(File::create("pov_king_positions.csv")?);
    writeln!(pov_king_positions_file, "square,count")?;
    for (sq, count) in pov_king_positions {
        writeln!(pov_king_positions_file, "{sq},{count}", sq = sq.index())?;
    }
    pov_king_positions_file.flush()?;

    let total_position_count = stats
        .length_counts
        .iter()
        .map(|(k, v)| k * v)
        .sum::<usize>() as u128;
    #[allow(clippy::cast_precision_loss)]
    let mean_game_len = ((total_position_count * 1000) / stats.games as u128) as f64 / 1000.0;
    println!("Mean game length: {mean_game_len}");
    println!("Total position count: {total_position_count}");

    let usable_evals = eval_counts
        .into_iter()
        .filter(|(eval, _)| !is_decisive(*eval * 2))
        .collect::<Vec<_>>();

    let total = usable_evals.iter().map(|(_, c)| *c as u128).sum::<u128>() as f64;
    let mean_eval = usable_evals
        .iter()
        .map(|(eval, c)| i128::from(*eval) * (*c as i128))
        .sum::<i128>() as f64
        / total;

    println!("Mean eval: {mean_eval:.2}");

    let mean_abs_eval = usable_evals
        .iter()
        .map(|(eval, c)| u128::from(eval.unsigned_abs()) * (*c as u128))
        .sum::<u128>() as f64
        / total;

    println!("Mean absolute eval: {mean_abs_eval:.2}");

    let variance = usable_evals
        .iter()
        .map(|(eval, c)| {
            let diff = i128::from(*eval) - mean_eval as i128;
            diff * diff * (*c as i128)
        })
        .sum::<i128>() as f64
        / total;

    let stddev = variance.sqrt();
    println!("Eval standard deviation: {stddev:.2}");

    Ok(())
}

/// Scans one or more variable-length game format files and prints the position counts.
pub fn dataset_count(path: &Path) -> anyhow::Result<()> {
    let paths = if path.is_dir() {
        fs::read_dir(path).map_or_else(
            |_| Vec::new(),
            |dir| {
                dir.filter_map(|entry| {
                    entry.ok().and_then(|entry| {
                        let path = entry.path();
                        if path.is_file() { Some(path) } else { None }
                    })
                })
                .collect()
            },
        )
    } else {
        vec![path.to_owned()]
    };

    if paths.is_empty() {
        bail!("No files found at {}", path.display());
    }

    let mpl = paths
        .iter()
        .map(|path| path.display().to_string().len())
        .max()
        .unwrap();
    let stdout_lock = Mutex::new(());
    let stdout_lock = &stdout_lock;

    let filter = &Filter::default();
    let (total_count, filtered_count, pass_count_buckets) = std::thread::scope(
        |s| -> anyhow::Result<(u64, u64, Vec<u64>)> {
            let mut thread_handles = Vec::new();
            for path in paths {
                thread_handles.push(s.spawn(move || -> anyhow::Result<(u64, u64, Vec<u64>)> {
                let file = File::open(&path)?;
                let len = file.metadata().with_context(|| "Failed to get file metadata!")?.len();
                let mut reader = BufReader::new(file);
                let mut count = 0u64;
                let mut filtered = 0u64;
                let mut pass_count_buckets = vec![0u64; Game::MAX_SPLATTABLE_GAME_SIZE];
                let mut move_buffer = Vec::new();
                loop {
                    match dataformat::Game::deserialise_from(&mut reader, std::mem::take(&mut move_buffer)) {
                        Ok(game) => {
                            count += game.len() as u64;
                            let pass_count = game.filter_pass_count(filter);
                            filtered += pass_count;
                            pass_count_buckets[usize::try_from(pass_count).unwrap().min(Game::MAX_SPLATTABLE_GAME_SIZE - 1)] += 1;
                            move_buffer = game.into_move_buffer();
                        }
                        Err(error) => {
                            match error.kind() {
                                std::io::ErrorKind::UnexpectedEof => {}
                                _ => eprintln!("[WARN] dataset_count encountered an unexpected error wile reading {file}: {error}\n[WARN] this occured at an offset of {:?} into the file (but probably earlier than this, as we use buffered IO)\n[WARN] for reference, {file} is {} bytes long.", reader.into_inner().stream_position(), len, file = path.file_name().map_or(Cow::Borrowed("<???>"), |oss| oss.to_string_lossy()))
                            }
                            break;
                        }
                    }
                }
                let lock = stdout_lock.lock().map_err(|_| anyhow!("Failed to lock mutex."))?;
                println!("{:mpl$}: {} | {}", path.display(), count, filtered);
                std::mem::drop(lock);
                Ok((count, filtered, pass_count_buckets))
            }));
            }
            let (mut total_count, mut filtered_count) = (0, 0);
            let mut total_pass_count_buckets = vec![0u64; Game::MAX_SPLATTABLE_GAME_SIZE];
            for handle in thread_handles {
                let (count, filtered, pass_count_buckets) = handle
                    .join()
                    .map_err(|_| anyhow!("Thread panicked."))
                    .with_context(|| "Failed to join processing thread")?
                    .with_context(|| "A processing job failed")?;
                total_count += count;
                filtered_count += filtered;
                for (i, count) in pass_count_buckets.into_iter().enumerate() {
                    total_pass_count_buckets[i] += count;
                }
            }
            Ok((total_count, filtered_count, total_pass_count_buckets))
        },
    )?;

    println!();
    println!("Total: {total_count}");
    println!("Total that pass the filter: {filtered_count}");
    for (i, c) in pass_count_buckets.chunks(16).enumerate() {
        println!(
            "Games with {:3} to {:3} filtered positions: {}",
            i * 16,
            i * 16 + 15,
            c.iter().sum::<u64>()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{chess::CHESS960, datagen::dataformat, evaluation::is_decisive};

    #[test]
    fn test_scaling() {
        CHESS960.store(true, std::sync::atomic::Ordering::Relaxed);

        let input_binpacks = include_bytes!("../embeds/test-vf.bin");
        let mut input_cursor = std::io::Cursor::new(input_binpacks);
        let mut output_cursor = std::io::Cursor::new(Vec::new());
        super::rescale_binpacks(0.5, &mut input_cursor, &mut output_cursor).unwrap();

        // deserialise the output and check that the evaluations are halved
        input_cursor.set_position(0);
        output_cursor.set_position(0);
        let mut input_move_buffer = Vec::new();
        let mut output_move_buffer = Vec::new();
        while let Ok(input_game) = dataformat::Game::deserialise_from(
            &mut input_cursor,
            std::mem::take(&mut input_move_buffer),
        ) {
            let output_game = dataformat::Game::deserialise_from(
                &mut output_cursor,
                std::mem::take(&mut output_move_buffer),
            )
            .unwrap();
            let input_slots = input_game.buffer();
            let output_slots = output_game.buffer();
            assert_eq!(input_slots.len(), output_slots.len());
            for ((m1, input_slot), (m2, output_slot)) in input_slots.iter().zip(output_slots.iter())
            {
                assert_eq!(m1, m2);
                let input_eval = i32::from(input_slot.get());
                let output_eval = i32::from(output_slot.get());
                #[allow(clippy::cast_possible_truncation)]
                let expected_output_eval = if is_decisive(input_eval) {
                    input_eval
                } else {
                    (f64::from(input_eval) * 0.5).round() as i32
                };
                assert_eq!(
                    output_eval, expected_output_eval,
                    "Input eval: {input_eval}, Output eval: {output_eval}, Expected output eval: {expected_output_eval}"
                );
            }
        }
    }
}
