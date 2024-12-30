#![allow(dead_code)]

mod dataformat;

use std::{
    borrow::Cow,
    cmp::Reverse,
    collections::HashMap,
    fmt::{Display, Formatter},
    fs::{self, File},
    hash::Hash,
    io::{BufReader, BufWriter, Seek, Write},
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex,
    },
    time::Instant,
};

use anyhow::{anyhow, bail, Context};
use bulletformat::ChessBoard;
use dataformat::Filter;
use rand::Rng;

use crate::{
    chess::{
        board::{
            evaluation::{is_game_theoretic_score, is_mate_score},
            Board, DrawType, GameOutcome, WinType,
        },
        piece::{Colour, PieceType},
    },
    datagen::dataformat::Game,
    nnue::network::NNUEParams,
    searchinfo::SearchInfo,
    tablebases::{self, probe::WDL},
    threadlocal::ThreadData,
    timemgmt::{SearchLimit, TimeManager},
    transpositiontable::TT,
    uci::{CHESS960, SYZYGY_ENABLED, SYZYGY_PATH},
    util::{Square, MEGABYTE},
};

const MIN_SAVE_PLY: usize = 16;
const MAX_RNG_PLY: usize = 24;

static FENS_GENERATED: AtomicU64 = AtomicU64::new(0);
static STOP_GENERATION: AtomicBool = AtomicBool::new(false);

/// Whether to limit searches by depth or by nodes.
#[derive(Clone, Debug, Hash)]
enum DataGenLimit {
    Depth(i32),
    Nodes(u64),
}

/// Configuration options for Viri's self-play data generation.
#[derive(Clone, Debug, Hash)]
struct DataGenOptions {
    // The number of games to generate.
    num_games: usize,
    // The number of threads to use.
    num_threads: usize,
    // The (optional) path to the directory containing syzygy endgame tablebases.
    tablebases_path: Option<PathBuf>,
    // The depth or node limit for searches.
    limit: DataGenLimit,
    // Whether to generate DFRC data.
    generate_dfrc: bool,
    // log level
    log_level: u8,
}

/// Builder for datagen options.
pub struct DataGenOptionsBuilder {
    // The number of games to generate.
    pub num_games: usize,
    // The number of threads to use.
    pub num_threads: usize,
    // The (optional) path to the directory containing syzygy endgame tablebases.
    pub tablebases_path: Option<PathBuf>,
    // The depth or node limit for searches.
    pub use_depth: bool,
    // Whether to generate DFRC data.
    pub generate_dfrc: bool,
}

impl DataGenOptionsBuilder {
    fn build(self) -> DataGenOptions {
        DataGenOptions {
            num_games: self.num_games,
            num_threads: self.num_threads,
            tablebases_path: self.tablebases_path,
            limit: if self.use_depth {
                DataGenLimit::Depth(8)
            } else {
                DataGenLimit::Nodes(5000)
            },
            generate_dfrc: self.generate_dfrc,
            log_level: 1,
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
            limit: DataGenLimit::Depth(8),
            generate_dfrc: true,
            log_level: 1,
        }
    }

    /// Gives a summarised string representation of the options.
    fn summary(&self) -> String {
        format!(
            "{}g-{}t-{}-{}-{}",
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
            match self.limit {
                DataGenLimit::Depth(depth) => format!("d{depth}"),
                DataGenLimit::Nodes(nodes) => format!("n{nodes}"),
            }
        )
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

    if options.log_level > 0 {
        println!("Starting data generation with the following configuration:");
        println!("{options}");
        if options.num_games % options.num_threads != 0 {
            println!(
                "Warning: The number of games is not evenly divisible by the number of threads,"
            );
            println!(
                "this will result in {} games being omitted.",
                options.num_games % options.num_threads
            );
        }
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
        if options.log_level > 0 {
            println!("Syzygy tablebases enabled.");
        }
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
    if options.log_level > 0 {
        println!("This run will be saved to the directory \"data/{run_id}\"");
        println!("Each thread will save its data to a separate file in this directory.");
    }

    // create the directory for the data
    let data_dir = PathBuf::from("data").join(run_id);
    std::fs::create_dir_all(&data_dir).with_context(|| "Failed to create data directory")?;

    let mut counters = Vec::new();
    std::thread::scope(|s| {
        let thread_handles = (0..options.num_threads)
            .map(|id| {
                let opt_ref = &options;
                let path_ref = &data_dir;
                let nnue_params_ref = &nnue_params;
                s.spawn(move || generate_on_thread(id, opt_ref, path_ref, nnue_params_ref))
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

    if options.log_level > 0 {
        println!("Done!");
    }

    let counters = counters.into_iter().reduce(|mut acc, e| {
        for (key, value) in e {
            *acc.entry(key).or_insert(0) += value;
        }
        acc
    });

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
    counters.sort_unstable_by_key(|(_, &value)| Reverse(value));
    for (&key, &value) in counters {
        eprintln!(
            "{key:?}: {value} ({percentage}%)",
            percentage = (value as f64 / total as f64 * 100.0).round()
        );
    }
}

#[allow(
    clippy::cognitive_complexity,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]
fn generate_on_thread(
    id: usize,
    options: &DataGenOptions,
    data_dir: &Path,
    nnue_params: &NNUEParams,
) -> anyhow::Result<HashMap<GameOutcome, u64>> {
    // this rng is different between each thread
    // (https://rust-random.github.io/book/guide-parallel.html)
    // so no worries :3
    let mut rng = rand::thread_rng();
    let mut board = Board::default();
    let mut tt = TT::new();
    tt.resize(16 * MEGABYTE);
    let mut thread_data = ThreadData::new(0, &board, tt.view(), nnue_params);
    let stopped = AtomicBool::new(false);
    let time_manager = TimeManager::default_with_limit(match options.limit {
        DataGenLimit::Depth(depth) => SearchLimit::Depth(depth),
        DataGenLimit::Nodes(nodes) => SearchLimit::SoftNodes {
            soft_limit: nodes,
            hard_limit: nodes * 8,
        },
    });
    let nodes = AtomicU64::new(0);
    let mut info = SearchInfo {
        time_manager,
        print_to_stdout: false,
        ..SearchInfo::new(&stopped, &nodes)
    };

    let n_games_to_run = std::cmp::max(options.num_games / options.num_threads, 1);

    let mut output_file = File::create(data_dir.join(format!("thread_{id}.bin")))
        .with_context(|| "Failed to create output file.")?;
    let mut output_buffer = BufWriter::new(&mut output_file);

    let mut counters = [
        (GameOutcome::WhiteWin(WinType::Mate), 0),
        (GameOutcome::WhiteWin(WinType::TB), 0),
        (GameOutcome::WhiteWin(WinType::Adjudication), 0),
        (GameOutcome::BlackWin(WinType::Mate), 0),
        (GameOutcome::BlackWin(WinType::TB), 0),
        (GameOutcome::BlackWin(WinType::Adjudication), 0),
        (GameOutcome::Draw(DrawType::FiftyMoves), 0),
        (GameOutcome::Draw(DrawType::Repetition), 0),
        (GameOutcome::Draw(DrawType::Stalemate), 0),
        (GameOutcome::Draw(DrawType::InsufficientMaterial), 0),
        (GameOutcome::Draw(DrawType::TB), 0),
        (GameOutcome::Draw(DrawType::Adjudication), 0),
    ]
    .into_iter()
    .collect::<HashMap<_, _>>();

    let start = Instant::now();
    'generation_main_loop: for game in 0..n_games_to_run {
        // report progress
        if id == 0 && game % 8 == 0 && options.log_level > 0 && game > 0 {
            let percentage = game * 100_000 / n_games_to_run;
            let percentage = percentage as f64 / 1000.0;
            let time_per_game = start.elapsed().as_secs_f64() / game as f64;
            let games_to_go = n_games_to_run as f64 - game as f64;
            let time_remaining = games_to_go * time_per_game;
            eprintln!("[+] Main thread: Generated {game} games ({percentage:.1}%). Time per game: {time_per_game:.2} seconds.");
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
            if game % 1024 == 0 {
                eprintln!("Game stats for main thread:");
                print_game_stats(&counters);
            }
        }
        // reset everything: board, thread data, tt, search info
        if options.generate_dfrc {
            board.set_dfrc_idx(rand::Rng::gen_range(&mut rng, 0..960 * 960));
        } else {
            board.set_startpos();
        }
        thread_data
            .nnue
            .reinit_from(&board, thread_data.nnue_params);
        tt.clear(1);
        info.set_up_for_search();
        // generate game
        if options.log_level > 1 {
            eprintln!("Generating game {game}...");
        }
        // STEP 1: make random moves for variety
        if options.log_level > 2 {
            eprintln!("Making random moves...");
        }
        // pick either 8 or 9 random moves (to balance out the win/loss/draw ratio)
        let max = if rng.gen_bool(0.5) { 8 } else { 9 };
        for _ in 0..max {
            let res = board.make_random_move(&mut rng, &mut thread_data);
            if res.is_none() {
                if options.log_level > 2 {
                    eprintln!("Reached a position with no legal moves, skipping...");
                }
                continue 'generation_main_loop;
            }
            if board.outcome() != GameOutcome::Ongoing {
                if options.log_level > 2 {
                    eprintln!("Game ended early, skipping...");
                }
                continue 'generation_main_loop;
            }
        }
        // STEP 2: evaluate the exit position with reasonable depth
        // to make sure that it isn't silly.
        if options.log_level > 2 {
            eprintln!("Evaluating position...");
        }
        let temp_limit = info.time_manager.limit().clone();
        info.time_manager.set_limit(SearchLimit::Depth(10));
        let (eval, _) =
            board.search_position(&mut info, std::array::from_mut(&mut thread_data), tt.view());
        info.time_manager.set_limit(temp_limit);
        if eval.abs() > 1000 {
            if options.log_level > 2 {
                eprintln!("Position is too good or too bad, skipping...");
            }
            // if the position is too good or too bad, we don't want it
            continue 'generation_main_loop;
        }
        let mut game = Game::new(&board);
        // STEP 3: play out to the end of the game
        if options.log_level > 2 {
            eprintln!("Playing out game...");
        }
        let mut win_adj_counter = 0;
        let mut draw_adj_counter = 0;
        let outcome = loop {
            let outcome = board.outcome();
            if outcome != GameOutcome::Ongoing {
                break outcome;
            }
            if options.tablebases_path.is_some() {
                if let Some(wdl) = tablebases::probe::get_wdl_white(&board) {
                    break match wdl {
                        WDL::Win => GameOutcome::WhiteWin(WinType::TB),
                        WDL::Loss => GameOutcome::BlackWin(WinType::TB),
                        WDL::Draw => GameOutcome::Draw(DrawType::TB),
                    };
                }
            }
            tt.increase_age();

            let (score, best_move) =
                board.search_position(&mut info, std::array::from_mut(&mut thread_data), tt.view());

            let Some(best_move) = best_move else {
                println!("[WARNING!] search returned a null move as the best move!");
                println!("[WARNING!] this occurred in position {board}");
                continue 'generation_main_loop;
            };

            game.add_move(
                best_move,
                score
                    .try_into()
                    .with_context(|| "Failed to convert score into eval.")?,
            );

            let abs_score = score.abs();
            if abs_score >= 2000 {
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
                let outcome = if score > 0 {
                    GameOutcome::WhiteWin(WinType::Adjudication)
                } else {
                    GameOutcome::BlackWin(WinType::Adjudication)
                };
                break outcome;
            }
            if draw_adj_counter >= 12 {
                break GameOutcome::Draw(DrawType::Adjudication);
            }
            if is_game_theoretic_score(score) {
                // if the score is game theoretic, we don't want to play out the rest of the game
                let is_mate = is_mate_score(score);
                break match (score.signum(), is_mate) {
                    (1, false) => GameOutcome::WhiteWin(WinType::TB),
                    (-1, false) => GameOutcome::BlackWin(WinType::TB),
                    (1, true) => GameOutcome::WhiteWin(WinType::Mate),
                    (-1, true) => GameOutcome::BlackWin(WinType::Mate),
                    _ => unreachable!(),
                };
            }

            board.make_move(best_move, &mut thread_data);
        };
        if options.log_level > 2 {
            eprintln!("Game is over, outcome: {outcome:?}");
        }
        assert_ne!(outcome, GameOutcome::Ongoing, "Game should be over by now.");
        // STEP 4: write the game to the output file
        if options.log_level > 2 {
            eprintln!("Writing {} moves to output file...", game.len());
        }
        let count = game.len();
        // update with outcome
        game.set_outcome(outcome);

        // write to file
        game.serialise_into(&mut output_buffer)
            .with_context(|| "Failed to serialise game into output buffer.")?;

        // increment the counter
        FENS_GENERATED.fetch_add(count as u64, Ordering::SeqCst);

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

fn show_boot_info(options: &DataGenOptions) {
    if options.log_level > 0 {
        println!("Welcome to Viri's data generation tool!");
        println!("This tool will generate self-play data for Viridithas.");
        println!(
            "You can configure the data generation process by setting the following parameters:"
        );
        println!("{options}");
        println!("you can also set positions_limit to limit the number of positions generated.");
        println!("It is recommended that you do not set the number of threads to more than the number of logical cores on your CPU, as performance will suffer.");
        println!("(You have {} logical cores on your CPU)", num_cpus::get());
        println!("To set a parameter, type \"set <PARAM> <VALUE>\"");
        println!("To start data generation, type \"start\" or \"go\".");
    }
}

fn config_loop(mut options: DataGenOptions) -> anyhow::Result<DataGenOptions> {
    #![allow(clippy::option_if_let_else, clippy::too_many_lines)]
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
            eprintln!("Invalid command, supported commands are \"set <PARAM> <VALUE>\", \"start\", and \"go\"");
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
                        eprintln!("Warning: The specified number of threads ({num_threads}) is greater than the number of available CPUs ({num_cpus}).");
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
                    eprintln!("Usage: \"set limit <TYPE> <NUMBER>\"");
                    eprintln!("Example: \"set limit depth 8\" (sets the limit to 8 plies)");
                    continue;
                };
                let full_limit = value.to_string() + " " + limit_size;
                let limit = match full_limit.parse::<DataGenLimit>() {
                    Ok(limit) => limit,
                    Err(e) => {
                        eprintln!("{e}");
                        continue;
                    }
                };
                options.limit = limit;
            }
            "dfrc" => {
                if let Ok(dfrc) = value.parse::<bool>() {
                    options.generate_dfrc = dfrc;
                } else {
                    eprintln!("Invalid value for dfrc, must be a boolean");
                }
            }
            "log_level" => {
                let log_level = match value.parse::<u8>() {
                    Ok(log_level) => log_level,
                    Err(e) => {
                        eprintln!("{e}");
                        continue;
                    }
                };
                options.log_level = log_level;
            }
            other => {
                eprintln!("Invalid parameter (\"{other}\"), supported parameters are \"num_games\", \"num_threads\", \"tablebases_path\", \"use_nnue\", \"limit\", and \"log_level\"");
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
        writeln!(
            f,
            " |> limit: {}",
            match self.limit {
                DataGenLimit::Depth(depth) => format!("depth {depth}"),
                DataGenLimit::Nodes(nodes) => format!("nodes {nodes}"),
            }
        )?;
        writeln!(f, " |> dfrc: {}", self.generate_dfrc)?;
        writeln!(f, " |> log_level: {}", self.log_level)?;
        if self.tablebases_path.is_none() {
            writeln!(
                f,
                "    ! Tablebases path not set - this will result in weaker data - are you sure you want to continue?"
            )?;
        }
        Ok(())
    }
}

impl FromStr for DataGenLimit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        #![allow(clippy::cast_possible_truncation)]
        let (limit_type, limit_value) = s
            .split_once(' ')
            .ok_or_else(|| format!("Invalid limit, no space: {s}"))?;
        let limit_value: u64 = limit_value
            .parse()
            .map_err(|_| format!("Invalid limit value: {limit_value}"))?;
        match limit_type {
            "depth" => {
                if limit_value > i32::MAX as u64 {
                    return Err(format!("Depth limit too large: {limit_value}"));
                }
                Ok(Self::Depth(limit_value as i32))
            }
            "nodes" => Ok(Self::Nodes(limit_value)),
            _ => Err(format!("Invalid limit type: {limit_type}")),
        }
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
    let mut rng = rand::thread_rng();

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
                &mut rng,
            )?;
        } else {
            game.splat_to_bulletformat(
                |chess_board| {
                    // SAFETY: ChessBoard is composed entirely of integer types, which are safe to transmute into bytes.
                    let bytes = unsafe { std::mem::transmute::<ChessBoard, [u8; 32]>(chess_board) };
                    output_buffer.write_all(&bytes).with_context(|| {
                        "Failed to write bulletformat::ChessBoard into buffered writer."
                    })
                },
                &filter,
                &mut rng,
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
        if let Some(limit) = limit {
            if game_count >= limit {
                break;
            }
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
        writeln!(output_buffer, "{header}\n").unwrap();
        let mut fullmoves = 0;
        for mv in game.moves() {
            if fullmoves % 12 == 0 && board.turn() == Colour::White {
                writeln!(output_buffer).unwrap();
            }
            let san = board.san(mv).with_context(|| {
                format!("Failed to create SAN for move {mv} in position {board:X}.")
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
        if let Some(limit) = limit {
            if game_count >= limit {
                break;
            }
        }
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

#[allow(clippy::fallible_impl_from)]
impl From<&Board> for MaterialConfiguration {
    fn from(board: &Board) -> Self {
        let mut mc = Self::default();
        let white = board.pieces.occupied_co(Colour::White);
        let black = board.pieces.occupied_co(Colour::Black);
        for piece in PieceType::all().take(5) {
            let pieces = board.pieces.of_type(piece);
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
                .last()
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
    games: usize,
    length_counts: HashMap<usize, usize>,
    eval_counts: HashMap<i32, usize>,
    piece_counts: HashMap<u8, usize>,
    material_counts: HashMap<MaterialConfiguration, usize>,
    pov_king_positions: HashMap<Square, usize>,
}

/// Scans a variable-length game format file and prints statistics about it.
#[allow(clippy::too_many_lines)]
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
        game.visit_positions(|position, evaluation| {
            *stats.eval_counts.entry(evaluation).or_default() += 1;
            *stats
                .piece_counts
                .entry(u8::try_from(position.pieces.occupied().count()).unwrap_or(u8::MAX))
                .or_default() += 1;
            *stats
                .material_counts
                .entry(MaterialConfiguration::from(position))
                .or_default() += 1;
            *stats
                .pov_king_positions
                .entry(position.king_sq(Colour::White))
                .or_default() += 1;
            *stats
                .pov_king_positions
                .entry(position.king_sq(Colour::Black).flip_rank())
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
    for (eval, count) in eval_counts {
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

    #[allow(clippy::cast_precision_loss)]
    let mean_game_len = ((stats
        .length_counts
        .iter()
        .map(|(k, v)| k * v)
        .sum::<usize>() as u128
        * 1000)
        / stats.games as u128) as f64
        / 1000.0;
    println!("Mean game length: {mean_game_len}");

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
                        if path.is_file() {
                            Some(path)
                        } else {
                            None
                        }
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
