#![allow(dead_code)]

mod dataformat;

use std::{
    cmp::Reverse,
    collections::HashMap,
    fmt::{Display, Formatter},
    fs::File,
    hash::Hash,
    io::{BufReader, BufWriter, Seek, Write},
    path::{Path, PathBuf},
    str::FromStr,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};

use bulletformat::ChessBoard;
use rand::Rng;

use crate::{
    board::{
        evaluation::{is_game_theoretic_score, MINIMUM_MATE_SCORE},
        Board, GameOutcome,
    },
    chessmove::Move,
    datagen::dataformat::Game,
    piece::{Colour, PieceType},
    searchinfo::SearchInfo,
    tablebases::{self, probe::WDL},
    threadlocal::ThreadData,
    timemgmt::{SearchLimit, TimeManager},
    transpositiontable::TT,
    uci::{CHESS960, SYZYGY_ENABLED, SYZYGY_PATH},
    util::{depth::Depth, Square, MEGABYTE},
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
    // Whether to use NNUE evaluation during self-play.
    use_nnue: bool,
    // The depth or node limit for searches.
    limit: DataGenLimit,
    // Whether to generate DFRC data.
    generate_dfrc: bool,
    // log level
    log_level: u8,
    // position count limit
    position_count_limit: Option<u64>,
}

impl DataGenOptions {
    /// Creates a new `DataGenOptions` instance.
    const fn new() -> Self {
        Self {
            num_games: 100,
            num_threads: 1,
            tablebases_path: None,
            use_nnue: true,
            limit: DataGenLimit::Depth(8),
            generate_dfrc: true,
            log_level: 1,
            position_count_limit: None,
        }
    }

    /// Gives a summarised string representation of the options.
    fn summary(&self) -> String {
        format!(
            "{}g-{}t-{}-{}-{}-{}",
            self.num_games,
            self.num_threads,
            self.tablebases_path.as_ref().map_or_else(
                || "no_tb".into(),
                |tablebases_path| tablebases_path.to_string_lossy()
            ),
            if self.use_nnue { "nnue" } else { "hce" },
            if self.generate_dfrc { "dfrc" } else { "classical" },
            match self.limit {
                DataGenLimit::Depth(depth) => format!("d{depth}"),
                DataGenLimit::Nodes(nodes) => format!("n{nodes}"),
            }
        )
    }
}

pub fn gen_data_main(cli_config: Option<&str>) {
    assert!(!cfg!(not(feature = "datagen")), "Data generation is not enabled, please enable the 'datagen' feature to use this functionality.");

    ctrlc::set_handler(move || {
        STOP_GENERATION.store(true, Ordering::SeqCst);
        println!("Stopping generation, please don't force quit.");
    })
    .expect("Failed to set Ctrl-C handler");

    let options: DataGenOptions = cli_config.map_or_else(|| {
            let options = DataGenOptions::new();
            show_boot_info(&options);
            config_loop(options)
        }, |s| s.parse().expect("Failed to parse CLI config, expected short def string (e.g. '100g-2t-<TBPATH>-nnue-d8')"));

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
        *SYZYGY_PATH.lock().unwrap() = tb_path.to_string();
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
    let run_id =
        format!("run_{}_{}", chrono::Utc::now().format("%Y-%m-%d_%H-%M-%S"), options.summary());
    if options.log_level > 0 {
        println!("This run will be saved to the directory \"data/{run_id}\"");
        println!("Each thread will save its data to a separate file in this directory.");
    }

    // create the directory for the data
    let data_dir = PathBuf::from("data").join(run_id);
    std::fs::create_dir_all(&data_dir).expect("Failed to create data directory");

    let mut counters = Vec::new();
    std::thread::scope(|s| {
        let thread_handles = (0..options.num_threads)
            .map(|id| {
                let opt_ref = &options;
                let path_ref = &data_dir;
                s.spawn(move || generate_on_thread(id, opt_ref, path_ref))
            })
            .collect::<Vec<_>>();
        for handle in thread_handles {
            counters.push(handle.join().unwrap());
        }
    });

    if options.log_level > 0 {
        println!("Done!");
    }

    let counters = counters.into_iter().reduce(|mut a, b| {
        for (key, value) in b {
            *a.entry(key).or_insert(0) += value;
        }
        a
    });

    if let Some(counters) = counters {
        print_game_stats(&counters);
    }
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

#[allow(clippy::cognitive_complexity)]
fn generate_on_thread(
    id: usize,
    options: &DataGenOptions,
    data_dir: &Path,
) -> HashMap<GameOutcome, u64> {
    #![allow(clippy::cast_precision_loss, clippy::too_many_lines, clippy::cast_possible_truncation)]
    // this rng is different between each thread
    // (https://rust-random.github.io/book/guide-parallel.html)
    // so no worries :3
    let mut rng = rand::thread_rng();
    let mut board = Board::default();
    let mut tt = TT::new();
    tt.resize(16 * MEGABYTE);
    let mut thread_data = ThreadData::new(0, &board, tt.view());
    let stopped = AtomicBool::new(false);
    let time_manager = TimeManager::default_with_limit(match options.limit {
        DataGenLimit::Depth(depth) => SearchLimit::Depth(Depth::new(depth)),
        DataGenLimit::Nodes(nodes) => {
            SearchLimit::SoftNodes { soft_limit: nodes, hard_limit: nodes * 8 }
        }
    });
    let nodes = AtomicU64::new(0);
    let mut info =
        SearchInfo { time_manager, print_to_stdout: false, ..SearchInfo::new(&stopped, &nodes) };

    let n_games_to_run = std::cmp::max(options.num_games / options.num_threads, 1);

    let mut output_file = File::create(data_dir.join(format!("thread_{id}.bin"))).unwrap();
    let mut output_buffer = BufWriter::new(&mut output_file);

    let mut counters = [
        (GameOutcome::WhiteWinMate, 0),
        (GameOutcome::BlackWinMate, 0),
        (GameOutcome::WhiteWinTB, 0),
        (GameOutcome::BlackWinTB, 0),
        (GameOutcome::DrawFiftyMoves, 0),
        (GameOutcome::DrawRepetition, 0),
        (GameOutcome::DrawStalemate, 0),
        (GameOutcome::DrawInsufficientMaterial, 0),
        (GameOutcome::DrawTB, 0),
        (GameOutcome::WhiteWinAdjudication, 0),
        (GameOutcome::BlackWinAdjudication, 0),
        (GameOutcome::DrawAdjudication, 0),
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
            eprintln!("[+] Main thread: Generated {game} games ({percentage:.1}%). Time per game: {time_per_game:.2} seconds.");
            eprintln!(
                " |> FENs generated: {fens} (FENs/sec = {fps:.2})",
                fens = FENS_GENERATED.load(Ordering::Relaxed),
                fps = FENS_GENERATED.load(Ordering::Relaxed) as f64 / start.elapsed().as_secs_f64()
            );
            eprintln!(
                " |> Estimated time remaining: {time_remaining:.2} seconds.",
                time_remaining = (n_games_to_run - game) as f64 * time_per_game
            );
            let est_completion_date = chrono::Local::now()
                .checked_add_signed(chrono::Duration::seconds(
                    <usize as std::convert::TryInto<i64>>::try_into(n_games_to_run - game).unwrap()
                        * time_per_game as i64,
                ))
                .unwrap();
            let time_completion = est_completion_date.format("%Y-%m-%d %H:%M:%S");
            eprintln!(" |> Estimated completion time: {time_completion}");
            std::io::stderr().flush().unwrap();
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
        thread_data.nnue.reinit_from(&board);
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
        info.time_manager.set_limit(SearchLimit::Depth(Depth::new(10)));
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
                        WDL::Win => GameOutcome::WhiteWinTB,
                        WDL::Loss => GameOutcome::BlackWinTB,
                        WDL::Draw => GameOutcome::DrawTB,
                    };
                }
            }
            tt.increase_age();

            let (score, best_move) =
                board.search_position(&mut info, std::array::from_mut(&mut thread_data), tt.view());

            game.add_move(best_move, score.try_into().unwrap());

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
                    GameOutcome::WhiteWinAdjudication
                } else {
                    GameOutcome::BlackWinAdjudication
                };
                break outcome;
            }
            if draw_adj_counter >= 12 {
                break GameOutcome::DrawAdjudication;
            }
            if is_game_theoretic_score(score) {
                // if the score is game theoretic, we don't want to play out the rest of the game
                let is_mate = score.abs() > MINIMUM_MATE_SCORE;
                break match (score.signum(), is_mate) {
                    (1, false) => GameOutcome::WhiteWinTB,
                    (-1, false) => GameOutcome::BlackWinTB,
                    (1, true) => GameOutcome::WhiteWinMate,
                    (-1, true) => GameOutcome::BlackWinMate,
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
        game.serialise_into(&mut output_buffer).unwrap();

        // increment the counter
        FENS_GENERATED.fetch_add(count as u64, Ordering::SeqCst);

        // STEP 5: update the game outcome statistics
        *counters.get_mut(&outcome).unwrap() += 1;

        // STEP 6: check if we should stop
        // either because the STOP_GENERATION signal was set,
        // or because we have generated enough positions.
        if STOP_GENERATION.load(Ordering::SeqCst)
            || options
                .position_count_limit
                .map_or(false, |limit| FENS_GENERATED.load(Ordering::SeqCst) >= limit)
        {
            break 'generation_main_loop;
        }
    }

    output_buffer.flush().unwrap();

    counters
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

fn config_loop(mut options: DataGenOptions) -> DataGenOptions {
    #![allow(clippy::option_if_let_else, clippy::too_many_lines)]
    println!();
    let mut user_input = String::new();
    loop {
        print!(">>> ");
        std::io::stdout().flush().unwrap();
        user_input.clear();
        std::io::stdin().read_line(&mut user_input).unwrap();
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
                        eprintln!("Warning: The specified number of threads ({num_threads}) is greater than the number of available CPUs ({num_cpus}).");
                    }
                    options.num_threads = num_threads;
                } else {
                    eprintln!("Invalid value for num_threads, must be a positive integer");
                }
            }
            "tablebases_path" => {
                if let Ok(tablebases_path) = value.parse::<PathBuf>() {
                    if !tablebases_path.exists() {
                        eprintln!("Warning: The specified tablebases path does not exist.");
                    } else if !tablebases_path.is_dir() {
                        eprintln!("Warning: The specified tablebases path is not a directory.");
                    } else {
                        options.tablebases_path = Some(tablebases_path);
                    }
                } else {
                    eprintln!("Invalid value for tablebases_path, must be a valid path");
                }
            }
            "use_nnue" => {
                if let Ok(use_nnue) = value.parse::<bool>() {
                    options.use_nnue = use_nnue;
                } else {
                    eprintln!("Invalid value for use_nnue, must be a boolean");
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
            "positions_limit" => {
                let positions_limit = match value.parse::<u64>() {
                    Ok(positions_limit) => positions_limit,
                    Err(e) => {
                        eprintln!("{e}");
                        continue;
                    }
                };
                options.position_count_limit = Some(positions_limit);
            }
            other => {
                eprintln!("Invalid parameter (\"{other}\"), supported parameters are \"num_games\", \"num_threads\", \"tablebases_path\", \"use_nnue\", \"limit\", and \"log_level\"");
            }
        }
    }
    options
}

impl FromStr for DataGenOptions {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut options = Self::new();
        let parts = s.split('-').collect::<Vec<_>>();
        if parts.len() != 6 {
            return Err(format!("Invalid options string: {s}"));
        }
        options.num_games = parts[0]
            .strip_suffix('g')
            .ok_or_else(|| format!("Invalid number of games: {}", parts[0]))?
            .parse()
            .map_err(|_| format!("Invalid number of games: {}", parts[0]))?;
        options.num_threads = parts[1]
            .strip_suffix('t')
            .ok_or_else(|| format!("Invalid number of threads: {}", parts[1]))?
            .parse()
            .map_err(|_| format!("Invalid number of threads: {}", parts[1]))?;
        if parts[2] != "no_tb" {
            options.tablebases_path = Some(PathBuf::from(parts[2]));
        }
        options.use_nnue = parts[3] == "nnue";
        options.generate_dfrc = match parts.get(4).copied() {
            Some("dfrc") => true,
            Some("classical") => false,
            _ => {
                return Err(format!(
                    "Invalid game type specifier: {}, must be \"dfrc\" or \"classical\"",
                    parts[4]
                ))
            }
        };
        let limit = match parts[5].chars().next() {
            Some('d') => DataGenLimit::Depth(
                parts[5]
                    .strip_prefix('d')
                    .ok_or_else(|| format!("Invalid depth limit: {}", parts[5]))?
                    .parse()
                    .map_err(|_| format!("Invalid depth limit: {}", parts[5]))?,
            ),
            Some('n') => DataGenLimit::Nodes(
                parts[5]
                    .strip_prefix('n')
                    .ok_or_else(|| format!("Invalid node limit: {}", parts[5]))?
                    .parse()
                    .map_err(|_| format!("Invalid node limit: {}", parts[5]))?,
            ),
            _ => return Err(format!("Invalid limit: {}", parts[5])),
        };
        options.limit = limit;
        options.log_level = 1;
        Ok(options)
    }
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
        writeln!(f, " |> use_nnue: {}", self.use_nnue)?;
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
            writeln!(f, "    ! Tablebases path not set - this will result in weaker data - are you sure you want to continue?")?;
        }
        Ok(())
    }
}

impl FromStr for DataGenLimit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        #![allow(clippy::cast_possible_truncation)]
        let (limit_type, limit_value) =
            s.split_once(' ').ok_or_else(|| format!("Invalid limit, no space: {s}"))?;
        let limit_value: u64 =
            limit_value.parse().map_err(|_| format!("Invalid limit value: {limit_value}"))?;
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
pub fn run_splat(input: &Path, output: &Path, filter: bool, marlinformat: bool) {
    // check that the input file exists
    if !input.exists() {
        eprintln!("Input file does not exist.");
        return;
    }
    // check that the output does not exist
    if output.exists() {
        eprintln!("Output file already exists.");
        return;
    }

    let filter_fn = |mv: Move, eval: i32, board: &Board| {
        if !filter {
            return true;
        }
        if board.ply() < MIN_SAVE_PLY {
            return false;
        }
        if is_game_theoretic_score(eval) {
            return false;
        }
        if board.pieces.occupied().count() < 4 {
            return false;
        }
        if board.is_tactical(mv) {
            return false;
        }
        if board.in_check() {
            return false;
        }
        true
    };

    // open the input file
    let input_file = File::open(input).unwrap();
    let mut input_buffer = BufReader::new(input_file);

    // open the output file
    let output_file = File::create(output).unwrap();
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
                    output_buffer.write_all(&packed_board.as_bytes()).unwrap();
                },
                filter_fn,
            );
        } else {
            game.splat_to_bulletformat(
                |chess_board| {
                    let bytes = unsafe {
                        std::mem::transmute::<_, [u8; std::mem::size_of::<ChessBoard>()]>(
                            chess_board,
                        )
                    };
                    output_buffer.write_all(&bytes).unwrap();
                },
                filter_fn,
            );
        }
        move_buffer = game.into_move_buffer();
        game_count += 1;
        if game_count % 2048 == 0 {
            print!("\r{game_count} games splatted");
            std::io::stdout().flush().unwrap();
        }
    }
    println!();

    output_buffer.flush().unwrap();
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
        let white = board.pieces.occupied_co(Colour::WHITE);
        let black = board.pieces.occupied_co(Colour::BLACK);
        for piece in PieceType::all().take(5) {
            let pieces = board.pieces.of_type(piece);
            let white_pieces = pieces & white;
            let black_pieces = pieces & black;
            mc.counts[piece.index()] = u8::try_from(white_pieces.count()).unwrap();
            mc.counts[piece.index() + 5] = u8::try_from(black_pieces.count()).unwrap();
        }
        // normalize the counts so that the white side has more material than the black side
        let ordering_key = |subslice: &[u8]| -> u64 {
            let count = u64::from(subslice.iter().sum::<u8>());
            let highest_piece =
                subslice.iter().enumerate().filter(|(_, v)| **v > 0).last().unwrap_or((0, &0)).0;
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
pub fn dataset_stats(dataset_path: &Path) {
    let mut move_buffer = Vec::new();
    let mut stats = DataSetStats::default();

    println!("Scanning dataset at {}", dataset_path.display());

    // because the format is variable-length, we can't know exactly our progress in terms of game-count.
    // we *can*, however, know how far we are through the file, so we use that as a progress indicator:
    print!("Progress: 0%");
    std::io::stdout().flush().unwrap();
    // get the file size
    let file_size = dataset_path.metadata().unwrap().len();

    let mut reader = BufReader::new(File::open(dataset_path).unwrap());

    while let Ok(game) =
        dataformat::Game::deserialise_from(&mut reader, std::mem::take(&mut move_buffer))
    {
        stats.games += 1;
        *stats.length_counts.entry(game.len()).or_default() += 1;
        game.visit_positions(|position, evaluation| {
            *stats.eval_counts.entry(evaluation).or_default() += 1;
            *stats
                .piece_counts
                .entry(u8::try_from(position.pieces.occupied().count()).unwrap())
                .or_default() += 1;
            *stats.material_counts.entry(MaterialConfiguration::from(position)).or_default() += 1;
            *stats.pov_king_positions.entry(position.king_sq(Colour::WHITE)).or_default() += 1;
            *stats
                .pov_king_positions
                .entry(position.king_sq(Colour::BLACK).flip_rank())
                .or_default() += 1;
        });
        move_buffer = game.into_move_buffer();

        // print progress
        if stats.games % 1024 == 0 {
            let progress = reader.stream_position().unwrap();
            let percentage = progress * 100 / file_size;
            print!("\rProgress: {percentage}%");
            std::io::stdout().flush().unwrap();
        }
    }
    println!();

    println!("Statistics for dataset at {}", dataset_path.display());
    println!("Number of games: {}", stats.games);
    println!("Writing length counts to length_counts.csv");
    let mut length_counts = stats.length_counts.iter().collect::<Vec<_>>();
    length_counts.sort_unstable_by_key(|(length, _)| *length);
    let mut length_counts_file = BufWriter::new(File::create("length_counts.csv").unwrap());
    writeln!(length_counts_file, "length,count").unwrap();
    for (length, count) in length_counts {
        writeln!(length_counts_file, "{length},{count}").unwrap();
    }
    length_counts_file.flush().unwrap();
    println!("Writing eval counts to eval_counts.csv");
    let mut eval_counts = stats.eval_counts.into_iter().collect::<Vec<_>>();
    eval_counts.sort_unstable_by_key(|(eval, _)| *eval);
    let mut eval_counts_file = BufWriter::new(File::create("eval_counts.csv").unwrap());
    writeln!(eval_counts_file, "eval,count").unwrap();
    for (eval, count) in eval_counts {
        writeln!(eval_counts_file, "{eval},{count}").unwrap();
    }
    eval_counts_file.flush().unwrap();
    println!("Writing piece counts to piece_counts.csv");
    let mut piece_counts = stats.piece_counts.into_iter().collect::<Vec<_>>();
    piece_counts.sort_unstable_by_key(|(count, _)| *count);
    let mut piece_counts_file = BufWriter::new(File::create("piece_counts.csv").unwrap());
    writeln!(piece_counts_file, "men,count").unwrap();
    for (men, count) in piece_counts {
        writeln!(piece_counts_file, "{men},{count}").unwrap();
    }
    piece_counts_file.flush().unwrap();
    println!("Writing material counts to material_counts.csv");
    let material_counts = stats.material_counts.into_iter().collect::<Vec<_>>();
    let mut material_counts_file = BufWriter::new(File::create("material_counts.csv").unwrap());
    writeln!(material_counts_file, "material,count").unwrap();
    for (mc, count) in material_counts {
        writeln!(material_counts_file, "{mc},{count}").unwrap();
    }
    material_counts_file.flush().unwrap();
    println!("Writing PoV king positions to pov_king_positions.csv");
    let pov_king_positions = stats.pov_king_positions.into_iter().collect::<Vec<_>>();
    let mut pov_king_positions_file =
        BufWriter::new(File::create("pov_king_positions.csv").unwrap());
    writeln!(pov_king_positions_file, "square,count").unwrap();
    for (sq, count) in pov_king_positions {
        writeln!(pov_king_positions_file, "{sq},{count}", sq = sq.index()).unwrap();
    }
    pov_king_positions_file.flush().unwrap();

    #[allow(clippy::cast_precision_loss)]
    let mean_game_len = ((stats.length_counts.iter().map(|(k, v)| k * v).sum::<usize>() as u128
        * 1000)
        / stats.games as u128) as f64
        / 1000.0;
    println!("Mean game length: {mean_game_len}");
}
