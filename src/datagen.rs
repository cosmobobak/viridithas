use std::{
    fmt::Display,
    fs::File,
    hash::Hash,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    str::FromStr,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};

use crate::{
    board::{evaluation::is_game_theoretic_score, Board, GameOutcome},
    definitions::{depth::Depth, MEGABYTE},
    searchinfo::{SearchInfo, SearchLimit},
    tablebases::{self, probe::WDL},
    threadlocal::ThreadData,
    transpositiontable::TT,
    uci::{SYZYGY_ENABLED, SYZYGY_PATH},
};

static FENS_GENERATED: AtomicU64 = AtomicU64::new(0);

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
    // log level
    log_level: u8,
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
            log_level: 1,
        }
    }

    /// Gives a summarised string representation of the options.
    fn summary(&self) -> String {
        format!(
            "{}g-{}t-{}-{}-{}",
            self.num_games,
            self.num_threads,
            self.tablebases_path.as_ref().map_or_else(
                || "no-tb".into(),
                |tablebases_path| tablebases_path.to_string_lossy()
            ),
            if self.use_nnue { "nnue" } else { "hce" },
            match self.limit {
                DataGenLimit::Depth(depth) => format!("d{depth}"),
                DataGenLimit::Nodes(nodes) => format!("n{nodes}"),
            }
        )
    }
}

pub fn gen_data_main(cli_config: Option<&str>) {
    FENS_GENERATED.store(0, Ordering::SeqCst);

    let options: DataGenOptions = cli_config.map_or_else(|| {
            let options = DataGenOptions::new();
            show_boot_info(&options);
            config_loop(options)
        }, |s| s.parse().expect("Failed to parse CLI config, expected short def string (e.g. '100g-2t-<TBPATH>-nnue-d8')"));
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

    std::thread::scope(|s| {
        let thread_handles = (0..options.num_threads)
            .map(|id| {
                let opt_ref = &options;
                let path_ref = &data_dir;
                s.spawn(move || generate_on_thread(id, opt_ref, path_ref))
            })
            .collect::<Vec<_>>();
        for handle in thread_handles {
            handle.join().unwrap();
        }
    });

    if options.log_level > 0 {
        println!("Done!");
    }
}

fn generate_on_thread(id: usize, options: &DataGenOptions, data_dir: &Path) {
    #![allow(clippy::cast_precision_loss, clippy::too_many_lines, clippy::cast_possible_truncation)]
    // this rng is different between each thread
    // (https://rust-random.github.io/book/guide-parallel.html)
    // so no worries :3
    let mut rng = rand::thread_rng();
    let mut board = Board::new();
    let mut thread_data = ThreadData::new(id);
    thread_data.alloc_tables();
    let mut tt = TT::new();
    tt.resize(16 * MEGABYTE);
    let stopped = AtomicBool::new(false);
    let mut info = SearchInfo {
        print_to_stdout: false,
        limit: match options.limit {
            DataGenLimit::Depth(depth) => SearchLimit::Depth(Depth::new(depth)),
            DataGenLimit::Nodes(nodes) => SearchLimit::Nodes(nodes),
        },
        ..SearchInfo::new(&stopped)
    };

    let n_games_to_run = std::cmp::max(options.num_games / options.num_threads, 1);

    let mut output_file = File::create(data_dir.join(format!("thread_{id}.txt"))).unwrap();
    let mut output_buffer = BufWriter::new(&mut output_file);

    let mut single_game_buffer = Vec::new();

    let start = Instant::now();
    'generation_main_loop: for game in 0..n_games_to_run {
        // report progress
        if id == 0 && game % 64 == 0 && options.log_level > 0 && game > 0 {
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
                    (n_games_to_run - game) as i64 * time_per_game as i64,
                ))
                .unwrap();
            let time_completion = est_completion_date.format("%Y-%m-%d %H:%M:%S");
            eprintln!(" |> Estimated completion time: {time_completion}");
            std::io::stderr().flush().unwrap();
        }
        // reset everything: board, thread data, tt, search info
        board.set_startpos();
        thread_data.nnue.refresh_acc(&board);
        tt.clear();
        info.setup_for_search();
        // flush output buffer
        output_buffer.flush().unwrap();
        // generate game
        if options.log_level > 1 {
            eprintln!("Generating game {game}...");
        }
        // STEP 1: make random moves for variety
        if options.log_level > 2 {
            eprintln!("Making random moves...");
        }
        for _ in 0..12 {
            let res = board.make_random_move::<true>(&mut rng, &mut thread_data);
            if res.is_none() {
                if options.log_level > 2 {
                    eprintln!("Reached a position with no legal moves, skipping...");
                }
                continue 'generation_main_loop;
            }
        }
        // STEP 2: evaluate the exit position with reasonable depth
        // to make sure that it isn't silly.
        if options.log_level > 2 {
            eprintln!("Evaluating position...");
        }
        let temp_limit = std::mem::replace(&mut info.limit, SearchLimit::Depth(Depth::new(10)));
        let (eval, _) = board.search_position::<true>(
            &mut info,
            std::array::from_mut(&mut thread_data),
            tt.view(),
        );
        if eval.abs() > 1000 {
            if options.log_level > 2 {
                eprintln!("Position is too good or too bad, skipping...");
            }
            // if the position is too good or too bad, we don't want it
            continue 'generation_main_loop;
        }
        info.limit = temp_limit;
        // STEP 3: play out to the end of the game
        if options.log_level > 2 {
            eprintln!("Playing out game...");
        }
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
            let (score, best_move) = board.search_position::<true>(
                &mut info,
                std::array::from_mut(&mut thread_data),
                tt.view(),
            );
            if !board.is_tactical(best_move)
                && !is_game_theoretic_score(score)
                && !board.in_check::<{ Board::US }>()
            {
                // we only save FENs where the best move is not tactical (promotions or captures)
                // and the score is not game theoretic (mate or TB-win),
                // and the side to move is not in check.
                single_game_buffer.push((score, board.fen()));
            }
            board.make_move::<true>(best_move, &mut thread_data);
        };
        if options.log_level > 2 {
            eprintln!("Game is over, outcome: {outcome:?}");
        }
        assert_ne!(outcome, GameOutcome::Ongoing, "Game should be over by now.");
        let outcome = outcome.as_float_str();
        // STEP 4: write the game to the output file
        if options.log_level > 2 {
            eprintln!("Writing {} moves to output file...", single_game_buffer.len());
        }
        for (score, fen) in &single_game_buffer {
            writeln!(output_buffer, "{fen} | {score} | {outcome}").unwrap();
        }
        FENS_GENERATED.fetch_add(single_game_buffer.len() as u64, Ordering::SeqCst);
        single_game_buffer.clear();
    }
}

fn show_boot_info(options: &DataGenOptions) {
    if options.log_level > 0 {
        println!("Welcome to Viri's data generation tool!");
        println!("This tool will generate self-play data for Viridithas.");
        println!(
            "You can configure the data generation process by setting the following parameters:"
        );
        println!("{options}");
        println!("It is recommended that you do not set the number of threads to more than the number of logical cores on your CPU, as performance will suffer.");
        println!("(You have {} logical cores on your CPU)", num_cpus::get());
        println!("To set a parameter, type \"set <PARAM> <VALUE>\"");
        println!("To start data generation, type \"start\" or \"go\".");
    }
}

fn config_loop(mut options: DataGenOptions) -> DataGenOptions {
    #![allow(clippy::option_if_let_else)]
    println!();
    let mut user_input = String::new();
    loop {
        print!(">>> ");
        std::io::stdout().flush().unwrap();
        user_input.clear();
        std::io::stdin().read_line(&mut user_input).unwrap();
        let mut user_input = user_input.split_whitespace();
        let Some(command) = user_input.next() else { continue; };
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
            eprintln!("Invalid command, supported commands are \"set <PARAM> <VALUE>\" and \"start\"");
            continue;
        };
        let Some(value) = user_input.next() else {
            eprintln!("Invalid command, supported commands are \"set <PARAM> <VALUE>\" and \"start\"");
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
    options
}

impl FromStr for DataGenOptions {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut options = Self::new();
        let parts = s.split('-').collect::<Vec<_>>();
        if parts.len() != 5 {
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
        let limit = match parts[4].chars().next() {
            Some('d') => DataGenLimit::Depth(
                parts[4]
                    .strip_prefix('d')
                    .ok_or_else(|| format!("Invalid depth limit: {}", parts[4]))?
                    .parse()
                    .map_err(|_| format!("Invalid depth limit: {}", parts[4]))?,
            ),
            Some('n') => DataGenLimit::Nodes(
                parts[4]
                    .strip_prefix('n')
                    .ok_or_else(|| format!("Invalid node limit: {}", parts[4]))?
                    .parse()
                    .map_err(|_| format!("Invalid node limit: {}", parts[4]))?,
            ),
            _ => return Err(format!("Invalid limit: {}", parts[4])),
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
