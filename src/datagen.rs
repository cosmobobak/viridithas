use std::{fmt::Display, path::{PathBuf, Path}, str::FromStr, hash::Hash, fs::File, io::{BufWriter, Write}, sync::atomic::Ordering, time::Instant};

use crate::{transpositiontable::TT, definitions::{MEGABYTE, depth::Depth}, board::{Board, GameOutcome, evaluation::is_game_theoretic_score}, threadlocal::ThreadData, searchinfo::{SearchInfo, SearchLimit}, tablebases::{self, probe::WDL}, uci::{SYZYGY_PATH, SYZYGY_ENABLED}};

/// Whether to limit searches by depth or by nodes.
#[derive(Clone, Debug, Hash)]
enum DataGenLimit {
    Depth(i32),
    Nodes(u64),
}

impl FromStr for DataGenLimit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        #![allow(clippy::cast_possible_truncation)]
        let (limit_type, limit_value) = s.split_once(' ').ok_or_else(|| format!("Invalid limit, no space: {s}"))?;
        let limit_value: u64 = limit_value.parse().map_err(|_| format!("Invalid limit value: {limit_value}"))?;
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
            use_nnue: false, 
            limit: DataGenLimit::Depth(8),
            log_level: 1,
        }
    }

    /// Sets the number of games to generate.
    const fn num_games(mut self, num_games: usize) -> Self {
        self.num_games = num_games;
        self
    }

    /// Sets the number of threads to use.
    const fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Sets the path to the directory containing syzygy endgame tablebases.
    #[allow(clippy::missing_const_for_fn)]
    fn tablebases_path(mut self, tablebases_path: PathBuf) -> Self {
        self.tablebases_path = Some(tablebases_path);
        self
    }

    /// Sets whether to use NNUE evaluation during self-play.
    const fn use_nnue(mut self, use_nnue: bool) -> Self {
        self.use_nnue = use_nnue;
        self
    }

    /// Sets the depth or node limit for searches.
    const fn limit(mut self, limit: DataGenLimit) -> Self {
        self.limit = limit;
        self
    }

    /// Sets the log level
    const fn log_level(mut self, log_level: u8) -> Self {
        self.log_level = log_level;
        self
    }

    /// Gives a summarised string representation of the options.
    fn summary(&self) -> String {
        format!(
            "{}g-{}-{}-{}",
            self.num_games,
            if self.tablebases_path.is_some() { "tb" } else { "no-tb" },
            if self.use_nnue { "nnue" } else { "hce" },
            match self.limit {
                DataGenLimit::Depth(depth) => format!("d{depth}"),
                DataGenLimit::Nodes(nodes) => format!("n{nodes}"),
            }
        )
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
        writeln!(f, " |> limit: {}", match self.limit {
            DataGenLimit::Depth(depth) => format!("depth {depth}"),
            DataGenLimit::Nodes(nodes) => format!("nodes {nodes}"),
        })?;
        writeln!(f, " |> log_level: {}", self.log_level)?;
        if self.tablebases_path.is_none() {
            writeln!(f, "    ! Tablebases path not set - this will result in weaker data - are you sure you want to continue?")?;
        }
        Ok(())
    }
}

pub fn gen_data_main() {
    let mut options = DataGenOptions::new()
        .num_games(100)
        .num_threads(1)
        .use_nnue(true);
    show_boot_info(&options);
    options = config_loop(options);
    println!("Starting data generation with the following configuration:");
    println!("{options}");
    if options.num_games % options.num_threads != 0 {
        println!("Warning: The number of games is not evenly divisible by the number of threads,");
        println!("this will result in {} games being omitted.", options.num_games % options.num_threads);
    }
    if let Some(tb_path) = &options.tablebases_path {
        let tb_path = tb_path.to_string_lossy();
        tablebases::probe::init(&tb_path);
        *SYZYGY_PATH.lock().unwrap() = tb_path.to_string();
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

    println!("Done!");
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
        if command == "start" {
            break;
        }
        if command != "set" {
            println!("Invalid command, supported commands are \"set <PARAM> <VALUE>\" and \"start\"");
            continue;
        }
        let Some(param) = user_input.next() else {
            println!("Invalid command, supported commands are \"set <PARAM> <VALUE>\" and \"start\"");
            continue;
        };
        let Some(value) = user_input.next() else {
            println!("Invalid command, supported commands are \"set <PARAM> <VALUE>\" and \"start\"");
            continue;
        };
        match param {
            "num_games" => {
                if let Ok(num_games) = value.parse::<usize>() {
                    options = options.num_games(num_games);
                } else {
                    println!("Invalid value for num_games, must be a positive integer");
                }
            }
            "num_threads" => {
                if let Ok(num_threads) = value.parse::<usize>() {
                    options = options.num_threads(num_threads);
                } else {
                    println!("Invalid value for num_threads, must be a positive integer");
                }
            }
            "tablebases_path" => {
                if let Ok(tablebases_path) = value.parse::<PathBuf>() {
                    if !tablebases_path.exists() {
                        println!("Warning: The specified tablebases path does not exist.");
                    } else if !tablebases_path.is_dir() {
                        println!("Warning: The specified tablebases path is not a directory.");
                    } else {
                        options = options.tablebases_path(tablebases_path);
                    }
                } else {
                    println!("Invalid value for tablebases_path, must be a valid path");
                }
            }
            "use_nnue" => {
                if let Ok(use_nnue) = value.parse::<bool>() {
                    options = options.use_nnue(use_nnue);
                } else {
                    println!("Invalid value for use_nnue, must be a boolean");
                }
            }
            "limit" => {
                let Some(limit_size) = user_input.next() else {
                    println!("Trying to set limit, but only one token was provided");
                    println!("Usage: \"set limit <TYPE> <NUMBER>\"");
                    println!("Example: \"set limit depth 8\" (sets the limit to 8 plies)");
                    continue;
                };
                let full_limit = value.to_string() + " " + limit_size;
                let limit = match full_limit.parse::<DataGenLimit>() {
                    Ok(limit) => limit,
                    Err(e) => {
                        println!("{e}");
                        continue;
                    }
                };
                options = options.limit(limit);
            }
            "log_level" => {
                let log_level = match value.parse::<u8>() {
                    Ok(log_level) => log_level,
                    Err(e) => {
                        println!("{e}");
                        continue;
                    }
                };
                options = options.log_level(log_level);
            }
            other => {
                println!("Invalid parameter (\"{other}\"), supported parameters are \"num_games\", \"num_threads\", \"tablebases_path\", \"use_nnue\", \"limit\", and \"log_level\"");
            }
        }
    }
    options
}

fn show_boot_info(options: &DataGenOptions) {
    println!("Welcome to Viri's data generation tool!");
    println!("This tool will generate self-play data for Viridithas.");
    println!("You can configure the data generation process by setting the following parameters:");
    println!("{options}");
    println!("It is recommended that you do not set the number of threads to more than the number of logical cores on your CPU, as performance will suffer.");
    println!("(You have {} logical cores on your CPU)", num_cpus::get());
    println!("To set a parameter, type \"set <PARAM> <VALUE>\"");
    println!("To start data generation, type \"start\"");
}

fn generate_on_thread(id: usize, options: &DataGenOptions, data_dir: &Path) {
    #![allow(clippy::cast_precision_loss)]
    // this rng is different between each thread 
    // (https://rust-random.github.io/book/guide-parallel.html)
    // so no worries :3
    let mut rng = rand::thread_rng();
    let mut board = Board::new();
    let mut thread_data = ThreadData::new(id);
    thread_data.alloc_tables();
    let mut tt = TT::new();
    tt.resize(16 * MEGABYTE);
    let mut info = SearchInfo {
        print_to_stdout: false,
        limit: match options.limit {
            DataGenLimit::Depth(depth) => SearchLimit::Depth(Depth::new(depth)),
            DataGenLimit::Nodes(nodes) => SearchLimit::Nodes(nodes),
        },
        ..Default::default()
    };

    let n_games_to_run = std::cmp::max(options.num_games / options.num_threads, 1);

    let mut output_file = File::create(data_dir.join(format!("thread_{id}.txt"))).unwrap();
    let mut output_buffer = BufWriter::new(&mut output_file);

    let mut single_game_buffer = Vec::new();

    let start = Instant::now();
    'generation_main_loop: for game in 0..n_games_to_run {
        // report progress
        if id == 0 && game % 256 == 0 && options.log_level > 0 {
            let percentage = game * 100_000 / n_games_to_run;
            let percentage = percentage as f64 / 1000.0;
            let time_per_game = start.elapsed().as_secs_f64() / game as f64;
            eprintln!("Main thread: Generated {game} games ({percentage:.1}%). Time per game: {time_per_game:.2} seconds.");
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
        let temp_limit = std::mem::replace(&mut info.limit, SearchLimit::Depth(Depth::new(12)));
        let (eval, _) = board.search_position::<true>(&mut info, std::array::from_mut(&mut thread_data), tt.view());
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
                    }
                }
            }
            tt.increase_age();
            let (score, best_move) = board.search_position::<true>(&mut info, std::array::from_mut(&mut thread_data), tt.view());
            single_game_buffer.push((score, board.fen()));
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
        for (i, (score, fen)) in single_game_buffer.iter().enumerate() {
            if is_game_theoretic_score(*score) {
                if options.log_level > 2 {
                    eprintln!("Move {i} is game theoretic, skipping rest.");
                }
                break;
            }
            writeln!(output_buffer, "{fen} | {score} | {outcome}").unwrap();
        }
        single_game_buffer.clear();
    }
}