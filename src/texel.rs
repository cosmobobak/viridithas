use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    sync::atomic::{AtomicBool, AtomicU64},
    time::Instant,
};

use rand::prelude::SliceRandom;

use crate::{
    board::{evaluation::parameters::EvalParams, Board},
    util::{INFINITY, MEGABYTE},
    piece::Colour,
    search::pv::PVariation,
    searchinfo::SearchInfo,
    threadlocal::ThreadData,
    transpositiontable::TT,
};

const CONTROL_GREEN: &str = "\u{001b}[32m";
const CONTROL_RED: &str = "\u{001b}[31m";
const CONTROL_RESET: &str = "\u{001b}[0m";

#[derive(Debug, Clone, PartialEq)]
struct TrainingExample {
    fen: String,
    outcome: f64, // expect 0.0, 0.5, 1.0
}

const DEFAULT_K: f64 = 0.5;
fn sigmoid(s: f64, k: f64) -> f64 {
    1.0 / (1.0 + 10.0f64.powf(-k * s / 400.0))
}

fn total_squared_error(data: &[TrainingExample], params: &EvalParams, k: f64) -> f64 {
    let stopped = AtomicBool::new(false);
    let mut pv = PVariation::default();
    let mut pos = Board::default();
    let nodes = AtomicU64::new(0);
    let mut info = SearchInfo::new(&stopped, &nodes);
    let mut tt = TT::new();
    tt.resize(MEGABYTE, 1);
    let mut t = ThreadData::new(0, &pos);
    info.eval_params = params.clone();
    data.iter()
        .map(|TrainingExample { fen, outcome }| {
            // set_from_fen does not allocate, so it should be pretty fast.
            pos.set_from_fen(fen).unwrap();
            // quiescence is likely the source of all computation time.
            // don't use NNUE, this only works for HCE.
            let pov_score = Board::quiescence::<true, false>(
                &mut pos,
                tt.view(),
                &mut pv,
                &mut info,
                &mut t,
                -INFINITY,
                INFINITY,
            );
            let score = if pos.turn() == Colour::WHITE { pov_score } else { -pov_score };
            let prediction = sigmoid(f64::from(score), k);
            (*outcome - prediction).powi(2)
        })
        .sum::<f64>()
}

fn compute_mse(data: &[TrainingExample], params: &[i32], k: f64) -> f64 {
    #![allow(clippy::cast_precision_loss)]
    let params = EvalParams::devectorise(params);
    let n: f64 = data.len() as f64;
    let chunk_size = data.len() / num_cpus::get() + 1; // +1 to avoid 0
    let chunks = data.chunks(chunk_size).collect::<Vec<_>>();
    let total = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for chunk in chunks {
            handles.push(s.spawn(|| total_squared_error(chunk, &params, k)));
        }
        handles.into_iter().map(|h| h.join().unwrap()).sum::<f64>()
    });
    total / n
}

struct PVec {
    pub params: Vec<i32>,
    pub mse: f64,
}

fn local_search_optimise<F1: FnMut(&[i32]) -> f64 + Sync>(
    starting_point: &[i32],
    resume: bool,
    mut cost_function: F1,
    params_to_tune: Option<&[usize]>,
) -> PVec {
    if let Some(ptt) = params_to_tune {
        println!("limiting tuning to params: {ptt:?}");
    }
    #[allow(clippy::cast_possible_truncation)]
    let nudge_size = |iteration: i32| -> i32 {
        let multiplier = if resume { 1.0 } else { 10.0 };
        ((1.0 / f64::from(iteration) * multiplier) as i32).max(1)
    };
    let init_start_time = Instant::now();
    let n_params = starting_point.len();
    let best_params = starting_point.to_vec();
    let mut best_params = PVec { mse: cost_function(&best_params), params: best_params };
    let initial_err = best_params.mse;
    println!("initial error: {initial_err}");
    let mut improved = true;
    let mut iteration = 1;
    println!("Initialised in {:.1}s", init_start_time.elapsed().as_secs_f64());
    while improved || (iteration <= 10 && !resume) {
        println!("Iteration {iteration}");
        improved = false;
        let nudge_size = nudge_size(iteration);
        println!("nudge size: {nudge_size}");

        for param_idx in 0..n_params {
            if let Some(params_to_tune) = params_to_tune {
                if !params_to_tune.contains(&param_idx) {
                    continue;
                }
            }
            println!("Optimising param {param_idx}");
            let start = Instant::now();
            let mut new_params = best_params.params.clone();
            new_params[param_idx] += nudge_size; // try adding step_size to the param
            let new_err = cost_function(&new_params);
            if new_err < best_params.mse {
                best_params = PVec { params: new_params, mse: new_err };
                improved = true;
                let time_taken = start.elapsed().as_secs_f64();
                println!("{CONTROL_GREEN}found improvement! (+{nudge_size}){CONTROL_RESET} ({time_taken:.2}s)");
                println!("new error: {}", best_params.mse);
                let percentage_of_initial = (best_params.mse / initial_err) * 100.0;
                println!("({percentage_of_initial:.2}% of initial error)");
            } else {
                new_params[param_idx] -= nudge_size * 2; // try subtracting step_size from the param
                let new_err = cost_function(&new_params);
                if new_err < best_params.mse {
                    best_params = PVec { params: new_params, mse: new_err };
                    improved = true;
                    let time_taken = start.elapsed().as_secs_f64();
                    println!("{CONTROL_GREEN}found improvement! (-{nudge_size}){CONTROL_RESET} ({time_taken:.2}s)");
                    println!("new error: {}", best_params.mse);
                    let percentage_of_initial = (best_params.mse / initial_err) * 100.0;
                    println!("({percentage_of_initial:.2}% of initial error)");
                } else {
                    new_params[param_idx] += nudge_size; // reset the param.
                    let time_taken = start.elapsed().as_secs_f64();
                    println!("{CONTROL_RED}no improvement.{CONTROL_RESET} ({time_taken:.2}s)");
                }
            }
        }
        EvalParams::save_param_vec(
            &best_params.params,
            format!("params/localsearch{iteration:0>3}.txt"),
        );
        iteration += 1;
    }
    best_params
}

pub fn tune(
    resume: bool,
    examples: usize,
    starting_params: &EvalParams,
    params_to_tune: Option<&[usize]>,
    data_path: impl AsRef<Path>,
) {
    println!("Parsing tuning data...");
    let start_time = Instant::now();
    let mut data = read_data(data_path);
    println!("Parsed {} examples in {:.1}s", data.len(), start_time.elapsed().as_secs_f32());

    println!("Shuffling data...");
    let start_time = Instant::now();
    let mut rng = rand::thread_rng();
    data.shuffle(&mut rng);
    println!("Shuffled data in {:.1}s", start_time.elapsed().as_secs_f32());

    println!("Splitting data...");

    assert!(
        examples <= data.len(),
        "not enough data for training, requested examples = {examples}, but data has {} examples",
        data.len()
    );

    println!("Optimising...");
    let n_params = params_to_tune.map_or_else(|| starting_params.vectorise().len(), <[usize]>::len);
    println!("There are {n_params} parameters to optimise");
    let start_time = Instant::now();

    let training_data = &data[..examples];

    let PVec { params: best_params, mse: best_loss } = local_search_optimise(
        &starting_params.vectorise(),
        resume,
        |params| compute_mse(training_data, params, DEFAULT_K),
        params_to_tune,
    );
    println!("Optimised in {:.1}s", start_time.elapsed().as_secs_f32());

    println!("Best loss: {best_loss:.6}");
    println!("Saving best parameters...");
    EvalParams::save_param_vec(&best_params, "params/localsearchfinal.txt");
}

fn read_data(path: impl AsRef<Path>) -> Vec<TrainingExample> {
    let data = File::open(path).unwrap();
    BufReader::new(data)
        .lines()
        .map(|line| {
            let line = line.unwrap();
            // we expect lines in the format "{fen};{outcome}"
            let (fen, outcome) = line.rsplit_once(';').unwrap();
            let outcome = outcome.parse::<f64>().unwrap();
            let fen = fen.to_owned();
            TrainingExample { fen, outcome }
        })
        .collect::<Vec<_>>()
}
