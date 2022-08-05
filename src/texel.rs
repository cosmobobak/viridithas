#![allow(dead_code)]

use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::Mutex,
    time::Instant,
};

use rand::{prelude::SliceRandom, Rng};
use rayon::prelude::*;

use crate::{
    board::{evaluation::parameters::Parameters, Board},
    definitions::{INFINITY, WHITE},
    searchinfo::SearchInfo,
};

const CONTROL_GREEN: &str = "\u{001b}[32m";
const CONTROL_RED: &str = "\u{001b}[31m";
const CONTROL_RESET: &str = "\u{001b}[0m";

#[derive(Debug, Clone, PartialEq)]
struct TrainingExample {
    fen: String,
    outcome: f64, // 0.0, 0.5, 1.0
}

const DEFAULT_K: f64 = 0.5;
fn sigmoid(s: f64, k: f64) -> f64 {
    1.0 / (1.0 + 10.0f64.powf(-k * s / 400.0))
}

fn total_squared_error(data: &[TrainingExample], params: &Parameters, k: f64) -> f64 {
    let mut pos = Board::default();
    let mut info = SearchInfo::default();
    pos.set_eval_params(params.clone());
    data.iter()
        .map(|TrainingExample { fen, outcome }| {
            // set_from_fen does not allocate, so it should be pretty fast.
            pos.set_from_fen(fen).unwrap();
            // quiescence is likely the source of all computation time.
            let pov_score = Board::quiescence(&mut pos, &mut info, -INFINITY, INFINITY);
            let score = if pos.turn() == WHITE {
                pov_score
            } else {
                -pov_score
            };
            let prediction = sigmoid(f64::from(score), k);
            (*outcome - prediction).powi(2)
        })
        .sum::<f64>()
}

fn compute_mse(data: &[TrainingExample], params: &Parameters, k: f64) -> f64 {
    #![allow(clippy::cast_precision_loss)]
    let n: f64 = data.len() as f64;
    let chunk_size = data.len() / num_cpus::get();
    data.par_chunks(chunk_size)
        .map(|chunk| total_squared_error(chunk, params, k))
        .sum::<f64>()
        / n
}

fn generate_particles(
    starting_point: Vec<i32>,
    n_particles: usize,
    parameter_distance: i32,
) -> Vec<Vec<i32>> {
    let mut particles = vec![starting_point];
    for _ in 1..n_particles {
        let mut new_particle = particles[0].clone();
        for param in &mut new_particle {
            *param += rand::random::<i32>() % parameter_distance * 2 - parameter_distance;
        }
        particles.push(new_particle);
    }
    particles
}

fn generate_velocities(
    n_params: usize,
    n_particles: usize,
    parameter_distance: i32,
) -> Vec<Vec<i32>> {
    let mut particles = vec![];
    for _ in 0..n_particles {
        let mut new_particle = vec![0; n_params];
        for param in &mut new_particle {
            *param = rand::random::<i32>() % parameter_distance * 2 - parameter_distance;
        }
        particles.push(new_particle);
    }
    particles
}

struct Particle {
    pub loc: Vec<i32>,
    pub best_known_params: Vec<i32>,
    pub best_cost: f64,
    pub velocity: Vec<i32>,
}

fn initialise<F1: Fn(&[i32]) -> f64 + Sync>(
    starting_point: Vec<i32>,
    cost_function: &F1,
    n_particles: usize,
    n_params: usize,
    particle_distance: i32,
    velocity_distance: i32,
) -> (Vec<Particle>, Vec<i32>, f64) {
    #![allow(clippy::cast_precision_loss)]
    // initialize the particles' positions with a uniformly distributed random vector
    println!("generating particle locations...");
    let particles = generate_particles(starting_point, n_particles, particle_distance);
    // initialize the particles' best known positions to their initial positions
    let mut particles = particles
        .into_iter()
        .map(|particle| Particle {
            loc: particle.clone(),
            best_known_params: particle,
            best_cost: 0.0,
            velocity: Vec::new(),
        })
        .collect::<Vec<_>>();
    // determine the lowest-energy particle in the whole swarm and initialise costs
    println!("finding the lowest-energy particle...");
    let mut best_loc = particles[0].loc.clone();
    println!("finding cost of particle 0 (the initial parameter configuration)...");
    let start_time = Instant::now();
    let mut best_cost = cost_function(&best_loc);
    let time_taken = start_time.elapsed().as_secs_f64();
    println!("cost of the default parameters: {}", best_cost);
    println!("costing the first particle took {:.1}s", time_taken);
    println!(
        "for {0} particles and {1} threads, it can be expected that we will need {0} * {2:.1} / {1} = ~{3:.1}s to compute an iteration.", 
        n_particles,
        num_cpus::get(),
        time_taken,
        n_particles as f64 * time_taken / num_cpus::get() as f64
    );
    let best_cost_mutex = Mutex::new(&mut best_cost);
    let best_loc_mutex = Mutex::new(&mut best_loc);
    particles[1..].par_iter_mut().for_each(|particle| {
        let cost = cost_function(&particle.loc);
        particle.best_cost = cost;
        let mut best_cost = best_cost_mutex.lock().unwrap();
        if cost < **best_cost {
            **best_loc_mutex.lock().unwrap() = particle.loc.clone();
            **best_cost = cost;
        }
    });
    let time_taken = start_time.elapsed().as_secs_f64();
    println!("finding the lowest-energy particle took {:.1}s", time_taken);
    // generate the initial velocity of each particle
    println!("generating particle velocities...");
    let velocities = generate_velocities(n_params, n_particles, velocity_distance);
    particles
        .iter_mut()
        .zip(velocities.into_iter())
        .for_each(|(particle, velocity)| {
            particle.velocity = velocity;
        });
    (particles, best_loc, best_cost)
}

fn particle_swarm_optimise<F1, F2>(
    starting_point: Vec<i32>,
    cost_function: F1,
    test_function: F2,
    inertia_weight: f64,
    cognitive_coeff: f64,
    social_coeff: f64,
    n_particles: usize,
    particle_distance: i32,
    velocity_distance: i32,
) -> (Vec<i32>, f64)
where
    F1: Fn(&[i32]) -> f64 + Sync,
    F2: Fn(&[i32]) -> f64 + Sync,
{
    #![allow(
        clippy::similar_names,
        clippy::cast_possible_truncation,
        clippy::needless_range_loop,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    let n_params = starting_point.len();
    let (mut particles, mut best_loc, mut best_cost) = initialise(
        starting_point,
        &cost_function,
        n_particles,
        n_params,
        particle_distance,
        velocity_distance,
    );
    let best_cost_mutex = Mutex::new(&mut best_cost);
    // begin the optimisation loop proper
    println!("optimising...");
    let mut iterations_without_progress = 0;
    for iteration in 1.. {
        println!("iteration {}", iteration);
        let it_start = Instant::now();
        let cost_before = **best_cost_mutex.lock().unwrap();
        println!("computing new velocities...");
        particles.par_iter_mut().for_each(|particle| {
            let mut rng = rand::thread_rng();
            // update velocity based on swarm information
            for (((&x_id, &p_id), v_id), &g_d) in particle
                .loc
                .iter()
                .zip(particle.best_known_params.iter())
                .zip(particle.velocity.iter_mut())
                .zip(best_loc.iter())
            {
                let r_p = rng.gen_range(0.0..1.0);
                let r_g = rng.gen_range(0.0..1.0);
                let d_p = f64::from(p_id - x_id);
                let d_g = f64::from(g_d - x_id);
                let inertial_component = f64::from(*v_id) * inertia_weight;
                let cognitive_component = r_p * d_p * cognitive_coeff;
                let social_component = r_g * d_g * social_coeff;
                *v_id = (inertial_component + cognitive_component + social_component) as i32;
            }
        });
        println!("updating particle positions...");
        particles.par_iter_mut().for_each(|particle| {
            // update position based on velocity
            for (x_i, v_i) in particle.loc.iter_mut().zip(particle.velocity.iter()) {
                *x_i += *v_i;
            }
        });
        let best_loc_mutex = Mutex::new(&mut best_loc);
        println!("finding new best particle...");
        particles
            .par_iter_mut()
            .map(|particle| {
                let new_cost = cost_function(&particle.loc);
                (particle, new_cost)
            })
            .for_each(|(particle, new_cost)| {
                if new_cost < particle.best_cost {
                    particle.best_cost = new_cost;
                    particle.best_known_params = particle.loc.clone();
                    let mut best_cost = best_cost_mutex.lock().unwrap();
                    if new_cost < **best_cost {
                        **best_loc_mutex.lock().unwrap() = particle.loc.clone();
                        **best_cost = new_cost;
                    }
                }
            });
        let mut best_cost = best_cost_mutex.lock().unwrap();
        let mut best_loc = best_loc_mutex.lock().unwrap();
        println!("iteration done in {:.1}s", it_start.elapsed().as_secs_f64());
        println!("best cost after iteration {iteration}: {best_cost}");
        println!("computing test score...");
        let test_score = test_function(&best_loc);
        println!("test score: {test_score}");
        if **best_cost < cost_before {
            iterations_without_progress = 0;
            println!(
                "{CONTROL_GREEN}best cost decreased by {}!{CONTROL_RESET}",
                cost_before - **best_cost
            );
            println!("saving params to params_{iteration}.txt");
            Parameters::save_param_vec(&best_loc, &format!("params/params_{iteration}.txt"));
        } else {
            iterations_without_progress += 1;
        }
        if iterations_without_progress == 10 {
            println!("{CONTROL_RED}no progress for 10 iterations, re-initialising particles...{CONTROL_RESET}");
            let (new_particles, g_best_loc, g_best_cost) = initialise(
                best_loc.clone(),
                &cost_function,
                n_particles,
                n_params,
                particle_distance,
                velocity_distance,
            );
            particles = new_particles;
            **best_loc = g_best_loc;
            **best_cost = g_best_cost;
            iterations_without_progress = 0;
            println!("re-initialised particles, continuing...");
        }
    }

    (best_loc, best_cost)
}

fn local_search_optimise<F1: Fn(&[i32]) -> f64 + Sync>(
    starting_point: &[i32],
    cost_function: F1,
) -> (Vec<i32>, f64) {
    #[allow(clippy::cast_possible_truncation)]
    fn nudge_size(iteration: i32) -> i32 { ((1.0 / f64::from(iteration) * 10.0) as i32).max(1) }
    let init_start_time = Instant::now();
    let n_params = starting_point.len();
    let mut best_params = starting_point.to_vec();
    let mut best_err = cost_function(&best_params);
    let mut improved = true;
    let mut iteration = 1;
    println!(
        "Initialised in {:.1}s",
        init_start_time.elapsed().as_secs_f64()
    );
    while improved || iteration <= 10 {
        println!("Iteration {iteration}");
        improved = false;
        let nudge_size = nudge_size(iteration);

        for param_idx in 0..n_params {
            println!("Optimising param {param_idx}");
            let mut new_params = best_params.clone();
            new_params[param_idx] += nudge_size; // try adding step_size to the param
            let new_err = cost_function(&new_params);
            if new_err < best_err {
                best_params = new_params;
                best_err = new_err;
                improved = true;
                println!("{CONTROL_GREEN}found improvement! (+){CONTROL_RESET}");
            } else {
                new_params[param_idx] -= nudge_size * 2; // try subtracting step_size from the param
                let new_err = cost_function(&new_params);
                if new_err < best_err {
                    best_params = new_params;
                    best_err = new_err;
                    improved = true;
                    println!("{CONTROL_GREEN}found improvement! (-){CONTROL_RESET}");
                } else {
                    new_params[param_idx] += nudge_size; // reset the param.
                    println!("{CONTROL_RED}no improvement.{CONTROL_RESET}");
                }
            }
        }
        Parameters::save_param_vec(&best_params, &format!("params/localsearch{iteration:0>3}.txt"));
        iteration += 1;
    }
    (best_params, best_err)
}

pub fn tune() {
    // hyperparameters
    let train = 12_000_000; // 8 million is recommended.
    let test = 100_000; // validation set.

    // let n_particles = 100; // No idea what a good value is.
    // let inertia_weight = 0.8; // the inertia of a particle
    // let cognitive_coeff = 1.7; // how much particles get drawn towards their best known position
    // let social_coeff = 1.7; // how much particles get drawn towards the best position of the whole swarm
    // let particle_distance = 10; // how far away from the default parameters a particle can be
    // let velocity_distance = 3; // how far away from the zero velocity a particle can begin

    let data = File::open("../texel_data.txt").unwrap();

    println!("Parsing tuning data...");
    let start_time = Instant::now();
    let mut data = BufReader::new(data)
        .lines()
        .map(|line| {
            let line = line.unwrap();
            let (fen, outcome) = line.rsplit_once(' ').unwrap();
            let outcome = outcome.parse::<f64>().unwrap();
            let fen = fen.to_owned();
            TrainingExample { fen, outcome }
        })
        .collect::<Vec<_>>();
    println!(
        "Parsed {} examples in {:.1}s",
        data.len(),
        start_time.elapsed().as_secs_f32()
    );

    println!("Shuffling data...");
    let start_time = Instant::now();
    let mut rng = rand::thread_rng();
    data.shuffle(&mut rng);
    println!(
        "Shuffled data in {:.1}s",
        start_time.elapsed().as_secs_f32()
    );

    println!("Splitting data...");

    assert!(train + test <= data.len(), "not enough data for training and testing, requested train = {train}, test = {test}, but data has {} examples", data.len());
    data.truncate(train + test);
    let (train_set, _test_set) = data.split_at(train);

    let params = Parameters::default();

    println!("Optimising...");
    println!(
        "There are {} parameters to optimise",
        params.vectorise().len()
    );
    let start_time = Instant::now();
    // let (best_params, best_loss) = particle_swarm_optimise(
    //     params.vectorise(),
    //     |pvec| compute_mse(&train_set, &Parameters::devectorise(pvec), DEFAULT_K),
    //     |pvec| compute_mse(&test_set, &Parameters::devectorise(pvec), DEFAULT_K),
    //     inertia_weight,
    //     cognitive_coeff,
    //     social_coeff,
    //     n_particles,
    //     particle_distance,
    //     velocity_distance,
    // );
    let (best_params, best_loss) = local_search_optimise(&params.vectorise(), |pvec| {
        compute_mse(train_set, &Parameters::devectorise(pvec), DEFAULT_K)
    });
    println!("Optimised in {:.1}s", start_time.elapsed().as_secs_f32());

    println!("Best loss: {best_loss:.6}");
    println!("Saving best parameters...");
    Parameters::save_param_vec(&best_params, "params/localsearchfinal.txt");
}
