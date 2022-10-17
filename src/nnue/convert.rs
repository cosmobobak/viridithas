use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use crate::{
    board::Board,
    definitions::{depth::Depth, INFINITY, WHITE},
    searchinfo::{SearchInfo, SearchLimit},
    threadlocal::ThreadData,
};

use rayon::prelude::*;

fn batch_convert(depth: i32, fens: &[String], evals: &mut Vec<i32>) {
    let mut pos = Board::default();
    let mut info = SearchInfo {
        limit: SearchLimit::Infinite,
        print_to_stdout: false,
        ..Default::default()
    };
    let mut t = ThreadData::new();
    pos.set_hash_size(1);
    pos.alloc_tables();
    for fen in fens {
        pos.set_from_fen(fen).unwrap();
        // no NNUE for generating training data.
        let pov_score =
            pos.alpha_beta::<true, true, false>(&mut info, &mut t, Depth::new(depth), -INFINITY, INFINITY);
        let score = if pos.turn() == WHITE { pov_score } else { -pov_score };
        evals.push(score);
    }
}

pub fn wdl_to_nnue<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_file: P1,
    output_file: P2,
) -> std::io::Result<()> {
    let input_file = File::open(input_file)?;
    let output_file = File::create(output_file)?;
    let mut reader = BufReader::new(input_file);
    let mut line = String::new();
    let mut fens = Vec::new();
    let mut outcomes = Vec::new();
    while reader.read_line(&mut line)? > 0 {
        let (fen, outcome) = line.trim().split_once(';').unwrap();
        fens.push(fen.to_string());
        outcomes.push(outcome.parse::<f32>().unwrap());
        line.clear();
    }
    let cores = num_cpus::get();
    let chunk_size = fens.len() / cores;
    let evals = fens
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut inner_evals = Vec::new();
            batch_convert(10, chunk, &mut inner_evals);
            inner_evals
        })
        .flatten()
        .collect::<Vec<_>>();
    let mut output = BufWriter::new(output_file);
    for ((fen, outcome), eval) in fens.into_iter().zip(outcomes).zip(&evals) {
        writeln!(output, "{fen} | {eval} | {outcome:.1}")?;
    }
    Ok(())
}
