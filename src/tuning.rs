use std::{fs::File, io::{BufReader, BufRead, BufWriter, Write}};

use crate::{evaluation::EvalVector, errors::FenParseError, board::Board};

pub fn eval_vec_for_fen(fen: &str, board: &mut Board) -> Result<EvalVector, FenParseError> {
    board.set_from_fen(fen)?;
    Ok(board.evaluate_recorded())
}

pub fn eval_positions(fens: &[&str]) -> Vec<EvalVector> {
    let mut board = Board::new();
    fens.iter()
        .filter_map(|fen| eval_vec_for_fen(fen, &mut board).ok())
        .collect()
}

pub fn annotate_positions(input_fname: &str, output_fname: &str) {
    #![allow(clippy::cast_possible_truncation)]
    let i = File::open(input_fname).unwrap();
    let reader = BufReader::new(i);
    let o = File::create(output_fname).unwrap();
    let mut writer = BufWriter::new(o);
    let mut board = Board::new();
    writer.write_all(format!("eval,{}\n", EvalVector::header()).as_bytes()).unwrap();
    for line in reader.lines() {
        let line = line.unwrap();
        let (fen, eval) = line.split_once(", ").unwrap();
        let eval = eval.trim().parse::<f64>().unwrap();
        let eval_millipawns = (eval * 1000.0) as i32;
        let eval_vec = eval_vec_for_fen(fen, &mut board).unwrap();
        if eval_vec.valid {
            let output = format!("{},{}\n", eval_millipawns, eval_vec.csvify());
            writer.write_all(output.as_bytes()).unwrap();
        }
    }
}