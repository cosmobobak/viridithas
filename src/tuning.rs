use std::{fs::File, io::{BufReader, BufRead, BufWriter, Write}};

use crate::{evaluation::EvalVector, errors::FenParseError, board::Board};

pub fn eval_vec_for_fen(fen: &str, board: &mut Board) -> Result<EvalVector, FenParseError> {
    board.set_from_fen(fen)?;
    if !board.is_quiet_position() {
        let mut vec_out = EvalVector::new();
        vec_out.valid = false;
        return Ok(vec_out);
    }
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
    let mut counter = 0;
    for line in reader.lines().skip(1) {
        if counter % 1000 == 0 {
            println!("{}                   \r", counter);
        }
        let line = line.unwrap();
        let (fen, evalstr) = line.split_once(',').unwrap();
        if evalstr.contains('#') { continue; }
        let evalstr = evalstr.trim_matches(|c| c == ' ' || c == '\n' || c == '+');
        let eval = evalstr.parse::<i32>();
        if eval.is_err() { println!("{}", evalstr); continue; }
        let eval = eval.unwrap();
        if eval.abs() > 1500 { continue; }
        let eval_millipawns = eval * 10;
        let eval_vec = eval_vec_for_fen(fen, &mut board).unwrap();
        if eval_vec.valid {
            let output = format!("{},{}\n", eval_millipawns, eval_vec.csvify());
            writer.write_all(output.as_bytes()).unwrap();
        }
        counter += 1;
    }
    println!("{}                   \r", counter);
}