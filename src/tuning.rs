use std::{fs::File, io::{BufReader, BufRead, BufWriter, Write}};

use crate::{board::evaluation::EvalVector, board::Board, searchinfo::SearchInfo};

#[allow(dead_code)]
pub fn annotate_positions(input_fname: &str, output_fname: &str, n_positions: usize, depth: usize) {
    #![allow(clippy::cast_possible_truncation)]
    let i = File::open(input_fname).unwrap();
    let reader = BufReader::new(i);
    let o = File::create(output_fname).unwrap();
    let mut writer = BufWriter::new(o);
    let mut board = Board::new();
    writer.write_all(format!("eval,{}\n", EvalVector::header()).as_bytes()).unwrap();
    let mut counter = 0;
    let mut info = SearchInfo { depth, ..SearchInfo::default() };
    for line in reader.lines().skip(1).take(n_positions) {
        if counter % 10 == 0 {
            print!("{}                   \r", counter);
        }
        counter += 1;
        let line = line.unwrap();
        let (fen, _) = line.split_once(',').unwrap();

        if board.set_from_fen(fen).is_err() {
            println!("{}", fen);
            continue;
        }
        if !board.is_quiet_position() {
            continue;
        }
        let eval_vec = board.eval_vector();
        if eval_vec.valid {
            let eval = board.search_position::<false>(&mut info);
            if eval.abs() > 700 { continue; }
            let eval_positional = eval - eval_vec.material_pst;
            let output = format!("{},{}\n", eval_positional, eval_vec.csvify());
            writer.write_all(output.as_bytes()).unwrap();
        }
    }
    print!("{}                   \r", counter);
    println!();
}