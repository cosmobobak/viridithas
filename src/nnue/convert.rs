use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write, self},
    path::Path, sync::atomic::{AtomicU64, self},
};

use crate::{
    board::Board,
    definitions::depth::Depth,
    searchinfo::{SearchInfo, SearchLimit},
    threadlocal::ThreadData,
};

fn batch_convert(depth: i32, fens: &[String], evals: &mut Vec<i32>, filter_quiescent: bool, counter: &AtomicU64, printing_thread: bool) {
    let mut pos = Board::default();
    let mut t = vec![ThreadData::new()];
    pos.set_hash_size(16);
    pos.alloc_tables();
    if printing_thread {
        print!("{c: >8} FENs converted.\r", c = counter.load(atomic::Ordering::SeqCst));
    }
    for fen in fens {
        if printing_thread && counter.load(atomic::Ordering::SeqCst) % 1000 == 0 {
            print!("{c: >8} FENs converted.\r", c = counter.load(atomic::Ordering::SeqCst));
        }
        pos.set_from_fen(fen).unwrap();
        if filter_quiescent && pos.in_check::<{ Board::US }>() {
            continue;
        }
        // no NNUE for generating training data.
        t.iter_mut().for_each(|thread_data| thread_data.nnue.refresh_acc(&pos));
        let mut info = SearchInfo {
            print_to_stdout: false,
            limit: SearchLimit::Depth(Depth::new(depth)),
            ..SearchInfo::default()
        };
        let (score, bm) = pos.search_position::<false>(&mut info, &mut t);
        if filter_quiescent && !bm.is_quiet() {
            continue;
        }
        evals.push(score);  
        counter.fetch_add(1, atomic::Ordering::SeqCst);
    }
    if printing_thread {
        println!("{} FENs converted.        ", counter.load(atomic::Ordering::SeqCst));
    }
}

pub fn evaluate_fens<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_file: P1,
    output_file: P2,
    format: Format,
    filter_quiescent: bool,
) -> io::Result<()> {
    let input_file = File::open(input_file)?;
    let output_file = File::create(output_file)?;
    let reader = BufReader::new(input_file);
    let (fens, outcomes) = match format {
        Format::OurTexel => from_our_texel_format(reader)?,
        Format::Marlinflow => from_marlinflow_format(reader)?,
    };
    let evals = parallel_evaluate(&fens, filter_quiescent);
    let mut output = BufWriter::new(output_file);
    for ((fen, outcome), eval) in fens.into_iter().zip(outcomes).zip(&evals) {
        // data is written to conform with marlinflow's data format.
        writeln!(output, "{fen} | {eval} | {outcome:.1}")?;
    }
    Ok(())
}

fn parallel_evaluate(fens: &[String], filter_quiescent: bool) -> Vec<i32> {
    let chunk_size = fens.len() / num_cpus::get() + 1;
    let chunks = fens.chunks(chunk_size).collect::<Vec<_>>();
    let counter = AtomicU64::new(0);
    let counter_ref = &counter;
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut evals = Vec::with_capacity(fens.len());
        for (i, chunk) in chunks.into_iter().enumerate() {
            let printing_thread = i == 0;
            handles.push(s.spawn(move || {
                let mut inner_evals = Vec::new();
                batch_convert(10, chunk, &mut inner_evals, filter_quiescent, counter_ref, printing_thread);
                inner_evals
            }));
        }
        for handle in handles {
            evals.extend_from_slice(&handle.join().unwrap());
        }
        evals
    })
}

fn from_our_texel_format(mut reader: BufReader<File>) -> io::Result<(Vec<String>, Vec<f32>)> {
    let mut line = String::new();
    let mut fens = Vec::new();
    let mut outcomes = Vec::new();
    // texel format is <FEN>;<WDL>
    while reader.read_line(&mut line)? > 0 {
        let (fen, outcome) = line.trim().split_once(';').unwrap();
        fens.push(fen.to_string());
        outcomes.push(outcome.parse::<f32>().unwrap());
        line.clear();
    }
    Ok((fens, outcomes))
}

fn from_marlinflow_format(mut reader: BufReader<File>) -> io::Result<(Vec<String>, Vec<f32>)> {
    let mut line = String::new();
    let mut fens = Vec::new();
    let mut outcomes = Vec::new();
    // marlinflow format is <FEN> | <EVAL> | <WDL>
    while reader.read_line(&mut line)? > 0 {
        let (fen, rest) = line.trim().split_once('|').unwrap();
        let fen = fen.trim();
        let outcome = rest.trim().split_once('|').unwrap().1.trim();
        fens.push(fen.to_string());
        outcomes.push(outcome.parse::<f32>().unwrap());
        line.clear();
    }
    Ok((fens, outcomes))
}

#[derive(Debug, Clone, Copy)]
pub enum Format {
    OurTexel,
    Marlinflow,
}