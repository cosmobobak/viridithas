use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
    sync::atomic::{self, AtomicU64},
};

use crate::{
    board::Board,
    definitions::depth::Depth,
    searchinfo::{SearchInfo, SearchLimit},
    threadlocal::ThreadData,
};

fn batch_convert(
    depth: i32,
    fens: &[String],
    evals: &mut Vec<Option<i32>>,
    filter_quiescent: bool,
    counter: &AtomicU64,
    printing_thread: bool,
    start_time: std::time::Instant,
) {
    let mut pos = Board::default();
    let mut t = vec![ThreadData::new()];
    let mut local_ticker = 0;
    pos.set_hash_size(16);
    pos.alloc_tables();
    if printing_thread {
        let c = counter.load(atomic::Ordering::SeqCst);
        #[allow(clippy::cast_precision_loss)]
        let ppersec = c as f64 / start_time.elapsed().as_secs_f64();
        print!("{c: >8} FENs converted. ({ppersec:.2}/s)\r");
        std::io::stdout().flush().unwrap();
    }
    for fen in fens {
        counter.fetch_add(1, atomic::Ordering::SeqCst);
        local_ticker += 1;
        if printing_thread && local_ticker % 25 == 0 {
            let c = counter.load(atomic::Ordering::SeqCst);
            #[allow(clippy::cast_precision_loss)]
            let ppersec = c as f64 / start_time.elapsed().as_secs_f64();
            print!("{c: >8} FENs converted. ({ppersec:.2}/s)\r");
            std::io::stdout().flush().unwrap();
        }
        pos.set_from_fen(fen).unwrap();
        if filter_quiescent && pos.in_check::<{ Board::US }>() {
            evals.push(None);
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
            evals.push(None);
            continue;
        }
        evals.push(Some(score));
    }
    if printing_thread {
        let c = counter.load(atomic::Ordering::SeqCst);
        #[allow(clippy::cast_precision_loss)]
        let ppersec = c as f64 / start_time.elapsed().as_secs_f64();
        print!("{c: >8} FENs converted. ({ppersec:.2}/s)\r");
        std::io::stdout().flush().unwrap();
    }
}

pub fn evaluate_fens<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_file: P1,
    output_file: P2,
    format: Format,
    depth: i32,
    filter_quiescent: bool,
) -> io::Result<()> {
    let input_file = File::open(input_file)?;
    let output_file = File::create(output_file)?;
    let reader = BufReader::new(input_file);
    let mut data_iterator: Box<dyn Iterator<Item = (String, f32)>> = match format {
        Format::OurTexel => Box::new(from_our_texel_format(reader)),
        Format::Marlinflow => Box::new(from_marlinflow_format(reader)),
    };
    let mut output = BufWriter::new(output_file);
    let fens_processed = AtomicU64::new(0);
    let start_time = std::time::Instant::now();
    loop {
        let mut fens = Vec::with_capacity(100_000);
        let mut outcomes = Vec::with_capacity(100_000);
        for _ in 0..100_000 {
            if let Some((fen, outcome)) = data_iterator.next() {
                fens.push(fen);
                outcomes.push(outcome);
            } else {
                break;
            }
        }
        if fens.is_empty() {
            break;
        }
        let evals = parallel_evaluate(&fens, depth, filter_quiescent, &fens_processed, start_time);
        for ((fen, outcome), eval) in fens.into_iter().zip(outcomes).zip(&evals) {
            if let Some(eval) = eval {
                // data is written to conform with marlinflow's data format.
                writeln!(output, "{fen} | {eval} | {outcome:.1}")?;
            }
        }
    }
    Ok(())
}

fn parallel_evaluate(fens: &[String], depth: i32, filter_quiescent: bool, fens_processed: &AtomicU64, start_time: std::time::Instant) -> Vec<Option<i32>> {
    let chunk_size = fens.len() / num_cpus::get() + 1;
    let chunks = fens.chunks(chunk_size).collect::<Vec<_>>();
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut evals = Vec::with_capacity(fens.len());
        for (i, chunk) in chunks.into_iter().enumerate() {
            let printing_thread = i == 0;
            handles.push(s.spawn(move || {
                let mut inner_evals = Vec::new();
                batch_convert(
                    depth,
                    chunk,
                    &mut inner_evals,
                    filter_quiescent,
                    fens_processed,
                    printing_thread,
                    start_time,
                );
                inner_evals
            }));
        }
        for handle in handles {
            evals.extend_from_slice(&handle.join().unwrap());
        }
        evals
    })
}

fn from_our_texel_format(reader: BufReader<File>) -> impl Iterator<Item = (String, f32)> {
    // texel format is <FEN>;<WDL>
    reader.lines().filter_map(|line| {
        line.ok().map(|line| {
            line.split_once(';').map(|(fen, wdl)| (fen.to_string(), wdl.parse().unwrap())).unwrap()
        })
    })
}

fn from_marlinflow_format(reader: BufReader<File>) -> impl Iterator<Item = (String, f32)> {
    // marlinflow format is <FEN> | <EVAL> | <WDL>
    reader.lines().filter_map(|line| {
        line.ok().map(|line| {
            let (fen, rest) = line.split_once('|').unwrap();
            let fen = fen.trim().to_string();
            let outcome = rest.trim().split_once('|').unwrap().1.trim().parse().unwrap();
            (fen, outcome)
        })
    })
}

#[derive(Debug, Clone, Copy)]
pub enum Format {
    OurTexel,
    Marlinflow,
}
