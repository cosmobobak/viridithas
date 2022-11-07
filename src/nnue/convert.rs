use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
    sync::atomic::{self, AtomicU64},
};

use crate::{
    board::{Board, evaluation::is_mate_score},
    definitions::depth::Depth,
    searchinfo::{SearchInfo, SearchLimit},
    threadlocal::ThreadData,
};

fn batch_convert(
    depth: i32,
    fens: &[String],
    filter_quiescent: bool,
    counter: &AtomicU64,
    printing_thread: bool,
    start_time: std::time::Instant,
    use_nnue: bool,
) -> Vec<Option<i32>> {
    let mut evals = Vec::with_capacity(fens.len());
    let mut pos = Board::default();
    let mut t = vec![ThreadData::new()];
    t.iter_mut().for_each(|td| td.use_nnue = use_nnue);
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
            print!("{c: >8} FENs converted. ({ppersec:.2}/s)                                        \r");
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
        let (score, bm) = pos.search_position(&mut info, &mut t);
        if filter_quiescent && (!bm.is_quiet() || is_mate_score(score)) {
            evals.push(None);
            continue;
        }
        evals.push(Some(score));
    }
    if printing_thread {
        let c = counter.load(atomic::Ordering::SeqCst);
        #[allow(clippy::cast_precision_loss)]
        let ppersec = c as f64 / start_time.elapsed().as_secs_f64();
        print!("{c: >8} FENs converted. ({ppersec:.2}/s)                                            \r");
        std::io::stdout().flush().unwrap();
    }
    evals
}

pub fn evaluate_fens<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_file: P1,
    output_file: P2,
    format: Format,
    depth: i32,
    filter_quiescent: bool,
    use_nnue: bool,
) -> io::Result<()> {
    let reader = BufReader::new(File::open(input_file)?);
    let mut output = BufWriter::new(File::create(output_file)?);
    let mut data_iterator = match format {
        Format::OurTexel => from_our_texel_format(reader),
        Format::Marlinflow => from_marlinflow_format(reader),
    };
    let fens_processed = AtomicU64::new(0);
    let start_time = std::time::Instant::now();
    let mut fens = Vec::with_capacity(100_000);
    let mut outcomes = Vec::with_capacity(100_000);
    loop {
        fens.clear();
        outcomes.clear();
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
        let evals = parallel_evaluate(&fens, depth, filter_quiescent, use_nnue, &fens_processed, start_time);
        for ((fen, outcome), eval) in fens.iter().zip(&outcomes).zip(&evals) {
            if let Some(eval) = eval {
                // data is written to conform with marlinflow's data format.
                writeln!(output, "{fen} | {eval} | {outcome:.1}")?;
            }
        }
    }
    output.flush()?;
    Ok(())
}

fn parallel_evaluate(fens: &[String], depth: i32, filter_quiescent: bool, use_nnue: bool, fens_processed: &AtomicU64, start_time: std::time::Instant) -> Vec<Option<i32>> {
    let chunk_size = fens.len() / num_cpus::get() + 1;
    let chunks = fens.chunks(chunk_size).collect::<Vec<_>>();
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut evals = Vec::with_capacity(fens.len());
        for (i, chunk) in chunks.into_iter().enumerate() {
            let printing_thread = i == 0;
            handles.push(s.spawn(move || {
                batch_convert(
                    depth,
                    chunk,
                    filter_quiescent,
                    fens_processed,
                    printing_thread,
                    start_time,
                    use_nnue,
                )
            }));
        }
        for handle in handles {
            evals.extend_from_slice(&handle.join().unwrap());
        }
        evals
    })
}

type EvaledFenIterator = Box<dyn Iterator<Item = (String, f32)>>;

fn from_our_texel_format(reader: BufReader<File>) -> EvaledFenIterator {
    // texel format is <FEN>;<WDL>
    Box::new(reader.lines().filter_map(|line| {
        line.ok().map(|line| {
            line.split_once(';').map(|(fen, wdl)| (fen.to_string(), wdl.parse().unwrap())).unwrap()
        })
    }))
}

fn from_marlinflow_format(reader: BufReader<File>) -> EvaledFenIterator {
    // marlinflow format is <FEN> | <EVAL> | <WDL>
    Box::new(reader.lines().filter_map(|line| {
        line.ok().map(|line| {
            let (fen, rest) = line.split_once('|').unwrap();
            let fen = fen.trim().to_string();
            let outcome = rest.trim().split_once('|').unwrap().1.trim().parse().unwrap();
            (fen, outcome)
        })
    }))
}

#[derive(Debug, Clone, Copy)]
pub enum Format {
    OurTexel,
    Marlinflow,
}
