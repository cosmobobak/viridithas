use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
    sync::atomic::{self, AtomicU64}, collections::hash_map::DefaultHasher, ops::Range,
};

use crate::{
    board::{Board, evaluation::is_mate_score},
    definitions::depth::Depth,
    searchinfo::{SearchInfo, SearchLimit},
    threadlocal::ThreadData,
};

fn batch_convert<const USE_NNUE: bool>(
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
    for thread in &mut t {
        thread.alloc_tables();
    }
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
        let (score, bm) = pos.search_position::<USE_NNUE>(&mut info, &mut t);
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

/// A container to optimise the comparison of strings.
struct QuicklySortableString {
    /// The string to be sorted.
    pub string: String,
    /// The substring that we are deduping based on.
    pub view: Range<usize>,
    /// The hash of the substring.
    pub hash: u64,
}

impl QuicklySortableString {
    pub fn new(string: String, view: Range<usize>) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        let interesting_range = &string[view.clone()];
        interesting_range.hash(&mut hasher);
        Self {
            string,
            view,
            hash: hasher.finish(),
        }
    }

    /// Compare the substring of the two strings.
    pub fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hash.cmp(&other.hash).then_with(|| {
            let a = &self.string[self.view.clone()];
            let b = &other.string[other.view.clone()];
            a.cmp(b)
        })
    }

    /// Get a reference to the backing string.
    pub fn as_str(&self) -> &str {
        &self.string
    }
}

/// Deduplicate a dataset by board.
/// This deduplicates based on the first two parts of FENs, which are the board and the side to move.
pub fn dedup<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_file: P1,
    output_file: P2,
) -> io::Result<()> {
    let reader = BufReader::new(File::open(input_file)?);
    let mut output = BufWriter::new(File::create(output_file)?);
    let start_time = std::time::Instant::now();

    // would be cool to use an arena allocator for the Strings.
    let mut data = reader.lines().filter_map(|line| {
        line.ok().map(|line| {
            let split_index = line.bytes().position(|b| b == b' ').unwrap(); // the space between the board part and the "w/b" part.
            QuicklySortableString::new(line, 0..(split_index + 2))
        })
    }).collect::<Vec<_>>();

    let n = data.len();

    data.sort_unstable_by(QuicklySortableString::cmp);
    data.dedup_by(|a, b| a.cmp(b) == std::cmp::Ordering::Equal);

    // write deduplicated data to output file.
    for line in data {
        let line = line.as_str();
        writeln!(output, "{line}")?;
    }

    let res = output.flush();

    let elapsed = start_time.elapsed();
    println!("Deduplicated {n} lines in {}.{:03}s", elapsed.as_secs(), elapsed.subsec_millis());

    res
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
                let mut inner_evals = Vec::new();
                if use_nnue {
                    batch_convert::<true>(
                        depth,
                        chunk,
                        &mut inner_evals,
                        filter_quiescent,
                        fens_processed,
                        printing_thread,
                        start_time,
                    );
                } else {
                    batch_convert::<false>(
                        depth,
                        chunk,
                        &mut inner_evals,
                        filter_quiescent,
                        fens_processed,
                        printing_thread,
                        start_time,
                    );
                }
                inner_evals
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
