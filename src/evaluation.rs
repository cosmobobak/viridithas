// The granularity of evaluation in this engine is in centipawns.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::Context;

use crate::{
    chess::{
        board::Board,
        chessmove::Move,
        piece::{Colour, PieceType},
        squareset::SquareSet,
    },
    nnue::network::{self, NNUEParams, NNUEState},
    search::{draw_score, parameters::Config},
    searchinfo::SearchInfo,
    threadlocal::ThreadData,
    util::MAX_DEPTH,
};

/// The value of checkmate.
/// To recover depth-to-mate, we subtract depth (ply) from this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// two less than `MATE_SCORE`.
pub const MATE_SCORE: i32 = i16::MAX as i32 - 367;
pub const fn mate_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH);
    MATE_SCORE - ply as i32
}
pub const fn mated_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH);
    -MATE_SCORE + ply as i32
}
pub const TB_WIN_SCORE: i32 = MATE_SCORE - 1000;
pub const fn tb_win_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH);
    TB_WIN_SCORE - ply as i32
}
pub const fn tb_loss_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH);
    -TB_WIN_SCORE + ply as i32
}

/// A threshold over which scores must be mate.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub const MINIMUM_MATE_SCORE: i32 = MATE_SCORE - 2 * MAX_DEPTH as i32 - 44;
/// A threshold over which scores must be a TB win (or mate).
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub const MINIMUM_TB_WIN_SCORE: i32 = TB_WIN_SCORE - 2 * MAX_DEPTH as i32 - 44;

pub const fn is_mate_score(score: i32) -> bool {
    score.abs() >= MINIMUM_MATE_SCORE
}
pub const fn is_decisive(score: i32) -> bool {
    score.abs() >= MINIMUM_TB_WIN_SCORE
}

pub const SEE_PAWN_VALUE: i32 = 254;
pub const SEE_KNIGHT_VALUE: i32 = 453;
pub const SEE_BISHOP_VALUE: i32 = 458;
pub const SEE_ROOK_VALUE: i32 = 712;
pub const SEE_QUEEN_VALUE: i32 = 1278;
pub const MATERIAL_SCALE_BASE: i32 = 825;

impl Board {
    pub fn material(&self, info: &SearchInfo) -> i32 {
        #![allow(clippy::cast_possible_wrap)]
        let b = &self.state.bbs;
        (info.conf.see_knight_value * b.pieces[PieceType::Knight].count() as i32
            + info.conf.see_bishop_value * b.pieces[PieceType::Bishop].count() as i32
            + info.conf.see_rook_value * b.pieces[PieceType::Rook].count() as i32
            + info.conf.see_queen_value * b.pieces[PieceType::Queen].count() as i32)
            / 32
    }

    pub fn zugzwang_unlikely(&self) -> bool {
        let stm = self.turn();
        let us = self.state.bbs.colours[stm];
        let kings = self.state.bbs.pieces[PieceType::King];
        let pawns = self.state.bbs.pieces[PieceType::Pawn];
        (us & (kings | pawns)) != us
    }

    pub fn estimated_see(&self, conf: &Config, m: Move) -> i32 {
        // initially take the value of the thing on the target square
        let mut value = self.state.mailbox[m.to()].map_or(0, |p| see_value(p.piece_type(), conf));

        if let Some(promo) = m.promotion_type() {
            // if it's a promo, swap a pawn for the promoted piece type
            value += see_value(promo, conf) - conf.see_pawn_value;
        } else if m.is_ep() {
            // for e.p. we will miss a pawn because the target square is empty
            value = conf.see_pawn_value;
        }

        value
    }
}

pub fn evaluate_nnue(t: &ThreadData) -> i32 {
    // get the raw network output
    let v = t.nnue.evaluate(t.nnue_params, &t.board);

    // clamp the value into the valid range.
    // this basically never comes up, but the network will
    // occasionally output OOB values in crazy positions with
    // massive material imbalances.
    v.clamp(-MINIMUM_TB_WIN_SCORE + 1024, MINIMUM_TB_WIN_SCORE - 1024)
}

pub fn evaluate(t: &mut ThreadData, nodes: u64) -> i32 {
    // detect draw by insufficient material
    if t.board.state.bbs.pieces[PieceType::Pawn] == SquareSet::EMPTY
        && t.board.state.bbs.is_material_draw()
    {
        return if t.board.turn() == Colour::White {
            draw_score(t, nodes, t.board.turn())
        } else {
            -draw_score(t, nodes, t.board.turn())
        };
    }
    // apply all in-waiting updates to generate a valid
    // neural network accumulator state.
    t.nnue.force(&t.board, t.nnue_params);
    // run the neural network evaluation
    evaluate_nnue(t)
}

pub const fn see_value(piece_type: PieceType, conf: &Config) -> i32 {
    match piece_type {
        PieceType::Pawn => conf.see_pawn_value,
        PieceType::Knight => conf.see_knight_value,
        PieceType::Bishop => conf.see_bishop_value,
        PieceType::Rook => conf.see_rook_value,
        PieceType::Queen => conf.see_queen_value,
        PieceType::King => 0,
    }
}

pub fn eval_stats(input: &Path) -> anyhow::Result<()> {
    let f = File::open(input).with_context(|| format!("Failed to open {}", input.display()))?;
    let mut board = Board::default();
    let nnue_params = NNUEParams::decompress_and_alloc()?;
    let mut nnue = NNUEState::new(&board, nnue_params);

    let mut total = 0i128;
    let mut count = 0i128;
    let mut abs_total = 0i128;
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    let mut sq_total = 0i128;

    let lines = BufReader::new(f)
        .lines()
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| "Failed to read lines from input file.")?;

    let file_len = lines.len();

    for (i, line) in lines.into_iter().enumerate() {
        // extract the first 6 fields as FEN
        let end_idx = line
            .match_indices(' ')
            .nth(5)
            .map(|(idx, _)| idx)
            .with_context(|| format!("Failed to parse FEN from line {}: {}", i + 1, line))?;
        let fen = &line[..end_idx];
        board.set_from_fen(fen)?;
        nnue.reinit_from(&board, nnue_params);
        let eval = if board.in_check() {
            continue;
        } else {
            nnue.evaluate(nnue_params, &board)
        };

        count += 1;
        total += i128::from(eval);
        abs_total += i128::from(eval.abs());
        sq_total += i128::from(eval) * i128::from(eval);
        if eval < min {
            min = eval;
        }
        if eval > max {
            max = eval;
        }

        if i % 1024 == 0 {
            print!("\rProcessed {:>10}/{}.", i + 1, file_len);
        }
    }

    println!("\rProcessed {file_len:>10}/{file_len}.");

    println!(" EVALUATION STATISTICS:");

    println!("    COUNT: {count:>7}");
    #[expect(clippy::cast_precision_loss)]
    if count > 0 {
        let mean = total as f64 / count as f64;
        let abs_mean = abs_total as f64 / count as f64;
        let mean_squared = mean * mean;
        let variance = (sq_total as f64 / count as f64) - mean_squared;
        let stddev = variance.sqrt();
        let min = f64::from(min);
        let max = f64::from(max);
        println!("     MEAN: {mean:>10.2}");
        println!(" ABS MEAN: {abs_mean:>10.2}");
        println!("   STDDEV: {stddev:>10.2}");
        println!("      MIN: {min:>10.2}");
        println!("      MAX: {max:>10.2}");

        // delenda's eval scale is 400, generating a mean absolute eval of ~780.49
        // compute the multiplier we'd need to hit that target - e.g. if our abs-mean
        // is 390, we should have an eval scale of 800 to push it up to 780. conversely,
        // if our abs-mean is 1560, we should have an eval scale of 200 to bring it down to 780.
        let delenda_abs_mean = 780.489_583_154_229_1;
        let scale = delenda_abs_mean / abs_mean * f64::from(network::SCALE);

        println!("  DELENDA SCALING FACTOR: {scale:.6}");
    }

    Ok(())
}
