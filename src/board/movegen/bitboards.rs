
use crate::lookups::SQ120_TO_SQ64;
use crate::lookups::filerank_to_square;

const FILE_1: u64 = 0x8080_8080_8080_8080;
const FILE_8: u64 = 0x0101_0101_0101_0101;
const RANK_2: u64 = 0x0000_0000_0000_FF00;
const RANK_7: u64 = 0x00FF_0000_0000_0000;
const RANK_MID: u64 = 0x0000_FFFF_FFFF_0000;
const BACKRANKS: u64 = 0xFF00_0000_0000_00FF;

/// least significant bit of a u64
/// ```
/// assert_eq!(3, bitboard::lsb(0b00001000));
/// ```
const fn lsb(x: u64) -> u64 {
    x.trailing_zeros() as u64
}

/// Iterator over the squares of a bitboard.
/// The squares are returned in increasing order.
/// ```
/// let bb = 0b010110;
/// let squares = BitLoop::new(bb).collect::<Vec<_>>();
/// assert_eq!(squares, vec![1, 2, 4]);
/// ```
pub struct BitLoop {
    value: u64,
}

impl BitLoop {
    pub const fn new(value: u64) -> Self {
        Self { value }
    }
}

impl Iterator for BitLoop {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.value == 0 {
            None
        } else {
            // faster if we have bmi (maybe)
            let lsb = lsb(self.value);
            self.value ^= 1 << lsb;
            Some(lsb)
        }
    }
}

const fn pawns_notleft() -> u64 {
    !FILE_1
}

const fn pawns_notright() -> u64 {
    !FILE_8
}

#[rustfmt::skip]
const fn pawn_forward<const IS_WHITE: bool>(mask: u64) -> u64 {
    if IS_WHITE { mask << 8 } else { mask >> 8 }
}

#[rustfmt::skip]
const fn pawn_forward_2<const IS_WHITE: bool>(mask: u64) -> u64 {
    if IS_WHITE { mask << 16 } else { mask >> 16 }
}

#[rustfmt::skip]
const fn pawn_backward<const IS_WHITE: bool>(mask: u64) -> u64 {
    if IS_WHITE { mask >> 8 } else { mask << 8 }
}

#[rustfmt::skip]
const fn pawn_backward_2<const IS_WHITE: bool>(mask: u64) -> u64 {
    if IS_WHITE { mask >> 16 } else { mask << 16 }
}

#[rustfmt::skip]
const fn pawn_attack_left<const IS_WHITE: bool>(mask: u64) -> u64 {
    if IS_WHITE { mask << 9 } else { mask >> 7 }
}

#[rustfmt::skip]
const fn pawn_attack_right<const IS_WHITE: bool>(mask: u64) -> u64 {
    if IS_WHITE { mask << 7 } else { mask >> 9 }
}

#[rustfmt::skip]
const fn pawn_invert_left<const IS_WHITE: bool>(mask: u64) -> u64 {
    if !IS_WHITE { mask << 7 } else { mask >> 9 }
}

#[rustfmt::skip]
const fn pawn_invert_right<const IS_WHITE: bool>(mask: u64) -> u64 {
    if !IS_WHITE { mask << 9 } else { mask >> 7 }
}

#[rustfmt::skip]
const fn pawns_first_rank<const IS_WHITE: bool>() -> u64 {
    if IS_WHITE { RANK_2 } else { RANK_7 }
}

#[rustfmt::skip]
const fn pawns_last_rank<const IS_WHITE: bool>() -> u64 {
    if IS_WHITE { RANK_7 } else { RANK_2 }
}

#[derive(Clone, PartialEq, Eq)]
struct BitBoard {
    w_pawns: u64,
    w_knights: u64,
    w_bishops: u64,
    w_rooks: u64,
    w_queens: u64,
    w_king: u64,

    b_pawns: u64,
    b_knights: u64,
    b_bishops: u64,
    b_rooks: u64,
    b_queens: u64,
    b_king: u64,

    white: u64,
    black: u64,
    occupied: u64,
}

impl BitBoard {
    #![allow(clippy::too_many_arguments)]
    pub const fn new(
        bp: u64,
        bn: u64,
        bb: u64,
        br: u64,
        bq: u64,
        bk: u64,
        wp: u64,
        wn: u64,
        wb: u64,
        wr: u64,
        wq: u64,
        wk: u64,
    ) -> Self {
        let white = wp | wk | wb | wr | wq;
        let black = bp | bk | bb | br | bq;
        let occupied = white | black;
        Self {
            w_pawns: wp,
            w_knights: wn,
            w_bishops: wb,
            w_rooks: wr,
            w_queens: wq,
            w_king: wk,

            b_pawns: bp,
            b_knights: bn,
            b_bishops: bb,
            b_rooks: br,
            b_queens: bq,
            b_king: bk,

            white,
            black,
            occupied,
        }
    }

    #[rustfmt::skip]
    pub const fn king<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_king } else { self.b_king }
    }

    #[rustfmt::skip]
    pub const fn enemy_king<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.b_king } else { self.w_king }
    }

    #[rustfmt::skip]
    pub const fn pawns<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_pawns } else { self.b_pawns }
    }

    #[rustfmt::skip]
    pub const fn our_pieces<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.white } else { self.black }
    }

    #[rustfmt::skip]
    pub const fn their_pieces<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.black } else { self.white }
    }

    #[rustfmt::skip]
    pub const fn enemy_rookqueen<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.b_rooks | self.b_queens } else { self.w_rooks | self.w_queens }
    }

    #[rustfmt::skip]
    pub const fn rookqueen<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_rooks | self.w_queens } else { self.b_rooks | self.b_queens }
    }

    #[rustfmt::skip]
    pub const fn enemy_bishopqueen<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.b_bishops | self.b_queens } else { self.w_bishops | self.w_queens }
    }

    #[rustfmt::skip]
    pub const fn bishopqueen<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_bishops | self.w_queens } else { self.b_bishops | self.b_queens }
    }

    #[rustfmt::skip]
    pub const fn enemy_or_empty<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { !self.white } else { !self.black }
    }

    #[rustfmt::skip]
    pub const fn empty(&self) -> u64 {
        !self.occupied
    }

    #[rustfmt::skip]
    pub const fn knights<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_knights } else { self.b_knights }
    }

    #[rustfmt::skip]
    pub const fn rooks<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_rooks } else { self.b_rooks }
    }

    #[rustfmt::skip]
    pub const fn bishops<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_bishops } else { self.b_bishops }
    }

    #[rustfmt::skip]
    pub const fn queens<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_queens } else { self.b_queens }
    }
}

pub fn _write_bb(bb: u64, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for rank in (0..=7).rev() {
        for file in 0..=7 {
            let sq = filerank_to_square(file, rank);
            let sq64 = SQ120_TO_SQ64[sq as usize];
            assert!(
                sq64 < 64,
                "sq64: {}, sq: {}, file: {}, rank: {}",
                sq64,
                sq,
                file,
                rank
            );
            if ((1 << sq64) & bb) == 0 {
                write!(f, ".")?;
            } else {
                write!(f, "X")?;
            }
        }
        writeln!(f)?;
    }
    Ok(())
}