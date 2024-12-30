pub mod depth;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::chess::{board::evaluation::MATE_SCORE, squareset::SquareSet, types::Square};

pub const BOARD_N_SQUARES: usize = 64;
pub const MAX_DEPTH: i32 = 128;
pub const MAX_PLY: usize = MAX_DEPTH as usize;
pub const INFINITY: i32 = MATE_SCORE + 1;
pub const VALUE_NONE: i32 = INFINITY + 1;
pub const MEGABYTE: usize = 1024 * 1024;

const fn in_between(sq1: Square, sq2: Square) -> SquareSet {
    const M1: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    const A2A7: u64 = 0x0001_0101_0101_0100;
    const B2G7: u64 = 0x0040_2010_0804_0200;
    const H1B7: u64 = 0x0002_0408_1020_4080;
    let sq1 = sq1.index();
    let sq2 = sq2.index();
    let btwn = (M1 << sq1) ^ (M1 << sq2);
    let file = ((sq2 & 7).wrapping_add((sq1 & 7).wrapping_neg())) as u64;
    let rank = (((sq2 | 7).wrapping_sub(sq1)) >> 3) as u64;
    let mut line = ((file & 7).wrapping_sub(1)) & A2A7;
    line += 2 * ((rank & 7).wrapping_sub(1) >> 58);
    line += ((rank.wrapping_sub(file) & 15).wrapping_sub(1)) & B2G7;
    line += ((rank.wrapping_add(file) & 15).wrapping_sub(1)) & H1B7;
    line = line.wrapping_mul(btwn & btwn.wrapping_neg());
    SquareSet::from_inner(line & btwn)
}

pub static RAY_BETWEEN: [[SquareSet; 64]; 64] = {
    let mut res = [[SquareSet::EMPTY; 64]; 64];
    let mut from = Square::A1;
    loop {
        let mut to = Square::A1;
        loop {
            res[from.index()][to.index()] = in_between(from, to);
            let Some(next) = to.add(1) else {
                break;
            };
            to = next;
        }
        let Some(next) = from.add(1) else {
            break;
        };
        from = next;
    }
    res
};

#[derive(Debug, Clone, Copy)]
pub struct BatchedAtomicCounter<'a> {
    buffer: u64,
    global: &'a AtomicU64,
    local: u64,
}

impl<'a> BatchedAtomicCounter<'a> {
    const GRANULARITY: u64 = 1024;

    pub const fn new(global: &'a AtomicU64) -> Self {
        Self {
            buffer: 0,
            global,
            local: 0,
        }
    }

    pub fn increment(&mut self) {
        self.buffer += 1;
        if self.buffer >= Self::GRANULARITY {
            self.global.fetch_add(self.buffer, Ordering::Relaxed);
            self.local += self.buffer;
            self.buffer = 0;
        }
    }

    pub fn get_global(&self) -> u64 {
        self.global.load(Ordering::Relaxed) + self.buffer
    }

    pub const fn get_buffer(&self) -> u64 {
        self.buffer
    }

    pub const fn get_local(&self) -> u64 {
        self.local + self.buffer
    }

    pub fn reset(&mut self) {
        self.buffer = 0;
        self.global.store(0, Ordering::Relaxed);
        self.local = 0;
    }

    pub const fn just_ticked_over(&self) -> bool {
        self.buffer == 0
    }
}

/// Polyfill for backwards compatibility with old rust compilers.
#[inline]
pub const fn from_ref<T>(r: &T) -> *const T
where
    T: ?Sized,
{
    r
}

/// Polyfill for backwards compatibility with old rust compilers.
#[inline]
pub fn from_mut<T>(r: &mut T) -> *mut T
where
    T: ?Sized,
{
    r
}

mod tests {
    #[test]
    fn square_flipping() {
        use super::Square;

        assert_eq!(Square::A1.flip_rank(), Square::A8);
        assert_eq!(Square::H1.flip_rank(), Square::H8);
        assert_eq!(Square::A8.flip_rank(), Square::A1);
        assert_eq!(Square::H8.flip_rank(), Square::H1);

        assert_eq!(Square::A1.flip_file(), Square::H1);
        assert_eq!(Square::H1.flip_file(), Square::A1);
        assert_eq!(Square::A8.flip_file(), Square::H8);
        assert_eq!(Square::H8.flip_file(), Square::A8);
    }

    #[test]
    fn ray_test() {
        use super::{Square, RAY_BETWEEN};
        use crate::chess::squareset::SquareSet;
        assert_eq!(RAY_BETWEEN[Square::A1][Square::A1], SquareSet::EMPTY);
        assert_eq!(RAY_BETWEEN[Square::A1][Square::B1], SquareSet::EMPTY);
        assert_eq!(RAY_BETWEEN[Square::A1][Square::C1], Square::B1.as_set());
        assert_eq!(
            RAY_BETWEEN[Square::A1][Square::D1],
            Square::B1.as_set() | Square::C1.as_set()
        );
        assert_eq!(RAY_BETWEEN[Square::B1][Square::D1], Square::C1.as_set());
        assert_eq!(RAY_BETWEEN[Square::D1][Square::B1], Square::C1.as_set());

        for from in Square::all() {
            for to in Square::all() {
                assert_eq!(RAY_BETWEEN[from][to], RAY_BETWEEN[to][from]);
            }
        }
    }

    #[test]
    fn ray_diag_test() {
        use super::{Square, RAY_BETWEEN};
        let ray = RAY_BETWEEN[Square::B5][Square::E8];
        assert_eq!(ray, Square::C6.as_set() | Square::D7.as_set());
    }
}
