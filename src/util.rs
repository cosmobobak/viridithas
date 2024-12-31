pub mod depth;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::chess::board::evaluation::MATE_SCORE;

pub const BOARD_N_SQUARES: usize = 64;
pub const MAX_DEPTH: i32 = 128;
pub const MAX_PLY: usize = MAX_DEPTH as usize;
pub const INFINITY: i32 = MATE_SCORE + 1;
pub const VALUE_NONE: i32 = INFINITY + 1;
pub const MEGABYTE: usize = 1024 * 1024;

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
