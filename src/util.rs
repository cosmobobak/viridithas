// SPDX-License-Identifier: AGPL-3.0-only

use std::{
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::evaluation::MATE_SCORE;

pub const MAX_DEPTH: usize = 128;
pub const INFINITY: i32 = MATE_SCORE + 1;
pub const VALUE_NONE: i32 = INFINITY + 1;
pub const MEGABYTE: usize = 1024 * 1024;

/// A flushing atomic counter to reduce contention.
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

    /// Add one to the local counter.
    /// Flushes to the global counter every `BatchedAtomicCounter::GRANULARITY` calls.
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

    /// Whether all local increments have been published globally.
    pub const fn just_ticked_over(&self) -> bool {
        self.buffer == 0
    }

    pub fn flush(&mut self) {
        self.global.fetch_add(self.buffer, Ordering::Relaxed);
        self.local += self.buffer;
        self.buffer = 0;
    }
}

/// A transparent wrapper that aligns its contents to 64 bytes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[repr(C, align(64))]
pub struct Align<T: ?Sized>(pub T);

impl<T, const SIZE: usize> Deref for Align<[T; SIZE]> {
    type Target = [T; SIZE];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const SIZE: usize> DerefMut for Align<[T; SIZE]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// An unsafe `Send` wrapper around a raw pointer, to transport a
/// pointer to other threads while preserving provenance for MIRI.
pub struct SendPtr<T>(*mut T);

// Safety: Upon the head of the caller of `SendPtr::new`.
unsafe impl<T> Send for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// Wrap a raw pointer so it can be sent across threads.
    ///
    /// # Safety
    ///
    /// This allows you to move a raw pointer to another thread.
    /// This isn’t inherently problematic, but be aware of what
    /// may be thusly permitted, and what horrors may visit you.
    pub const unsafe fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// Recover the wrapped pointer.
    pub const fn get(self) -> *mut T {
        self.0
    }
}
