

#[derive(Debug, Clone, Copy)]
pub struct BasicAccumulator<const HIDDEN: usize> {
    pub white: [i16; HIDDEN],
    pub black: [i16; HIDDEN],
}

impl<const HIDDEN: usize> BasicAccumulator<HIDDEN> {
    pub const fn new() -> Self {
        Self {
            white: [0; HIDDEN],
            black: [0; HIDDEN],
        }
    }

    pub fn zero_out(&mut self) {
        self.white.fill(0);
        self.black.fill(0);
    }
}