#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accumulator<const HIDDEN: usize> {
    pub white: [i16; HIDDEN],
    pub black: [i16; HIDDEN],
}

impl<const HIDDEN: usize> Accumulator<HIDDEN> {
    pub const fn new() -> Self {
        Self { white: [0; HIDDEN], black: [0; HIDDEN] }
    }

    pub fn init(&mut self, bias: &[i16; HIDDEN]) {
        self.white.copy_from_slice(bias);
        self.black.copy_from_slice(bias);
    }
}
