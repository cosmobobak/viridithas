use super::Align;

#[derive(Debug, Clone, Copy)]
pub struct Accumulator<const HIDDEN: usize> {
    pub white: Align<[i16; HIDDEN]>,
    pub black: Align<[i16; HIDDEN]>,
}

impl<const HIDDEN: usize> Accumulator<HIDDEN> {
    pub const fn new() -> Self {
        Self { white: Align([0; HIDDEN]), black: Align([0; HIDDEN]) }
    }

    pub fn init(&mut self, bias: &[i16; HIDDEN]) {
        self.white.copy_from_slice(bias);
        self.black.copy_from_slice(bias);
    }
}
