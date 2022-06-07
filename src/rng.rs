const SEED: u128 = 0x246C_CB2D_3B40_2853_9918_0A6D_BC3A_F444;
pub struct XorShiftState {
    pub state: u128,
}

impl XorShiftState {
    pub const fn new() -> Self {
        Self { state: SEED }
    }

    /// Generates the next random number in the sequence, consuming self
    /// This is done to allow for const evaluation.
    pub const fn next_self(mut self) -> (u64, Self) {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        #[allow(clippy::cast_possible_truncation)]
        let r = x as u64; // truncation is the intended behavior here.
        let r = r ^ (x >> 64) as u64; // add in the high bits.
        (r, self)
    }

    /// Generates the next random number in the sequence.
    pub fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        #[allow(clippy::cast_possible_truncation)]
        let r = x as u64; // truncation is the intended behavior here.
        r ^ (x >> 64) as u64 // add in the high bits.
    }

    /// Generates a random number with only a few bits set.
    /// This will advance the generator by three steps.
    pub fn random_few_bits(&mut self) -> u64 {
        let first = self.next();
        let second = self.next();
        let third = self.next();

        first & second & third
    }
}
