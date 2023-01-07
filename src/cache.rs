
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InCheckCache {
    in_check: bool,
    key: u64,
}

impl InCheckCache {
    pub const fn new() -> Self {
        Self {
            in_check: false,
            key: 0,
        }
    }

    pub fn set(&mut self, key: u64, in_check: bool) {
        self.key = key;
        self.in_check = in_check;
    }

    pub const fn get(&self, key: u64) -> Option<bool> {
        if self.key == key {
            Some(self.in_check)
        } else {
            None
        }
    }
}