use crate::chessmove::Move;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PVEntry {
    pub key: u64,
    pub best_move: Move,
}

#[derive(Clone, PartialEq, Eq)]
pub struct PVTable {
    entries: Vec<PVEntry>,
    table_size: usize,
}

impl PVTable {
    const DEFAULT_PV_SIZE: usize = 0x0010_0000 * 2;

    pub fn new() -> Self {
        Self::with_capacity_bytes(Self::DEFAULT_PV_SIZE)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: vec![
                PVEntry {
                    key: 0,
                    best_move: Move::null(),
                };
                capacity
            ],
            table_size: capacity,
        }
    }

    pub fn with_capacity_bytes(bytes: usize) -> Self {
        let capacity = bytes / std::mem::size_of::<PVEntry>();
        Self {
            entries: vec![
                PVEntry {
                    key: 0,
                    best_move: Move::null(),
                };
                capacity
            ],
            table_size: capacity,
        }
    }

    pub fn store_pv_move(&mut self, key: u64, best_move: Move) {
        let idx = key % self.table_size as u64;
        debug_assert!(idx < self.table_size as u64);

        unsafe {
            let idx: usize = idx.try_into().unwrap_unchecked();
            *self.entries.get_unchecked_mut(idx) = PVEntry { key, best_move };
        }
    }

    pub fn clear(&mut self) {
        self.entries.fill(PVEntry {
            key: 0,
            best_move: Move::null(),
        });
    }

    pub fn probe_pv_move(&self, key: u64) -> Option<&Move> {
        let idx = key % self.table_size as u64;
        debug_assert!(idx < self.table_size as u64);

        unsafe {
            let idx: usize = idx.try_into().unwrap_unchecked();
            let entry = self.entries.get_unchecked(idx);
            if entry.key == key {
                Some(&entry.best_move)
            } else {
                None
            }
        }
    }
}
