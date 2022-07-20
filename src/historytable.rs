use crate::definitions::BOARD_N_SQUARES;

#[derive(Default)]
pub struct HistoryTable {
    table: Box<[[[[i32; BOARD_N_SQUARES]; 6]; BOARD_N_SQUARES]]>
}

impl HistoryTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        if self.table.len() == 0 {
            self.table = vec![[[[0; BOARD_N_SQUARES]; 6]; BOARD_N_SQUARES]; 6].into_boxed_slice();
        } else {
            self.table
                .iter_mut()
                .flatten()
                .flatten()
                .flatten()
                .for_each(|x| *x = 0);
        }
    }

    #[allow(clippy::only_used_in_recursion)] // wtf??
    pub fn add(&mut self, p1: u8, sq1: u8, p2: u8, sq2: u8, score: i32) {
        self.table[p1 as usize][sq1 as usize][p2 as usize][sq2 as usize] += score;
    }

    pub const fn get(&self, p1: u8, sq1: u8, p2: u8, sq2: u8) -> i32 {
        self.table[p1 as usize][sq1 as usize][p2 as usize][sq2 as usize]
    }
}