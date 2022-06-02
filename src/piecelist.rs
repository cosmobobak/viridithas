use crate::definitions::square_name;

#[derive(Clone, Copy)]
pub struct PieceList {
    data: [u8; 10],
    len: u8,
}

impl PartialEq for PieceList {
    fn eq(&self, other: &Self) -> bool {
        self.data[..self.len as usize] == other.data[..other.len as usize]
    }
}

impl Eq for PieceList {}

impl PieceList {
    pub const fn new() -> Self {
        Self {
            data: [0; 10],
            len: 0,
        }
    }

    pub fn first(&self) -> Option<&u8> {
        self.data[..self.len as usize].first()
    }

    pub fn insert(&mut self, sq: u8) {
        debug_assert!(
            self.len < 10, 
            "PieceList is full: [{}]",
            self.data[..self.len as usize]
                .iter()
                .map(|&s| square_name(s).map(std::string::ToString::to_string).unwrap_or(format!("offboard: {}", s)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        debug_assert!(
            !self.data[..self.len as usize].contains(&sq),
            "PieceList already contains square {}: [{}]",
            square_name(sq).map(std::string::ToString::to_string).unwrap_or(format!("offboard: {}", sq)),
            self.data[..self.len as usize]
                .iter()
                .map(|&s| square_name(s).map(std::string::ToString::to_string).unwrap_or(format!("offboard: {}", s)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        unsafe {
            *self.data.get_unchecked_mut(self.len as usize) = sq;
        }
        self.len += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        unsafe { self.data.get_unchecked(..self.len as usize).iter() }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut u8> {
        unsafe { self.data.get_unchecked_mut(..self.len as usize).iter_mut() }
    }

    pub fn remove(&mut self, sq: u8) {
        debug_assert!(self.len > 0, "PieceList is empty");
        let mut idx = 0;
        while idx < self.len {
            if unsafe { *self.data.get_unchecked(idx as usize) } == sq {
                self.len -= 1;
                unsafe {
                    *self.data.get_unchecked_mut(idx as usize) =
                        *self.data.get_unchecked((self.len) as usize);
                    return;
                }
            }
            idx += 1;
        }
        debug_assert!(
            false, 
            "PieceList::remove: piece not found: looking for {} in [{}]", 
            square_name(sq).unwrap_or(&format!("offboard: {}", sq)), 
            self.data[..self.len as usize]
                .iter()
                .map(|&s| square_name(s).map(std::string::ToString::to_string).unwrap_or(format!("offboard: {}", s)))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    pub const fn len(&self) -> u8 {
        self.len
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }
}
