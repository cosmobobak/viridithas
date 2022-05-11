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

    #[inline]
    pub fn insert(&mut self, piece: u8) {
        debug_assert!(self.len < 10, "PieceList is full");
        unsafe {
            *self.data.get_unchecked_mut(self.len as usize) = piece;
        }
        self.len += 1;
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        unsafe { self.data.get_unchecked(..self.len as usize).iter() }
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut u8> {
        unsafe { self.data.get_unchecked_mut(..self.len as usize).iter_mut() }
    }

    #[inline]
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
        debug_assert!(false, "PieceList::remove: piece not found");
    }

    #[inline]
    pub const fn len(&self) -> u8 {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }
}
