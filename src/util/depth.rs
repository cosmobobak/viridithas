
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct CompactDepthStorage(u8);

impl CompactDepthStorage {
    pub const NULL: Self = Self(0);
}

impl CompactDepthStorage {
    pub const fn inner(self) -> u8 {
        self.0
    }
}

impl TryFrom<i32> for CompactDepthStorage {
    type Error = <u8 as std::convert::TryFrom<i32>>::Error;
    fn try_from(depth: i32) -> Result<Self, Self::Error> {
        let inner = depth.try_into()?;
        Ok(inner)
    }
}

impl From<CompactDepthStorage> for i32 {
    fn from(depth: CompactDepthStorage) -> Self {
        Self::from(depth.0)
    }
}
