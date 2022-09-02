use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, Mul, Sub, SubAssign},
    str::FromStr,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Depth(i32);

pub const ONE_PLY: Depth = Depth::new(1);

pub const ZERO_PLY: Depth = Depth::new(0);

impl Depth {
    pub(crate) const INNER_INCR_BY_PLY: i32 = 100;

    pub const fn new(depth: i32) -> Self {
        Self(depth * Self::INNER_INCR_BY_PLY)
    }

    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    pub const fn ply_to_horizon(self) -> usize {
        #![allow(clippy::cast_sign_loss)]
        if self.0 <= 0 {
            0
        } else {
            (self.0 / Self::INNER_INCR_BY_PLY) as usize
        }
    }

    pub const fn round(self) -> i32 {
        self.0 / Self::INNER_INCR_BY_PLY
    }

    pub const fn nearest_full_ply(self) -> Self {
        let x = self.0;
        let x = x + Self::INNER_INCR_BY_PLY / 2;
        Self(x - x % Self::INNER_INCR_BY_PLY)
    }

    pub const fn is_exact_ply(self) -> bool {
        self.0 % Self::INNER_INCR_BY_PLY == 0
    }

    pub const fn raw_inner(self) -> i32 {
        self.0
    }

    pub const fn squared(self) -> i32 {
        self.0 * self.0 / Self::INNER_INCR_BY_PLY / Self::INNER_INCR_BY_PLY
    }
}

impl Add<Self> for Depth {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}

impl AddAssign<Self> for Depth {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Add<i32> for Depth {
    type Output = Self;
    fn add(self, other: i32) -> Self::Output {
        Self(self.0 + other * Self::INNER_INCR_BY_PLY)
    }
}

impl AddAssign<i32> for Depth {
    fn add_assign(&mut self, other: i32) {
        *self = *self + other;
    }
}

impl Sub<Self> for Depth {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self(self.0 - other.0)
    }
}

impl SubAssign<Self> for Depth {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl Sub<i32> for Depth {
    type Output = Self;
    fn sub(self, other: i32) -> Self::Output {
        Self(self.0 - other * Self::INNER_INCR_BY_PLY)
    }
}

impl SubAssign<i32> for Depth {
    fn sub_assign(&mut self, other: i32) {
        *self = *self - other;
    }
}

impl Mul<i32> for Depth {
    type Output = Self;
    fn mul(self, other: i32) -> Self::Output {
        Self(self.0 * other)
    }
}

impl Div<i32> for Depth {
    type Output = Self;
    fn div(self, other: i32) -> Self::Output {
        Self(self.0 / other)
    }
}

impl Mul<Depth> for i32 {
    type Output = Self;
    fn mul(self, other: Depth) -> Self::Output {
        self * other.0 / Depth::INNER_INCR_BY_PLY
    }
}

impl From<i32> for Depth {
    fn from(depth: i32) -> Self {
        Self::new(depth)
    }
}

impl From<bool> for Depth {
    fn from(depth: bool) -> Self {
        if depth {
            ONE_PLY
        } else {
            ZERO_PLY
        }
    }
}

impl From<f32> for Depth {
    fn from(depth: f32) -> Self {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let inner_depth = depth * Self::INNER_INCR_BY_PLY as f32;
        Self::from_raw(inner_depth as i32)
    }
}

impl From<Depth> for f64 {
    fn from(depth: Depth) -> Self {
        Self::from(depth.0) / Self::from(Depth::INNER_INCR_BY_PLY)
    }
}

impl From<f64> for Depth {
    fn from(depth: f64) -> Self {
        #![allow(clippy::cast_possible_truncation)]
        let inner_depth = depth * f64::from(Self::INNER_INCR_BY_PLY);
        Self::from_raw(inner_depth as i32)
    }
}

impl From<Depth> for f32 {
    fn from(depth: Depth) -> Self {
        #![allow(clippy::cast_precision_loss)]
        depth.0 as Self / Depth::INNER_INCR_BY_PLY as Self
    }
}

impl TryFrom<Depth> for i16 {
    type Error = <Self as std::convert::TryFrom<i32>>::Error;
    fn try_from(depth: Depth) -> Result<Self, Self::Error> {
        depth.0.try_into()
    }
}

impl From<i16> for Depth {
    fn from(depth: i16) -> Self {
        Self::from_raw(i32::from(depth))
    }
}

impl Display for Depth {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let sign = match self.0.signum() {
            1 | 0 => "",
            -1 => "-",
            _ => unreachable!(),
        };
        write!(
            f,
            "{}{}.{}",
            sign,
            self.0.abs() / Self::INNER_INCR_BY_PLY,
            self.0.abs() % Self::INNER_INCR_BY_PLY
        )
    }
}

impl FromStr for Depth {
    type Err = std::num::ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let floating_repr: f32 = s.parse()?;
        Ok(Self::from(floating_repr))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct CompactDepthStorage(i16);

impl CompactDepthStorage {
    pub const NULL: Self = Self(0);
}

impl TryFrom<Depth> for CompactDepthStorage {
    type Error = <i16 as std::convert::TryFrom<i32>>::Error;
    fn try_from(depth: Depth) -> Result<Self, Self::Error> {
        let inner = depth.0.try_into()?;
        Ok(Self(inner))
    }
}

impl From<CompactDepthStorage> for Depth {
    fn from(depth: CompactDepthStorage) -> Self {
        Self::from_raw(i32::from(depth.0))
    }
}
