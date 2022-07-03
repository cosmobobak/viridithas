use super::lerp;

use std;

use std::fmt::Display;

use std::iter::Sum;

use std::ops::Mul;

use std::ops::Neg;

use std::ops::SubAssign;

use std::ops::AddAssign;

use std::ops::Sub;

use std::ops::Add;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct S(pub i32, pub i32);

impl Add for S {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sub for S {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl AddAssign for S {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl SubAssign for S {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}

impl Neg for S {
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }
}

impl Mul<i32> for S {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self {
        Self(self.0 * rhs, self.1 * rhs)
    }
}

impl Sum for S {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(0, 0), |acc, x| acc + x)
    }
}

impl Display for S {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "S({}, {})", self.0, self.1)
    }
}

impl S {
    pub const NULL: Self = Self(0, 0);

    pub fn value(self, phase: i32) -> i32 {
        lerp(self.0, self.1, phase)
    }
}
