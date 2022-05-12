use crate::lookups::{filerank_to_square, SQ120_TO_SQ64};

#[inline]
pub fn pop_lsb(bb: &mut u64) -> u32 {
    let lsb = bb.trailing_zeros();
    *bb &= *bb - 1;
    lsb
}

pub fn _write_bb(bb: u64, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for rank in (0..=7).rev() {
        for file in 0..=7 {
            let sq = filerank_to_square(file, rank);
            let sq64 = SQ120_TO_SQ64[sq as usize];
            assert!(
                sq64 < 64,
                "sq64: {}, sq: {}, file: {}, rank: {}",
                sq64,
                sq,
                file,
                rank
            );
            if ((1 << sq64) & bb) == 0 {
                write!(f, ".")?;
            } else {
                write!(f, "X")?;
            }
        }
        writeln!(f)?;
    }
    Ok(())
}
