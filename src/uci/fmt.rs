use std::fmt;

use crate::{
    chess::piece::Colour,
    evaluation::{MATE_SCORE, TB_WIN_SCORE, is_decisive, is_mate_score},
};

pub struct ScoreFormatWrapper(i32);
impl fmt::Display for ScoreFormatWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if is_mate_score(self.0) {
            let plies_to_mate = MATE_SCORE - self.0.abs();
            let moves_to_mate = (plies_to_mate + 1) / 2;
            if self.0 > 0 {
                write!(f, "mate {moves_to_mate}")
            } else {
                write!(f, "mate -{moves_to_mate}")
            }
        } else if is_decisive(self.0) {
            write!(f, "cp {}", self.0)
        } else {
            write!(f, "cp {}", self.0 * 100 / NORMALISE_TO_PAWN_VALUE)
        }
    }
}
pub const fn format_score(score: i32) -> ScoreFormatWrapper {
    ScoreFormatWrapper(score)
}
pub struct PrettyScoreFormatWrapper(i32, Colour);
impl fmt::Display for PrettyScoreFormatWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            -20..=20 => write!(f, "\u{001b}[0m")?, // drawish, no colour.
            21..=100 => write!(f, "\u{001b}[38;5;10m")?, // slightly better for us, light green.
            -100..=-21 => write!(f, "\u{001b}[38;5;9m")?, // slightly better for them, light red.
            101..=500 => write!(f, "\u{001b}[38;5;2m")?, // clearly better for us, green.
            -10000..=-101 => write!(f, "\u{001b}[38;5;1m")?, // clearly/much better for them, red.
            501..=10000 => write!(f, "\u{001b}[38;5;4m")?, // much better for us, blue.
            _ => write!(f, "\u{001b}[38;5;219m")?, // probably a mate score, pink.
        }
        let white_pov = if self.1 == Colour::White {
            self.0
        } else {
            -self.0
        };
        if is_mate_score(white_pov) {
            let plies_to_mate = MATE_SCORE - white_pov.abs();
            let moves_to_mate = (plies_to_mate + 1) / 2;
            if white_pov > 0 {
                write!(f, "   #{moves_to_mate:<2}")?;
            } else {
                write!(f, "  #-{moves_to_mate:<2}")?;
            }
        } else if is_decisive(white_pov) {
            let plies_to_tb = TB_WIN_SCORE - white_pov.abs();
            if white_pov > 0 {
                write!(f, " +TB{plies_to_tb:<2}")?;
            } else {
                write!(f, " -TB{plies_to_tb:<2}")?;
            }
        } else {
            let white_pov = white_pov * 100 / NORMALISE_TO_PAWN_VALUE;
            let white_pov = white_pov.clamp(-9999, 9999);
            if white_pov == 0 {
                // same as below, but with no sign
                write!(f, "{:6.2}", f64::from(white_pov) / 100.0)?;
            } else {
                // six chars wide: one for the sign, two for the pawn values,
                // one for the decimal point, and two for the centipawn values
                write!(f, "{:+6.2}", f64::from(white_pov) / 100.0)?;
            }
        }
        write!(f, "\u{001b}[0m") // reset
    }
}
pub const fn pretty_format_score(v: i32, c: Colour) -> PrettyScoreFormatWrapper {
    PrettyScoreFormatWrapper(v, c)
}

pub struct HumanTimeFormatWrapper {
    millis: u128,
}
impl fmt::Display for HumanTimeFormatWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let millis = self.millis;
        let seconds = millis / 1000;
        let minutes = seconds / 60;
        let hours = minutes / 60;
        let days = hours / 24;
        if days > 0 {
            write!(f, "{days:2}d{r_hours:02}h", r_hours = hours % 24)
        } else if hours > 0 {
            write!(f, "{hours:2}h{r_minutes:02}m", r_minutes = minutes % 60)
        } else if minutes > 0 {
            write!(f, "{minutes:2}m{r_seconds:02}s", r_seconds = seconds % 60)
        } else if seconds > 0 {
            write!(
                f,
                "{seconds:2}.{r_millis:02}s",
                r_millis = millis % 1000 / 10
            )
        } else {
            write!(f, "{millis:4}ms")
        }
    }
}
pub const fn format_time(millis: u128) -> HumanTimeFormatWrapper {
    HumanTimeFormatWrapper { millis }
}

/// Normalizes the internal value as reported by evaluate or search
/// to the UCI centipawn result used in output. This value is derived from
/// [the WLD model](https://github.com/vondele/WLD_model) such that Viridithas
/// outputs an advantage of 100 centipawns for a position if the engine has a
/// 50% probability to win from this position in selfplay at 16s+0.16s time control.
pub const NORMALISE_TO_PAWN_VALUE: i32 = 229;

pub fn wdl_model(eval: i32, ply: usize) -> (i32, i32, i32) {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    const AS: [f64; 4] = [6.871_558_62, -39.652_263_91, 90.684_603_52, 170.669_963_64];
    const BS: [f64; 4] = [
        -7.198_907_10,
        56.139_471_85,
        -139.910_911_83,
        182.810_074_27,
    ];
    debug_assert_eq!(
        NORMALISE_TO_PAWN_VALUE,
        AS.iter().sum::<f64>().round() as i32,
        "AS sum should be {NORMALISE_TO_PAWN_VALUE} but is {:.2}",
        AS.iter().sum::<f64>()
    );

    let m = std::cmp::min(240, ply) as f64 / 64.0;

    let a = AS[0].mul_add(m, AS[1]).mul_add(m, AS[2]).mul_add(m, AS[3]);
    let b = BS[0].mul_add(m, BS[1]).mul_add(m, BS[2]).mul_add(m, BS[3]);

    let x = f64::clamp(
        f64::from(100 * eval) / f64::from(NORMALISE_TO_PAWN_VALUE),
        -2000.0,
        2000.0,
    );
    let win = 1.0 / (1.0 + f64::exp((a - x) / b));
    let loss = 1.0 / (1.0 + f64::exp((a + x) / b));
    let draw = 1.0 - win - loss;

    // Round to the nearest integer
    (
        (1000.0 * win).round() as i32,
        (1000.0 * draw).round() as i32,
        (1000.0 * loss).round() as i32,
    )
}

pub struct UciWdlFormat {
    pub(crate) eval: i32,
    pub(crate) ply: usize,
}

impl fmt::Display for UciWdlFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (wdl_w, wdl_d, wdl_l) = wdl_model(self.eval, self.ply);
        write!(f, "{wdl_w} {wdl_d} {wdl_l}")
    }
}

pub struct PrettyUciWdlFormat {
    pub eval: i32,
    pub ply: usize,
}

impl fmt::Display for PrettyUciWdlFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #![allow(clippy::cast_possible_truncation)]
        let (wdl_w, wdl_d, wdl_l) = wdl_model(self.eval, self.ply);
        let wdl_w = (f64::from(wdl_w) / 10.0).round() as i32;
        let wdl_d = (f64::from(wdl_d) / 10.0).round() as i32;
        let wdl_l = (f64::from(wdl_l) / 10.0).round() as i32;
        write!(
            f,
            "\u{001b}[38;5;243m{wdl_w:3.0}W {wdl_d:3.0}D {wdl_l:3.0}L\u{001b}[0m",
        )
    }
}

pub fn format_wdl(eval: i32, ply: usize) -> impl fmt::Display {
    UciWdlFormat { eval, ply }
}

pub fn pretty_format_wdl(eval: i32, ply: usize) -> impl fmt::Display {
    PrettyUciWdlFormat { eval, ply }
}

pub struct PrettyCounterFormat(u64);

impl fmt::Display for PrettyCounterFormat {
    #[expect(clippy::cast_precision_loss, clippy::match_overlapping_arm)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            ..1_000 => write!(f, "{:>4}", self.0),
            ..10_000 => write!(f, "{:.1}K", self.0 as f64 / 1_000.0),
            ..1_000_000 => write!(f, "{:>3}K", self.0 / 1_000),
            ..10_000_000 => write!(f, "{:.1}M", self.0 as f64 / 1_000_000.0),
            ..1_000_000_000 => write!(f, "{:>3}M", self.0 / 1_000_000),
            ..10_000_000_000 => write!(f, "{:.1}G", self.0 as f64 / 1_000_000_000.0),
            ..1_000_000_000_000 => write!(f, "{:>3}G", self.0 / 1_000_000_000),
            ..10_000_000_000_000 => write!(f, "{:.1}T", self.0 as f64 / 1_000_000_000_000.0),
            _ => write!(f, "{:>3}T", self.0 / 1_000_000_000_000),
        }
    }
}

pub const fn pretty_format_counter(v: u64) -> impl fmt::Display {
    PrettyCounterFormat(v)
}
