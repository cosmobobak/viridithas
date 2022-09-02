use std::fmt::Display;

use crate::definitions::depth::Depth;

use super::{
    ASPIRATION_WINDOW, RFP_DEPTH, RFP_IMPROVING_MARGIN, RFP_MARGIN,
    FUTILITY_COEFF_0, FUTILITY_COEFF_1, FUTILITY_COEFF_2, FUTILITY_DEPTH, LMP_BASE_MOVES,
    LMP_MAX_DEPTH, LMR_BASE, LMR_DIVISION, NMP_IMPROVING_MARGIN,
    NMP_BASE_REDUCTION, SEE_TACTICAL_MARGIN, SEE_DEPTH, SEE_QUIET_MARGIN,
    SINGULARITY_DEPTH, TT_REDUCTION_DEPTH,
};

pub struct SearchParams {
    pub aspiration_window: i32,
    pub rfp_margin: i32,
    pub rfp_improving_margin: i32,
    pub nmp_improving_margin: i32,
    pub see_quiet_margin: i32,
    pub see_tactical_margin: i32,
    pub lmp_base_moves: i32,
    pub futility_coeff_2: i32,
    pub futility_coeff_1: i32,
    pub futility_coeff_0: i32,
    pub rfp_depth: Depth,
    pub nmp_base_reduction: Depth,
    pub lmp_depth: Depth,
    pub tt_reduction_depth: Depth,
    pub futility_depth: Depth,
    pub singularity_depth: Depth,
    pub see_depth: Depth,
    pub lmr_base: f64,
    pub lmr_division: f64,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            aspiration_window: ASPIRATION_WINDOW,
            rfp_margin: RFP_MARGIN,
            rfp_improving_margin: RFP_IMPROVING_MARGIN,
            nmp_improving_margin: NMP_IMPROVING_MARGIN,
            see_quiet_margin: SEE_QUIET_MARGIN,
            see_tactical_margin: SEE_TACTICAL_MARGIN,
            lmp_base_moves: LMP_BASE_MOVES,
            futility_coeff_2: FUTILITY_COEFF_2,
            futility_coeff_1: FUTILITY_COEFF_1,
            futility_coeff_0: FUTILITY_COEFF_0,
            rfp_depth: RFP_DEPTH,
            nmp_base_reduction: NMP_BASE_REDUCTION,
            lmp_depth: LMP_MAX_DEPTH,
            tt_reduction_depth: TT_REDUCTION_DEPTH,
            futility_depth: FUTILITY_DEPTH,
            singularity_depth: SINGULARITY_DEPTH,
            see_depth: SEE_DEPTH,
            lmr_base: LMR_BASE,
            lmr_division: LMR_DIVISION,
        }
    }
}

impl Display for SearchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Search parameters:")?;
        for (id, value) in self.ids_with_values() {
            writeln!(f, "    {}: {}", id, value)?;
        }
        Ok(())
    }
}


macro_rules! id_parser_gen {
    ($($option:ident = [$($field:tt)*]),*) => {
        vec![$(
            (stringify!($option), Box::new(|s: &str| {
                if let Ok(res) = s.parse() {
                    $($field)* = res;
                } else {
                    return Err(format!("Invalid value for {}: {}", stringify!($option), s).into());
                }
                Ok(())
            })),)
            *
        ]
    }
}

macro_rules! id_value_gen {
    ($($option:ident, [$($field:tt)*]),*) => {
        vec![$(
            (stringify!($option), $($field)*),)
            *
        ]
    }
}

type LazyFieldParser<'a> = Box<dyn FnMut(&str) -> Result<(), Box<dyn std::error::Error>> + 'a>;

impl SearchParams {
    pub fn ids_with_parsers(&mut self) -> Vec<(&str, LazyFieldParser)> {
        id_parser_gen![
            ASPIRATION_WINDOW = [self.aspiration_window],
            RFP_MARGIN = [self.rfp_margin],
            RFP_IMPROVING_MARGIN = [self.rfp_improving_margin],
            NMP_IMPROVING_MARGIN = [self.nmp_improving_margin],
            SEE_QUIET_MARGIN = [self.see_quiet_margin],
            SEE_TACTICAL_MARGIN = [self.see_tactical_margin],
            LMP_BASE_MOVES = [self.lmp_base_moves],
            FUTILITY_COEFF_2 = [self.futility_coeff_2],
            FUTILITY_COEFF_1 = [self.futility_coeff_1],
            FUTILITY_COEFF_0 = [self.futility_coeff_0],
            RFP_DEPTH = [self.rfp_depth],
            NMP_BASE_REDUCTION = [self.nmp_base_reduction],
            LMP_MAX_DEPTH = [self.lmp_depth],
            TT_REDUCTION_DEPTH = [self.tt_reduction_depth],
            FUTILITY_DEPTH = [self.futility_depth],
            SINGULARITY_DEPTH = [self.singularity_depth],
            SEE_DEPTH = [self.see_depth],
            LMR_BASE = [self.lmr_base],
            LMR_DIVISION = [self.lmr_division]
        ]
    }

    pub fn ids_with_values(&self) -> Vec<(&str, f64)> {
        id_value_gen![
            ASPIRATION_WINDOW, [self.aspiration_window.into()],
            RFP_MARGIN, [self.rfp_margin.into()],
            RFP_IMPROVING_MARGIN, [self.rfp_improving_margin.into()],
            NMP_IMPROVING_MARGIN, [self.nmp_improving_margin.into()],
            SEE_QUIET_MARGIN, [self.see_quiet_margin.into()],
            SEE_TACTICAL_MARGIN, [self.see_tactical_margin.into()],
            LMP_BASE_MOVES, [self.lmp_base_moves.into()],
            FUTILITY_COEFF_2, [self.futility_coeff_2.into()],
            FUTILITY_COEFF_1, [self.futility_coeff_1.into()],
            FUTILITY_COEFF_0, [self.futility_coeff_0.into()],
            RFP_DEPTH, [self.rfp_depth.into()],
            NMP_BASE_REDUCTION, [self.nmp_base_reduction.into()],
            LMP_MAX_DEPTH, [self.lmp_depth.into()],
            TT_REDUCTION_DEPTH, [self.tt_reduction_depth.into()],
            FUTILITY_DEPTH, [self.futility_depth.into()],
            SINGULARITY_DEPTH, [self.singularity_depth.into()],
            SEE_DEPTH, [self.see_depth.into()],
            LMR_BASE, [self.lmr_base],
            LMR_DIVISION, [self.lmr_division]
        ]
    }
}