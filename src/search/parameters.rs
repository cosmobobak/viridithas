use std::fmt::Display;

use crate::definitions::depth::Depth;

use super::{
    ASPIRATION_WINDOW, RFP_DEPTH, RFP_IMPROVING_MARGIN, RFP_MARGIN,
    FUTILITY_COEFF_0, FUTILITY_COEFF_1, FUTILITY_COEFF_2, FUTILITY_DEPTH, LMP_BASE_MOVES,
    LMP_MAX_DEPTH, LMR_BASE, LMR_DIVISION, NULLMOVE_PRUNING_IMPROVING_MARGIN,
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
            nmp_improving_margin: NULLMOVE_PRUNING_IMPROVING_MARGIN,
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
        write!(f, "Search parameters:")?;
        write!(f, "    Aspiration window: {}", self.aspiration_window)?;
        write!(f, "    RFP margin: {}", self.rfp_margin)?;
        write!(f, "    RFP improving margin: {}", self.rfp_improving_margin)?;
        write!(f, "    NMP improving margin: {}", self.nmp_improving_margin)?;
        write!(f, "    SEE quiet margin: {}", self.see_quiet_margin)?;
        write!(f, "    SEE tactical margin: {}", self.see_tactical_margin)?;
        write!(f, "    LMP base moves: {}", self.lmp_base_moves)?;
        write!(f, "    Futility coeff 2: {}", self.futility_coeff_2)?;
        write!(f, "    Futility coeff 1: {}", self.futility_coeff_1)?;
        write!(f, "    Futility coeff 0: {}", self.futility_coeff_0)?;
        write!(f, "    RFP depth: {}", self.rfp_depth)?;
        write!(f, "    NMP base reduction: {}", self.nmp_base_reduction)?;
        write!(f, "    LMP depth: {}", self.lmp_depth)?;
        write!(f, "    TT reduction depth: {}", self.tt_reduction_depth)?;
        write!(f, "    Futility depth: {}", self.futility_depth)?;
        write!(f, "    Singularity depth: {}", self.singularity_depth)?;
        write!(f, "    SEE depth: {}", self.see_depth)?;
        write!(f, "    LMR base: {}", self.lmr_base)?;
        write!(f, "    LMR division: {}", self.lmr_division)?;
        Ok(())
    }
}
