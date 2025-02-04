use std::fmt::Display;

use crate::{
    evaluation::{
        MATERIAL_SCALE_BASE, SEE_BISHOP_VALUE, SEE_KNIGHT_VALUE, SEE_PAWN_VALUE, SEE_QUEEN_VALUE, SEE_ROOK_VALUE
    },
    timemgmt::{
        DEFAULT_MOVES_TO_GO, FAIL_LOW_TM_BONUS, HARD_WINDOW_FRAC, INCREMENT_FRAC,
        NODE_TM_SUBTREE_MULTIPLIER, OPTIMAL_WINDOW_FRAC, STRONG_FORCED_TM_FRAC,
        WEAK_FORCED_TM_FRAC,
    },
};

use super::{
    ASPIRATION_WINDOW, DOUBLE_EXTENSION_MARGIN, DO_DEEPER_BASE_MARGIN, DO_DEEPER_DEPTH_MARGIN,
    FUTILITY_COEFF_0, FUTILITY_COEFF_1, HISTORY_BONUS_MAX, HISTORY_BONUS_MUL, HISTORY_BONUS_OFFSET,
    HISTORY_LMR_DIVISOR, HISTORY_MALUS_MAX, HISTORY_MALUS_MUL, HISTORY_MALUS_OFFSET,
    HISTORY_PRUNING_MARGIN, LMR_BASE, LMR_CUT_NODE_MUL, LMR_DIVISION, LMR_NON_IMPROVING_MUL,
    LMR_NON_PV_MUL, LMR_REFUTATION_MUL, LMR_TTPV_MUL, LMR_TT_CAPTURE_MUL, MAIN_SEE_BOUND,
    MAJOR_CORRHIST_WEIGHT, MINOR_CORRHIST_WEIGHT, NMP_IMPROVING_MARGIN, NMP_REDUCTION_EVAL_DIVISOR,
    NONPAWN_CORRHIST_WEIGHT, PAWN_CORRHIST_WEIGHT, PROBCUT_IMPROVING_MARGIN, PROBCUT_MARGIN,
    QS_FUTILITY, QS_SEE_BOUND, RAZORING_COEFF_0, RAZORING_COEFF_1, RFP_IMPROVING_MARGIN,
    RFP_MARGIN, SEE_QUIET_MARGIN, SEE_STAT_SCORE_MUL, SEE_TACTICAL_MARGIN,
};

#[derive(Clone, Debug)]
pub struct Config {
    pub aspiration_window: i32,
    pub rfp_margin: i32,
    pub rfp_improving_margin: i32,
    pub nmp_improving_margin: i32,
    pub nmp_reduction_eval_divisor: i32,
    pub see_quiet_margin: i32,
    pub see_tactical_margin: i32,
    pub futility_coeff_0: i32,
    pub futility_coeff_1: i32,
    pub razoring_coeff_0: i32,
    pub razoring_coeff_1: i32,
    pub dext_margin: i32,
    pub lmr_base: f64,
    pub lmr_division: f64,
    pub probcut_margin: i32,
    pub probcut_improving_margin: i32,
    pub strong_forced_tm_frac: u32,
    pub weak_forced_tm_frac: u32,
    pub default_moves_to_go: u32,
    pub hard_window_frac: u32,
    pub optimal_window_frac: u32,
    pub increment_frac: u32,
    pub node_tm_subtree_multiplier: u32,
    pub fail_low_tm_bonus: u32,
    pub history_lmr_divisor: i32,
    pub qs_see_bound: i32,
    pub main_see_bound: i32,
    pub do_deeper_base_margin: i32,
    pub do_deeper_depth_margin: i32,
    pub history_pruning_margin: i32,
    pub qs_futility: i32,
    pub see_stat_score_mul: i32,
    pub lmr_refutation_mul: i32,
    pub lmr_non_pv_mul: i32,
    pub lmr_ttpv_mul: i32,
    pub lmr_cut_node_mul: i32,
    pub lmr_non_improving_mul: i32,
    pub lmr_tt_capture_mul: i32,
    pub history_bonus_mul: i32,
    pub history_bonus_offset: i32,
    pub history_bonus_max: i32,
    pub history_malus_mul: i32,
    pub history_malus_offset: i32,
    pub history_malus_max: i32,
    pub pawn_corrhist_weight: i32,
    pub major_corrhist_weight: i32,
    pub minor_corrhist_weight: i32,
    pub nonpawn_corrhist_weight: i32,
    pub see_pawn_value: i32,
    pub see_knight_value: i32,
    pub see_bishop_value: i32,
    pub see_rook_value: i32,
    pub see_queen_value: i32,
    pub material_scale_base: i32,
}

impl Config {
    pub const fn default() -> Self {
        Self {
            aspiration_window: ASPIRATION_WINDOW,
            rfp_margin: RFP_MARGIN,
            rfp_improving_margin: RFP_IMPROVING_MARGIN,
            nmp_improving_margin: NMP_IMPROVING_MARGIN,
            nmp_reduction_eval_divisor: NMP_REDUCTION_EVAL_DIVISOR,
            see_quiet_margin: SEE_QUIET_MARGIN,
            see_tactical_margin: SEE_TACTICAL_MARGIN,
            futility_coeff_0: FUTILITY_COEFF_0,
            futility_coeff_1: FUTILITY_COEFF_1,
            razoring_coeff_0: RAZORING_COEFF_0,
            razoring_coeff_1: RAZORING_COEFF_1,
            dext_margin: DOUBLE_EXTENSION_MARGIN,
            lmr_base: LMR_BASE,
            lmr_division: LMR_DIVISION,
            probcut_margin: PROBCUT_MARGIN,
            probcut_improving_margin: PROBCUT_IMPROVING_MARGIN,
            strong_forced_tm_frac: STRONG_FORCED_TM_FRAC,
            weak_forced_tm_frac: WEAK_FORCED_TM_FRAC,
            default_moves_to_go: DEFAULT_MOVES_TO_GO,
            hard_window_frac: HARD_WINDOW_FRAC,
            optimal_window_frac: OPTIMAL_WINDOW_FRAC,
            increment_frac: INCREMENT_FRAC,
            node_tm_subtree_multiplier: NODE_TM_SUBTREE_MULTIPLIER,
            fail_low_tm_bonus: FAIL_LOW_TM_BONUS,
            history_lmr_divisor: HISTORY_LMR_DIVISOR,
            qs_see_bound: QS_SEE_BOUND,
            main_see_bound: MAIN_SEE_BOUND,
            do_deeper_base_margin: DO_DEEPER_BASE_MARGIN,
            do_deeper_depth_margin: DO_DEEPER_DEPTH_MARGIN,
            history_pruning_margin: HISTORY_PRUNING_MARGIN,
            qs_futility: QS_FUTILITY,
            see_stat_score_mul: SEE_STAT_SCORE_MUL,
            lmr_refutation_mul: LMR_REFUTATION_MUL,
            lmr_non_pv_mul: LMR_NON_PV_MUL,
            lmr_ttpv_mul: LMR_TTPV_MUL,
            lmr_cut_node_mul: LMR_CUT_NODE_MUL,
            lmr_non_improving_mul: LMR_NON_IMPROVING_MUL,
            lmr_tt_capture_mul: LMR_TT_CAPTURE_MUL,
            history_bonus_mul: HISTORY_BONUS_MUL,
            history_bonus_offset: HISTORY_BONUS_OFFSET,
            history_bonus_max: HISTORY_BONUS_MAX,
            history_malus_mul: HISTORY_MALUS_MUL,
            history_malus_offset: HISTORY_MALUS_OFFSET,
            history_malus_max: HISTORY_MALUS_MAX,
            pawn_corrhist_weight: PAWN_CORRHIST_WEIGHT,
            major_corrhist_weight: MAJOR_CORRHIST_WEIGHT,
            minor_corrhist_weight: MINOR_CORRHIST_WEIGHT,
            nonpawn_corrhist_weight: NONPAWN_CORRHIST_WEIGHT,
            see_pawn_value: SEE_PAWN_VALUE,
            see_knight_value: SEE_KNIGHT_VALUE,
            see_bishop_value: SEE_BISHOP_VALUE,
            see_rook_value: SEE_ROOK_VALUE,
            see_queen_value: SEE_QUEEN_VALUE,
            material_scale_base: MATERIAL_SCALE_BASE,
        }
    }
}

impl Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Search parameters:")?;
        for (id, value) in self.ids_with_values() {
            writeln!(f, "    {id}: {value}")?;
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
    ($($option:ident = [$field:expr, $min:expr, $max:expr, $step:expr]),*) => {
        vec![$(
            (stringify!($option), f64::from($field), f64::from($min), f64::from($max), f64::from($step)),)
            *
        ]
    }
}

type LazyFieldParser<'a> = Box<dyn FnMut(&str) -> Result<(), Box<dyn std::error::Error>> + 'a>;

impl Config {
    pub fn ids_with_parsers(&mut self) -> Vec<(&str, LazyFieldParser)> {
        id_parser_gen![
            ASPIRATION_WINDOW = [self.aspiration_window],
            RFP_MARGIN = [self.rfp_margin],
            RFP_IMPROVING_MARGIN = [self.rfp_improving_margin],
            NMP_IMPROVING_MARGIN = [self.nmp_improving_margin],
            NMP_REDUCTION_EVAL_DIVISOR = [self.nmp_reduction_eval_divisor],
            SEE_QUIET_MARGIN = [self.see_quiet_margin],
            SEE_TACTICAL_MARGIN = [self.see_tactical_margin],
            FUTILITY_COEFF_0 = [self.futility_coeff_0],
            FUTILITY_COEFF_1 = [self.futility_coeff_1],
            RAZORING_COEFF_0 = [self.razoring_coeff_0],
            RAZORING_COEFF_1 = [self.razoring_coeff_1],
            DOUBLE_EXTENSION_MARGIN = [self.dext_margin],
            LMR_BASE = [self.lmr_base],
            LMR_DIVISION = [self.lmr_division],
            PROBCUT_MARGIN = [self.probcut_margin],
            PROBCUT_IMPROVING_MARGIN = [self.probcut_improving_margin],
            STRONG_FORCED_TM_FRAC = [self.strong_forced_tm_frac],
            WEAK_FORCED_TM_FRAC = [self.weak_forced_tm_frac],
            DEFAULT_MOVES_TO_GO = [self.default_moves_to_go],
            HARD_WINDOW_FRAC = [self.hard_window_frac],
            OPTIMAL_WINDOW_FRAC = [self.optimal_window_frac],
            INCREMENT_FRAC = [self.increment_frac],
            NODE_TM_SUBTREE_MULTIPLIER = [self.node_tm_subtree_multiplier],
            FAIL_LOW_TM_BONUS = [self.fail_low_tm_bonus],
            HISTORY_LMR_DIVISOR = [self.history_lmr_divisor],
            QS_SEE_BOUND = [self.qs_see_bound],
            MAIN_SEE_BOUND = [self.main_see_bound],
            DO_DEEPER_BASE_MARGIN = [self.do_deeper_base_margin],
            DO_DEEPER_DEPTH_MARGIN = [self.do_deeper_depth_margin],
            HISTORY_PRUNING_MARGIN = [self.history_pruning_margin],
            QS_FUTILITY = [self.qs_futility],
            SEE_STAT_SCORE_MUL = [self.see_stat_score_mul],
            LMR_REFUTATION_MUL = [self.lmr_refutation_mul],
            LMR_NON_PV_MUL = [self.lmr_non_pv_mul],
            LMR_TTPV_MUL = [self.lmr_ttpv_mul],
            LMR_CUT_NODE_MUL = [self.lmr_cut_node_mul],
            LMR_NON_IMPROVING_MUL = [self.lmr_non_improving_mul],
            LMR_TT_CAPTURE_MUL = [self.lmr_tt_capture_mul],
            HISTORY_BONUS_MUL = [self.history_bonus_mul],
            HISTORY_BONUS_OFFSET = [self.history_bonus_offset],
            HISTORY_BONUS_MAX = [self.history_bonus_max],
            HISTORY_MALUS_MUL = [self.history_malus_mul],
            HISTORY_MALUS_OFFSET = [self.history_malus_offset],
            HISTORY_MALUS_MAX = [self.history_malus_max],
            PAWN_CORRHIST_WEIGHT = [self.pawn_corrhist_weight],
            MAJOR_CORRHIST_WEIGHT = [self.major_corrhist_weight],
            MINOR_CORRHIST_WEIGHT = [self.minor_corrhist_weight],
            NONPAWN_CORRHIST_WEIGHT = [self.nonpawn_corrhist_weight],
            SEE_PAWN_VALUE = [self.see_pawn_value],
            SEE_KNIGHT_VALUE = [self.see_knight_value],
            SEE_BISHOP_VALUE = [self.see_bishop_value],
            SEE_ROOK_VALUE = [self.see_rook_value],
            SEE_QUEEN_VALUE = [self.see_queen_value],
            MATERIAL_SCALE_BASE = [self.material_scale_base]
        ]
    }

    #[rustfmt::skip]
    pub fn ids_with_values(&self) -> Vec<(&str, f64)> {
        self.base_config()
            .into_iter()
            .map(|(id, value, _, _, _)| (id, value))
            .collect()
    }

    pub fn base_config(&self) -> Vec<(&str, f64, f64, f64, f64)> {
        #![allow(clippy::cast_precision_loss)]
        id_value_gen![
            ASPIRATION_WINDOW = [self.aspiration_window, 1, 50, 3],
            RFP_MARGIN = [self.rfp_margin, 40, 200, 10],
            RFP_IMPROVING_MARGIN = [self.rfp_improving_margin, 30, 150, 10],
            NMP_IMPROVING_MARGIN = [self.nmp_improving_margin, 30, 200, 10],
            NMP_REDUCTION_EVAL_DIVISOR = [self.nmp_reduction_eval_divisor, 100, 400, 20],
            SEE_QUIET_MARGIN = [self.see_quiet_margin, -150, -20, 5],
            SEE_TACTICAL_MARGIN = [self.see_tactical_margin, -100, -1, 3],
            FUTILITY_COEFF_0 = [self.futility_coeff_0, 40, 200, 10],
            FUTILITY_COEFF_1 = [self.futility_coeff_1, 40, 200, 10],
            RAZORING_COEFF_0 = [self.razoring_coeff_0, 200, 700, 30],
            RAZORING_COEFF_1 = [self.razoring_coeff_1, 150, 600, 30],
            DOUBLE_EXTENSION_MARGIN = [self.dext_margin, 5, 100, 1],
            LMR_BASE = [self.lmr_base, 40, 150, 7],
            LMR_DIVISION = [self.lmr_division, 150, 500, 15],
            PROBCUT_MARGIN = [self.probcut_margin, 100, 400, 20],
            PROBCUT_IMPROVING_MARGIN = [self.probcut_improving_margin, 20, 150, 10],
            STRONG_FORCED_TM_FRAC = [self.strong_forced_tm_frac, 1, 1000, 30],
            WEAK_FORCED_TM_FRAC = [self.weak_forced_tm_frac, 1, 1000, 30],
            DEFAULT_MOVES_TO_GO = [self.default_moves_to_go, 1, 100, 3],
            HARD_WINDOW_FRAC = [self.hard_window_frac, 1, 100, 5],
            OPTIMAL_WINDOW_FRAC = [self.optimal_window_frac, 1, 100, 5],
            INCREMENT_FRAC = [self.increment_frac, 1, 100, 10],
            NODE_TM_SUBTREE_MULTIPLIER = [self.node_tm_subtree_multiplier, 1, 1000, 15],
            FAIL_LOW_TM_BONUS = [self.fail_low_tm_bonus, 1, 1000, 30],
            HISTORY_LMR_DIVISOR = [self.history_lmr_divisor, 1, 16383, 100],
            QS_SEE_BOUND = [self.qs_see_bound, -500, 500, 50],
            MAIN_SEE_BOUND = [self.main_see_bound, -500, 500, 50],
            DO_DEEPER_BASE_MARGIN = [self.do_deeper_base_margin, 1, 200, 20],
            DO_DEEPER_DEPTH_MARGIN = [self.do_deeper_depth_margin, 1, 50, 2],
            HISTORY_PRUNING_MARGIN = [self.history_pruning_margin, -5000, 1000, 500],
            QS_FUTILITY = [self.qs_futility, -500, 500, 25],
            SEE_STAT_SCORE_MUL = [self.see_stat_score_mul, 1, 100, 5],
            LMR_REFUTATION_MUL = [self.lmr_refutation_mul, 1, 4096, 96],
            LMR_NON_PV_MUL = [self.lmr_non_pv_mul, 1, 4096, 96],
            LMR_TTPV_MUL = [self.lmr_ttpv_mul, 1, 4096, 96],
            LMR_CUT_NODE_MUL = [self.lmr_cut_node_mul, 1, 4096, 96],
            LMR_NON_IMPROVING_MUL = [self.lmr_non_improving_mul, 1, 4096, 96],
            LMR_TT_CAPTURE_MUL = [self.lmr_tt_capture_mul, 1, 4096, 96],
            HISTORY_BONUS_MUL = [self.history_bonus_mul, 1, 1500, 32],
            HISTORY_BONUS_OFFSET = [self.history_bonus_offset, -1024, 1024, 64],
            HISTORY_BONUS_MAX = [self.history_bonus_max, 1, 4096, 256],
            HISTORY_MALUS_MUL = [self.history_malus_mul, 1, 1500, 32],
            HISTORY_MALUS_OFFSET = [self.history_malus_offset, -1024, 1024, 64],
            HISTORY_MALUS_MAX = [self.history_malus_max, 1, 4096, 256],
            PAWN_CORRHIST_WEIGHT = [self.pawn_corrhist_weight, 1, 4096, 144],
            MAJOR_CORRHIST_WEIGHT = [self.major_corrhist_weight, 1, 4096, 144],
            MINOR_CORRHIST_WEIGHT = [self.minor_corrhist_weight, 1, 4096, 144],
            NONPAWN_CORRHIST_WEIGHT = [self.nonpawn_corrhist_weight, 1, 4096, 144],
            SEE_PAWN_VALUE = [self.see_pawn_value, 1, 4096, 16],
            SEE_KNIGHT_VALUE = [self.see_knight_value, 1, 4096, 16],
            SEE_BISHOP_VALUE = [self.see_bishop_value, 1, 4096, 16],
            SEE_ROOK_VALUE = [self.see_rook_value, 1, 4096, 16],
            SEE_QUEEN_VALUE = [self.see_queen_value, 1, 4096, 16],
            MATERIAL_SCALE_BASE = [self.material_scale_base, 1, 4096, 32]
        ]
    }

    pub fn emit_json_for_spsa(&self) -> String {
        let mut json = String::new();
        json.push_str("{\n");
        let mut tunegroups = Vec::new();
        for (id, value, min, max, step) in self.base_config() {
            // formatted like
            // "PROBCUT_MARGIN": {
            //     "value": 200,
            //     "min_value": 100,
            //     "max_value": 400,
            //     "step": 20
            //   },
            tunegroups.push(format!("  \"{id}\": {{\n    \"value\": {value},\n    \"min_value\": {min},\n    \"max_value\": {max},\n    \"step\": {step}\n  }}"));
        }
        // stupid json comma handling
        json.push_str(&tunegroups.join(",\n"));
        json.push_str("\n}\n");
        json
    }

    pub fn emit_csv_for_spsa(&self) -> String {
        let mut csv = String::new();
        let mut tunegroups = Vec::new();
        for (id, value, min, max, step) in self.base_config() {
            tunegroups.push(format!(
                "{id}, int, {value:.1}, {min:.1}, {max:.1}, {step:.1}, 0.002"
            ));
        }
        csv.push_str(&tunegroups.join("\n"));
        csv
    }
}

mod tests {
    #[test]
    fn macro_hackery_same_length() {
        let mut sp = super::Config::default();
        let l1 = sp.ids_with_parsers().len();
        let l2 = sp.ids_with_values().len();
        assert_eq!(l1, l2);
    }

    #[test]
    fn parser_actually_works() {
        let mut sp = super::Config::default();
        let rfp_margin = sp
            .ids_with_values()
            .iter()
            .find(|(id, _)| *id == "RFP_MARGIN")
            .unwrap()
            .1;
        assert!((rfp_margin - f64::from(crate::search::RFP_MARGIN)).abs() < f64::EPSILON);
        // set using the parser:
        sp.ids_with_parsers()
            .iter_mut()
            .find(|(id, _)| *id == "RFP_MARGIN")
            .unwrap()
            .1("10")
        .unwrap();
        // re-extract:
        let rfp_margin = sp
            .ids_with_values()
            .iter()
            .find(|(id, _)| *id == "RFP_MARGIN")
            .unwrap()
            .1;
        assert!((rfp_margin - 10.0).abs() < f64::EPSILON);
    }
}
