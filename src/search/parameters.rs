use std::fmt::Display;

use crate::{
    timemgmt::{
        DEFAULT_MOVES_TO_GO, FAIL_LOW_TM_BONUS, HARD_WINDOW_FRAC, INCREMENT_FRAC, NODE_TM_SUBTREE_MULTIPLIER,
        OPTIMAL_WINDOW_FRAC, STRONG_FORCED_TM_FRAC, WEAK_FORCED_TM_FRAC,
    },
    util::depth::Depth,
};

use super::{
    ASPIRATION_WINDOW, DOUBLE_EXTENSION_MARGIN, DO_DEEPER_BASE_MARGIN, DO_DEEPER_DEPTH_MARGIN, FUTILITY_COEFF_0,
    FUTILITY_COEFF_1, FUTILITY_DEPTH, HISTORY_LMR_BOUND, HISTORY_LMR_DIVISOR, HISTORY_PRUNING_DEPTH,
    HISTORY_PRUNING_MARGIN, LMP_BASE_MOVES, LMP_DEPTH, LMR_BASE, LMR_BASE_MOVES, LMR_DIVISION, MAIN_SEE_BOUND,
    MAX_NMP_EVAL_REDUCTION, NMP_BASE_REDUCTION, NMP_IMPROVING_MARGIN, NMP_REDUCTION_DEPTH_DIVISOR,
    NMP_REDUCTION_EVAL_DIVISOR, NMP_VERIFICATION_DEPTH, PROBCUT_IMPROVING_MARGIN, PROBCUT_MARGIN, PROBCUT_MIN_DEPTH,
    PROBCUT_REDUCTION, QS_SEE_BOUND, RAZORING_COEFF_0, RAZORING_COEFF_1, RFP_DEPTH, RFP_IMPROVING_MARGIN, RFP_MARGIN,
    SEE_DEPTH, SEE_QUIET_MARGIN, SEE_TACTICAL_MARGIN, SINGULARITY_DEPTH, TT_EXTENSION_DEPTH, TT_REDUCTION_DEPTH,
};

#[derive(Clone, Debug)]
pub struct Config {
    pub aspiration_window: i32,
    pub rfp_margin: i32,
    pub rfp_improving_margin: i32,
    pub nmp_improving_margin: i32,
    pub nmp_reduction_depth_divisor: i32,
    pub nmp_reduction_eval_divisor: i32,
    pub max_nmp_eval_reduction: i32,
    pub see_quiet_margin: i32,
    pub see_tactical_margin: i32,
    pub lmp_base_moves: i32,
    pub futility_coeff_0: i32,
    pub futility_coeff_1: i32,
    pub razoring_coeff_0: i32,
    pub razoring_coeff_1: i32,
    pub rfp_depth: Depth,
    pub nmp_base_reduction: Depth,
    pub nmp_verification_depth: Depth,
    pub lmp_depth: Depth,
    pub tt_reduction_depth: Depth,
    pub tt_extension_depth: Depth,
    pub futility_depth: Depth,
    pub singularity_depth: Depth,
    pub dext_margin: i32,
    pub see_depth: Depth,
    pub lmr_base: f64,
    pub lmr_division: f64,
    pub probcut_margin: i32,
    pub probcut_improving_margin: i32,
    pub probcut_reduction: Depth,
    pub probcut_min_depth: Depth,
    pub strong_forced_tm_frac: u32,
    pub weak_forced_tm_frac: u32,
    pub default_moves_to_go: u32,
    pub hard_window_frac: u32,
    pub optimal_window_frac: u32,
    pub increment_frac: u32,
    pub node_tm_subtree_multiplier: u32,
    pub fail_low_tm_bonus: u32,
    pub lmr_base_moves: u32,
    pub history_lmr_divisor: i32,
    pub history_lmr_bound: i32,
    pub qs_see_bound: i32,
    pub main_see_bound: i32,
    pub do_deeper_base_margin: i32,
    pub do_deeper_depth_margin: i32,
    pub history_pruning_depth: Depth,
    pub history_pruning_margin: i32,
}

impl Config {
    pub const fn default() -> Self {
        Self {
            aspiration_window: ASPIRATION_WINDOW,
            rfp_margin: RFP_MARGIN,
            rfp_improving_margin: RFP_IMPROVING_MARGIN,
            nmp_improving_margin: NMP_IMPROVING_MARGIN,
            nmp_reduction_depth_divisor: NMP_REDUCTION_DEPTH_DIVISOR,
            nmp_reduction_eval_divisor: NMP_REDUCTION_EVAL_DIVISOR,
            max_nmp_eval_reduction: MAX_NMP_EVAL_REDUCTION,
            see_quiet_margin: SEE_QUIET_MARGIN,
            see_tactical_margin: SEE_TACTICAL_MARGIN,
            lmp_base_moves: LMP_BASE_MOVES,
            futility_coeff_0: FUTILITY_COEFF_0,
            futility_coeff_1: FUTILITY_COEFF_1,
            razoring_coeff_0: RAZORING_COEFF_0,
            razoring_coeff_1: RAZORING_COEFF_1,
            rfp_depth: RFP_DEPTH,
            nmp_base_reduction: NMP_BASE_REDUCTION,
            nmp_verification_depth: NMP_VERIFICATION_DEPTH,
            lmp_depth: LMP_DEPTH,
            tt_reduction_depth: TT_REDUCTION_DEPTH,
            tt_extension_depth: TT_EXTENSION_DEPTH,
            futility_depth: FUTILITY_DEPTH,
            singularity_depth: SINGULARITY_DEPTH,
            dext_margin: DOUBLE_EXTENSION_MARGIN,
            see_depth: SEE_DEPTH,
            lmr_base: LMR_BASE,
            lmr_division: LMR_DIVISION,
            probcut_margin: PROBCUT_MARGIN,
            probcut_improving_margin: PROBCUT_IMPROVING_MARGIN,
            probcut_reduction: PROBCUT_REDUCTION,
            probcut_min_depth: PROBCUT_MIN_DEPTH,
            strong_forced_tm_frac: STRONG_FORCED_TM_FRAC,
            weak_forced_tm_frac: WEAK_FORCED_TM_FRAC,
            default_moves_to_go: DEFAULT_MOVES_TO_GO,
            hard_window_frac: HARD_WINDOW_FRAC,
            optimal_window_frac: OPTIMAL_WINDOW_FRAC,
            increment_frac: INCREMENT_FRAC,
            node_tm_subtree_multiplier: NODE_TM_SUBTREE_MULTIPLIER,
            fail_low_tm_bonus: FAIL_LOW_TM_BONUS,
            lmr_base_moves: LMR_BASE_MOVES,
            history_lmr_divisor: HISTORY_LMR_DIVISOR,
            history_lmr_bound: HISTORY_LMR_BOUND,
            qs_see_bound: QS_SEE_BOUND,
            main_see_bound: MAIN_SEE_BOUND,
            do_deeper_base_margin: DO_DEEPER_BASE_MARGIN,
            do_deeper_depth_margin: DO_DEEPER_DEPTH_MARGIN,
            history_pruning_depth: HISTORY_PRUNING_DEPTH,
            history_pruning_margin: HISTORY_PRUNING_MARGIN,
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
            NMP_REDUCTION_DEPTH_DIVISOR = [self.nmp_reduction_depth_divisor],
            NMP_REDUCTION_EVAL_DIVISOR = [self.nmp_reduction_eval_divisor],
            MAX_NMP_EVAL_REDUCTION = [self.max_nmp_eval_reduction],
            SEE_QUIET_MARGIN = [self.see_quiet_margin],
            SEE_TACTICAL_MARGIN = [self.see_tactical_margin],
            LMP_BASE_MOVES = [self.lmp_base_moves],
            FUTILITY_COEFF_0 = [self.futility_coeff_0],
            FUTILITY_COEFF_1 = [self.futility_coeff_1],
            RAZORING_COEFF_0 = [self.razoring_coeff_0],
            RAZORING_COEFF_1 = [self.razoring_coeff_1],
            RFP_DEPTH = [self.rfp_depth],
            NMP_BASE_REDUCTION = [self.nmp_base_reduction],
            NMP_VERIFICATION_DEPTH = [self.nmp_verification_depth],
            LMP_DEPTH = [self.lmp_depth],
            TT_REDUCTION_DEPTH = [self.tt_reduction_depth],
            TT_EXTENSION_DEPTH = [self.tt_extension_depth],
            FUTILITY_DEPTH = [self.futility_depth],
            SINGULARITY_DEPTH = [self.singularity_depth],
            DOUBLE_EXTENSION_MARGIN = [self.dext_margin],
            SEE_DEPTH = [self.see_depth],
            LMR_BASE = [self.lmr_base],
            LMR_DIVISION = [self.lmr_division],
            PROBCUT_MARGIN = [self.probcut_margin],
            PROBCUT_IMPROVING_MARGIN = [self.probcut_improving_margin],
            PROBCUT_REDUCTION = [self.probcut_reduction],
            PROBCUT_MIN_DEPTH = [self.probcut_min_depth],
            STRONG_FORCED_TM_FRAC = [self.strong_forced_tm_frac],
            WEAK_FORCED_TM_FRAC = [self.weak_forced_tm_frac],
            DEFAULT_MOVES_TO_GO = [self.default_moves_to_go],
            HARD_WINDOW_FRAC = [self.hard_window_frac],
            OPTIMAL_WINDOW_FRAC = [self.optimal_window_frac],
            INCREMENT_FRAC = [self.increment_frac],
            NODE_TM_SUBTREE_MULTIPLIER = [self.node_tm_subtree_multiplier],
            FAIL_LOW_TM_BONUS = [self.fail_low_tm_bonus],
            LMR_BASE_MOVES = [self.lmr_base_moves],
            HISTORY_LMR_DIVISOR = [self.history_lmr_divisor],
            HISTORY_LMR_BOUND = [self.history_lmr_bound],
            QS_SEE_BOUND = [self.qs_see_bound],
            MAIN_SEE_BOUND = [self.main_see_bound],
            DO_DEEPER_BASE_MARGIN = [self.do_deeper_base_margin],
            DO_DEEPER_DEPTH_MARGIN = [self.do_deeper_depth_margin],
            HISTORY_PRUNING_DEPTH = [self.history_pruning_depth],
            HISTORY_PRUNING_MARGIN = [self.history_pruning_margin]
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
            NMP_REDUCTION_DEPTH_DIVISOR = [self.nmp_reduction_depth_divisor, 2, 5, 1],
            NMP_REDUCTION_EVAL_DIVISOR = [self.nmp_reduction_eval_divisor, 100, 400, 20],
            MAX_NMP_EVAL_REDUCTION = [self.max_nmp_eval_reduction, 2, 7, 1],
            SEE_QUIET_MARGIN = [self.see_quiet_margin, -150, -20, 5],
            SEE_TACTICAL_MARGIN = [self.see_tactical_margin, -100, -1, 3],
            LMP_BASE_MOVES = [self.lmp_base_moves, 1, 5, 1],
            FUTILITY_COEFF_0 = [self.futility_coeff_0, 40, 200, 10],
            FUTILITY_COEFF_1 = [self.futility_coeff_1, 40, 200, 10],
            RAZORING_COEFF_0 = [self.razoring_coeff_0, 200, 700, 30],
            RAZORING_COEFF_1 = [self.razoring_coeff_1, 150, 600, 30],
            RFP_DEPTH = [self.rfp_depth, 5, 12, 1],
            NMP_BASE_REDUCTION = [self.nmp_base_reduction, 2, 5, 1],
            NMP_VERIFICATION_DEPTH = [self.nmp_verification_depth, 8, 16, 1],
            LMP_DEPTH = [self.lmp_depth, 5, 12, 1],
            TT_REDUCTION_DEPTH = [self.tt_reduction_depth, 2, 8, 1],
            TT_EXTENSION_DEPTH = [self.tt_extension_depth, 2, 16, 1],
            FUTILITY_DEPTH = [self.futility_depth, 2, 10, 1],
            SINGULARITY_DEPTH = [self.singularity_depth, 6, 12, 1],
            DOUBLE_EXTENSION_MARGIN = [self.dext_margin, 5, 100, 1],
            SEE_DEPTH = [self.see_depth, 6, 14, 1],
            LMR_BASE = [self.lmr_base, 40, 150, 7],
            LMR_DIVISION = [self.lmr_division, 150, 500, 15],
            PROBCUT_MARGIN = [self.probcut_margin, 100, 400, 20],
            PROBCUT_IMPROVING_MARGIN = [self.probcut_improving_margin, 20, 150, 10],
            PROBCUT_REDUCTION = [self.probcut_reduction, 2, 8, 1],
            PROBCUT_MIN_DEPTH = [self.probcut_min_depth, 2, 10, 1],
            STRONG_FORCED_TM_FRAC = [self.strong_forced_tm_frac, 1, 1000, 30],
            WEAK_FORCED_TM_FRAC = [self.weak_forced_tm_frac, 1, 1000, 30],
            DEFAULT_MOVES_TO_GO = [self.default_moves_to_go, 1, 100, 3],
            HARD_WINDOW_FRAC = [self.hard_window_frac, 1, 100, 5],
            OPTIMAL_WINDOW_FRAC = [self.optimal_window_frac, 1, 100, 5],
            INCREMENT_FRAC = [self.increment_frac, 1, 100, 10],
            NODE_TM_SUBTREE_MULTIPLIER = [self.node_tm_subtree_multiplier, 1, 1000, 15],
            FAIL_LOW_TM_BONUS = [self.fail_low_tm_bonus, 1, 1000, 30],
            LMR_BASE_MOVES = [self.lmr_base_moves, 1, 5, 1],
            HISTORY_LMR_DIVISOR = [self.history_lmr_divisor, 1, 16383, 100],
            HISTORY_LMR_BOUND = [self.history_lmr_bound, 1, 8, 1],
            QS_SEE_BOUND = [self.qs_see_bound, -500, 500, 50],
            MAIN_SEE_BOUND = [self.main_see_bound, -500, 500, 50],
            DO_DEEPER_BASE_MARGIN = [self.do_deeper_base_margin, 1, 200, 20],
            DO_DEEPER_DEPTH_MARGIN = [self.do_deeper_depth_margin, 1, 50, 2],
            HISTORY_PRUNING_DEPTH = [self.history_pruning_depth, 2, 14, 1],
            HISTORY_PRUNING_MARGIN = [self.history_pruning_margin, -5000, 1000, 500]
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
            tunegroups.push(format!("{id}, int, {value:.1}, {min:.1}, {max:.1}, {step:.1}, 0.002"));
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
        use crate::search::PROBCUT_MIN_DEPTH;

        let mut sp = super::Config::default();
        let probcut_min_depth = sp.ids_with_values().iter().find(|(id, _)| *id == "PROBCUT_MIN_DEPTH").unwrap().1;
        assert!((probcut_min_depth - f64::from(PROBCUT_MIN_DEPTH)).abs() < f64::EPSILON);
        // set using the parser:
        sp.ids_with_parsers().iter_mut().find(|(id, _)| *id == "PROBCUT_MIN_DEPTH").unwrap().1("10").unwrap();
        // re-extract:
        let probcut_min_depth = sp.ids_with_values().iter().find(|(id, _)| *id == "PROBCUT_MIN_DEPTH").unwrap().1;
        assert!((probcut_min_depth - 10.0).abs() < f64::EPSILON);
    }
}
