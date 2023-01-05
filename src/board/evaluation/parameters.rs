use std::{error::Error, fmt::Display, path::Path};

use crate::{definitions::Square, piece::{Piece, PieceType}};

use super::{
    score::S, BISHOP_MOBILITY_BONUS, BISHOP_PAIR_BONUS, DOUBLED_PAWN_MALUS, ISOLATED_PAWN_MALUS,
    KING_DANGER_COEFFS, KING_DANGER_PIECE_WEIGHTS, KNIGHT_MOBILITY_BONUS, MINOR_THREAT_ON_MAJOR,
    PASSED_PAWN_BONUS, PAWN_THREAT_ON_MAJOR, PAWN_THREAT_ON_MINOR, PIECE_VALUES,
    QUEEN_HALF_OPEN_FILE_BONUS, QUEEN_MOBILITY_BONUS, QUEEN_OPEN_FILE_BONUS,
    ROOK_HALF_OPEN_FILE_BONUS, ROOK_MOBILITY_BONUS, ROOK_OPEN_FILE_BONUS, TEMPO_BONUS,
};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EvalParams {
    pub piece_values: [S; 13],
    pub isolated_pawn_malus: S,
    pub doubled_pawn_malus: S,
    pub bishop_pair_bonus: S,
    pub rook_open_file_bonus: S,
    pub rook_half_open_file_bonus: S,
    pub queen_open_file_bonus: S,
    pub queen_half_open_file_bonus: S,
    pub knight_mobility_bonus: [S; 9],
    pub bishop_mobility_bonus: [S; 14],
    pub rook_mobility_bonus: [S; 15],
    pub queen_mobility_bonus: [S; 28],
    pub passed_pawn_bonus: [S; 6],
    pub piece_square_tables: [[S; 64]; 13],
    pub tempo: S,
    pub pawn_threat_on_minor: S,
    pub pawn_threat_on_major: S,
    pub minor_threat_on_major: S,
    pub king_danger_coeffs: [i32; 3],
    pub king_danger_piece_weights: [i32; 8],
}

impl EvalParams {
    pub const fn default() -> Self {
        Self {
            piece_values: PIECE_VALUES,
            isolated_pawn_malus: ISOLATED_PAWN_MALUS,
            doubled_pawn_malus: DOUBLED_PAWN_MALUS,
            bishop_pair_bonus: BISHOP_PAIR_BONUS,
            rook_open_file_bonus: ROOK_OPEN_FILE_BONUS,
            rook_half_open_file_bonus: ROOK_HALF_OPEN_FILE_BONUS,
            queen_open_file_bonus: QUEEN_OPEN_FILE_BONUS,
            queen_half_open_file_bonus: QUEEN_HALF_OPEN_FILE_BONUS,
            knight_mobility_bonus: KNIGHT_MOBILITY_BONUS,
            bishop_mobility_bonus: BISHOP_MOBILITY_BONUS,
            rook_mobility_bonus: ROOK_MOBILITY_BONUS,
            queen_mobility_bonus: QUEEN_MOBILITY_BONUS,
            passed_pawn_bonus: PASSED_PAWN_BONUS,
            piece_square_tables: crate::piecesquaretable::tables::construct_piece_square_table(),
            tempo: TEMPO_BONUS,
            pawn_threat_on_minor: PAWN_THREAT_ON_MINOR,
            pawn_threat_on_major: PAWN_THREAT_ON_MAJOR,
            minor_threat_on_major: MINOR_THREAT_ON_MAJOR,
            king_danger_coeffs: KING_DANGER_COEFFS,
            king_danger_piece_weights: KING_DANGER_PIECE_WEIGHTS,
        }
    }
}

impl Display for EvalParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "Parameters {{")?;
        writeln!(f, "    piece_values: {:?},", &self.piece_values[1..6])?;
        writeln!(f, "    isolated_pawn_malus: {:?},", self.isolated_pawn_malus)?;
        writeln!(f, "    doubled_pawn_malus: {:?},", self.doubled_pawn_malus)?;
        writeln!(f, "    bishop_pair_bonus: {:?},", self.bishop_pair_bonus)?;
        writeln!(f, "    rook_open_file_bonus: {:?},", self.rook_open_file_bonus)?;
        writeln!(f, "    rook_half_open_file_bonus: {:?},", self.rook_half_open_file_bonus)?;
        writeln!(f, "    queen_open_file_bonus: {:?},", self.queen_open_file_bonus)?;
        writeln!(f, "    queen_half_open_file_bonus: {:?},", self.queen_half_open_file_bonus)?;
        writeln!(f, "    knight_mobility_bonus: {:?},", self.knight_mobility_bonus)?;
        writeln!(f, "    bishop_mobility_bonus: {:?},", self.bishop_mobility_bonus)?;
        writeln!(f, "    rook_mobility_bonus: {:?},", self.rook_mobility_bonus)?;
        writeln!(f, "    queen_mobility_bonus: {:?},", self.queen_mobility_bonus)?;
        writeln!(f, "    passed_pawn_bonus: {:?},", self.passed_pawn_bonus)?;
        writeln!(f, "    tempo: {:?},", self.tempo)?;
        writeln!(f, "    pawn_threat_on_minor: {:?},", self.pawn_threat_on_minor)?;
        writeln!(f, "    pawn_threat_on_major: {:?},", self.pawn_threat_on_major)?;
        writeln!(f, "    minor_threat_on_major: {:?},", self.minor_threat_on_major)?;
        writeln!(f, "    king_danger_formula: {:?},", self.king_danger_coeffs)?;
        write!(f, "}}")?;
        Ok(())
    }
}

impl EvalParams {
    pub const NULL: Self = Self {
        piece_values: [S::NULL; 13],
        isolated_pawn_malus: S::NULL,
        doubled_pawn_malus: S::NULL,
        bishop_pair_bonus: S::NULL,
        rook_open_file_bonus: S::NULL,
        rook_half_open_file_bonus: S::NULL,
        queen_open_file_bonus: S::NULL,
        queen_half_open_file_bonus: S::NULL,
        knight_mobility_bonus: [S::NULL; 9],
        bishop_mobility_bonus: [S::NULL; 14],
        rook_mobility_bonus: [S::NULL; 15],
        queen_mobility_bonus: [S::NULL; 28],
        passed_pawn_bonus: [S::NULL; 6],
        piece_square_tables: [[S::NULL; 64]; 13],
        tempo: S::NULL,
        pawn_threat_on_minor: S::NULL,
        pawn_threat_on_major: S::NULL,
        minor_threat_on_major: S::NULL,
        king_danger_coeffs: [0; 3],
        king_danger_piece_weights: [0; 8],
    };

    pub fn vectorise(&self) -> Vec<i32> {
        let ss = self.piece_values[1..6] // pawn to queen
            .iter()
            .copied()
            .chain(Some(self.isolated_pawn_malus))
            .chain(Some(self.doubled_pawn_malus))
            .chain(Some(self.bishop_pair_bonus))
            .chain(Some(S(self.rook_open_file_bonus.0, self.rook_half_open_file_bonus.0)))
            .chain(Some(S(self.queen_open_file_bonus.0, self.queen_half_open_file_bonus.0)))
            .chain(self.knight_mobility_bonus.into_iter())
            .chain(self.bishop_mobility_bonus.into_iter())
            .chain(self.rook_mobility_bonus.into_iter())
            .chain(self.queen_mobility_bonus.into_iter())
            .chain(self.passed_pawn_bonus.into_iter())
            // take the left halves of the white piece square tables, except for the pawn table.
            .chain(self.piece_square_tables[Piece::WP.index()].iter().copied())
            .chain(
                self.piece_square_tables[(Piece::WN.index())..=(Piece::WK.index())]
                    .iter()
                    .flat_map(|x| x.chunks(4).step_by(2).flatten().copied()),
            )
            .chain(Some(self.tempo))
            .chain(Some(self.pawn_threat_on_minor))
            .chain(Some(self.pawn_threat_on_major))
            .chain(Some(self.minor_threat_on_major));
        ss.flat_map(|s| [s.0, s.1].into_iter())
            .chain(self.king_danger_coeffs.iter().copied())
            .chain(self.king_danger_piece_weights.iter().copied())
            .collect()
    }

    #[allow(clippy::too_many_lines)]
    pub fn devectorise(data: &[i32]) -> Self {
        let mut out = Self::NULL;
        let mut s_iter = data[..data.len() - 3 - 8].chunks(2).map(|x| S(x[0], x[1]));
        for p in 1..6 {
            let val = s_iter.next().expect("failed to read piece_value term from vector");
            out.piece_values[p] = val;
            out.piece_values[p + 6] = val;
        }
        out.isolated_pawn_malus =
            s_iter.next().expect("failed to read isolated_pawn_malus term from vector");
        out.doubled_pawn_malus =
            s_iter.next().expect("failed to read doubled_pawn_malus term from vector");
        out.bishop_pair_bonus =
            s_iter.next().expect("failed to read bishop_pair_bonus term from vector");
        let rook_file_bonus =
            s_iter.next().expect("failed to read rook_file_bonus term from vector");
        out.rook_open_file_bonus = S(rook_file_bonus.0, 0);
        out.rook_half_open_file_bonus = S(rook_file_bonus.1, 0);
        let queen_file_bonus =
            s_iter.next().expect("failed to read queen_file_bonus term from vector");
        out.queen_open_file_bonus = S(queen_file_bonus.0, 0);
        out.queen_half_open_file_bonus = S(queen_file_bonus.1, 0);
        for knight_mobility_bonus in &mut out.knight_mobility_bonus {
            *knight_mobility_bonus =
                s_iter.next().expect("failed to read knight_mobility_bonus term from vector");
        }
        for bishop_mobility_bonus in &mut out.bishop_mobility_bonus {
            *bishop_mobility_bonus =
                s_iter.next().expect("failed to read bishop_mobility_bonus term from vector");
        }
        for rook_mobility_bonus in &mut out.rook_mobility_bonus {
            *rook_mobility_bonus =
                s_iter.next().expect("failed to read rook_mobility_bonus term from vector");
        }
        for queen_mobility_bonus in &mut out.queen_mobility_bonus {
            *queen_mobility_bonus =
                s_iter.next().expect("failed to read queen_mobility_bonus term from vector");
        }
        for passed_pawn_bonus in &mut out.passed_pawn_bonus {
            *passed_pawn_bonus =
                s_iter.next().expect("failed to read passed_pawn_bonus term from vector");
        }
        // load in the pawn table
        for sq in 0..64 {
            let sq = Square::new(sq);
            let val =
                s_iter.next().expect("failed to read pawn piece_square_table term from vector");
            out.piece_square_tables[Piece::WP.index()][sq.index()] = val;
            out.piece_square_tables[Piece::BP.index()][sq.flip_rank().index()] = -val;
        }
        // load in the rest of the tables
        for pt in PieceType::all().skip(1) {
            for sq in 0..64 {
                let sq = Square::new(sq);
                let file = sq.file();
                if file > 3 {
                    // load from the other half of the piece-square table.
                    // the left-hand sides of the tables are loaded first, so we
                    // can safely load out of LHS to populate RHS.
                    let mirrored_sq = sq.flip_file();
                    out.piece_square_tables[pt.index()][sq.index()] =
                        out.piece_square_tables[pt.index()][mirrored_sq.index()];
                    out.piece_square_tables[pt.index() + 6][sq.flip_rank().index()] =
                        out.piece_square_tables[pt.index() + 6][mirrored_sq.flip_rank().index()];
                } else {
                    let val =
                        s_iter.next().expect("failed to read piece_square_table term from vector");
                    out.piece_square_tables[pt.index()][sq.index()] = val;
                    out.piece_square_tables[pt.index() + 6][sq.flip_rank().index()] = -val;
                }
            }
        }
        // read in the tempo term
        out.tempo = s_iter.next().expect("failed to read tempo term from vector");
        // read in the pawn threat on minor term
        out.pawn_threat_on_minor =
            s_iter.next().expect("failed to read pawn_threat_on_minor term from vector");
        // read in the pawn threat on major term
        out.pawn_threat_on_major =
            s_iter.next().expect("failed to read pawn_threat_on_major term from vector");
        // read in the minor threat on major term
        out.minor_threat_on_major =
            s_iter.next().expect("failed to read minor_threat_on_major term from vector");
        assert!(s_iter.next().is_none(), "reading data from a vector of wrong size (too big)");
        for (coeff_out, coeff_in) in
            out.king_danger_coeffs.iter_mut().zip(&data[data.len() - 8 - 3..data.len() - 8])
        {
            *coeff_out = *coeff_in;
        }
        for (weight_out, weight_in) in
            out.king_danger_piece_weights.iter_mut().zip(&data[data.len() - 8..])
        {
            *weight_out = *weight_in;
        }
        out
    }

    pub fn save_param_vec<P: AsRef<Path>>(param_vec: &[i32], path: P) {
        let mut output = std::fs::File::create(path).unwrap();
        let out = param_vec.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
        std::io::Write::write_all(&mut output, out.as_bytes()).unwrap();
    }

    pub fn load_param_vec<P: AsRef<Path>>(path: P) -> Result<Vec<i32>, Box<dyn Error>> {
        let mut params = Vec::new();
        let input = std::fs::read_to_string(path)?;
        for param in input.trim().split(',') {
            let param: i32 = param.parse()?;
            params.push(param);
        }
        Ok(params)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let vec = Self::load_param_vec(path)?;
        Ok(Self::devectorise(&vec))
    }
}

mod tests {
    #[test]
    fn params_round_trip() {
        use crate::board::evaluation::EvalParams;

        let params = EvalParams::default();
        let vec = params.vectorise();
        let params2 = EvalParams::devectorise(&vec);
        assert_eq!(params, params2);

        let n_params = vec.len();
        for _ in 0..100 {
            let vec = (0..n_params).map(|_| rand::random::<i32>()).collect::<Vec<_>>();
            let params = EvalParams::devectorise(&vec);
            let vec2 = params.vectorise();
            assert_eq!(vec, vec2);
        }
    }

    #[test]
    fn params_round_trip_fuzz() {
        use crate::board::evaluation::EvalParams;

        let n_params = EvalParams::default().vectorise().len();

        for _ in 1..100 {
            let vec = (0..n_params).map(|_| rand::random::<i32>()).collect::<Vec<_>>();
            let params = EvalParams::devectorise(&vec);
            let vec2 = params.vectorise();
            assert_eq!(vec, vec2);
        }
    }
}
