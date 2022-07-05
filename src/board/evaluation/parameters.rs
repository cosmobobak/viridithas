use std::{fmt::Display, error::Error};

use crate::{definitions::{WP, WN, WK, flip_rank, BP, KING, KNIGHT, flip_file}, lookups::file};

use super::{score::S, PIECE_VALUES, ISOLATED_PAWN_MALUS, DOUBLED_PAWN_MALUS, BISHOP_PAIR_BONUS, ROOK_OPEN_FILE_BONUS, ROOK_HALF_OPEN_FILE_BONUS, QUEEN_OPEN_FILE_BONUS, QUEEN_HALF_OPEN_FILE_BONUS, KNIGHT_MOBILITY_BONUS, BISHOP_MOBILITY_BONUS, ROOK_MOBILITY_BONUS, QUEEN_MOBILITY_BONUS, PASSED_PAWN_BONUS};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Parameters {
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
}

impl Default for Parameters {
    fn default() -> Self {
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
        }
    }
}

impl Display for Parameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "Parameters {{")?;
        writeln!(f, "    piece_values: {:?},", &self.piece_values[1..6])?;
        writeln!(
            f,
            "    isolated_pawn_malus: {:?},",
            self.isolated_pawn_malus
        )?;
        writeln!(f, "    doubled_pawn_malus: {:?},", self.doubled_pawn_malus)?;
        writeln!(f, "    bishop_pair_bonus: {:?},", self.bishop_pair_bonus)?;
        writeln!(
            f,
            "    rook_open_file_bonus: {:?},",
            self.rook_open_file_bonus
        )?;
        writeln!(
            f,
            "    rook_half_open_file_bonus: {:?},",
            self.rook_half_open_file_bonus
        )?;
        writeln!(
            f,
            "    queen_open_file_bonus: {:?},",
            self.queen_open_file_bonus
        )?;
        writeln!(
            f,
            "    queen_half_open_file_bonus: {:?},",
            self.queen_half_open_file_bonus
        )?;
        writeln!(
            f,
            "    knight_mobility_bonus: {:?},",
            self.knight_mobility_bonus
        )?;
        writeln!(
            f,
            "    bishop_mobility_bonus: {:?},",
            self.bishop_mobility_bonus
        )?;
        writeln!(
            f,
            "    rook_mobility_bonus: {:?},",
            self.rook_mobility_bonus
        )?;
        writeln!(
            f,
            "    queen_mobility_bonus: {:?},",
            self.queen_mobility_bonus
        )?;
        writeln!(f, "    passed_pawn_bonus: {:?},", self.passed_pawn_bonus)?;
        writeln!(
            f,
            "    piece_square_tables: {:?},",
            &self.piece_square_tables[1..7]
        )?;
        write!(f, "}}")?;
        Ok(())
    }
}

impl Parameters {
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
    };

    pub fn vectorise(&self) -> Vec<i32> {
        let ss = self.piece_values[1..6] // pawn to queen
            .iter()
            .copied()
            .chain(Some(self.isolated_pawn_malus))
            .chain(Some(self.doubled_pawn_malus))
            .chain(Some(self.bishop_pair_bonus))
            .chain(Some(S(
                self.rook_open_file_bonus.0,
                self.rook_half_open_file_bonus.0,
            )))
            .chain(Some(S(
                self.queen_open_file_bonus.0,
                self.queen_half_open_file_bonus.0,
            )))
            .chain(self.knight_mobility_bonus.into_iter())
            .chain(self.bishop_mobility_bonus.into_iter())
            .chain(self.rook_mobility_bonus.into_iter())
            .chain(self.queen_mobility_bonus.into_iter())
            .chain(self.passed_pawn_bonus.into_iter())
            // take the left halves of the white piece square tables, except for the pawn table.
            .chain(self.piece_square_tables[WP as usize].iter().copied())
            .chain(
                self.piece_square_tables[(WN as usize)..=(WK as usize)]
                    .iter()
                    .flat_map(|x| x.chunks(4).step_by(2).flatten().copied()),
            );
        ss.flat_map(|s| [s.0, s.1].into_iter()).collect()
    }

    pub fn devectorise(data: &[i32]) -> Self {
        let mut out = Self::NULL;
        let mut data = data.chunks(2).map(|x| S(x[0], x[1]));
        for p in 1..6 {
            let val = data
                .next()
                .expect("failed to read piece_value term from vector");
            out.piece_values[p] = val;
            out.piece_values[p + 6] = val;
        }
        out.isolated_pawn_malus = data
            .next()
            .expect("failed to read isolated_pawn_malus term from vector");
        out.doubled_pawn_malus = data
            .next()
            .expect("failed to read doubled_pawn_malus term from vector");
        out.bishop_pair_bonus = data
            .next()
            .expect("failed to read bishop_pair_bonus term from vector");
        let rook_file_bonus = data
            .next()
            .expect("failed to read rook_file_bonus term from vector");
        out.rook_open_file_bonus = S(rook_file_bonus.0, 0);
        out.rook_half_open_file_bonus = S(rook_file_bonus.1, 0);
        let queen_file_bonus = data
            .next()
            .expect("failed to read queen_file_bonus term from vector");
        out.queen_open_file_bonus = S(queen_file_bonus.0, 0);
        out.queen_half_open_file_bonus = S(queen_file_bonus.1, 0);
        for knight_mobility_bonus in &mut out.knight_mobility_bonus {
            *knight_mobility_bonus = data
                .next()
                .expect("failed to read knight_mobility_bonus term from vector");
        }
        for bishop_mobility_bonus in &mut out.bishop_mobility_bonus {
            *bishop_mobility_bonus = data
                .next()
                .expect("failed to read bishop_mobility_bonus term from vector");
        }
        for rook_mobility_bonus in &mut out.rook_mobility_bonus {
            *rook_mobility_bonus = data
                .next()
                .expect("failed to read rook_mobility_bonus term from vector");
        }
        for queen_mobility_bonus in &mut out.queen_mobility_bonus {
            *queen_mobility_bonus = data
                .next()
                .expect("failed to read queen_mobility_bonus term from vector");
        }
        for passed_pawn_bonus in &mut out.passed_pawn_bonus {
            *passed_pawn_bonus = data
                .next()
                .expect("failed to read passed_pawn_bonus term from vector");
        }
        // load in the pawn table
        for sq in 0..64 {
            let val = data
                .next()
                .expect("failed to read pawn piece_square_table term from vector");
            out.piece_square_tables[WP as usize][sq as usize] = val;
            out.piece_square_tables[BP as usize][flip_rank(sq) as usize] = -val;
        }
        // load in the rest of the tables
        for pt in KNIGHT..=KING {
            for sq in 0..64 {
                let file = file(sq);
                if file > 3 {
                    // load from the other half of the piece-square table.
                    // the left-hand sides of the tables are loaded first, so we
                    // can safely load out of LHS to populate RHS.
                    let mirrored_sq = flip_file(sq);
                    out.piece_square_tables[pt as usize][sq as usize] =
                        out.piece_square_tables[pt as usize][mirrored_sq as usize];
                    out.piece_square_tables[pt as usize + 6][flip_rank(sq) as usize] =
                        out.piece_square_tables[pt as usize + 6][flip_rank(mirrored_sq) as usize];
                } else {
                    let val = data
                        .next()
                        .expect("failed to read piece_square_table term from vector");
                    out.piece_square_tables[pt as usize][sq as usize] = val;
                    out.piece_square_tables[pt as usize + 6][flip_rank(sq) as usize] = -val;
                }
            }
        }
        assert!(
            data.next().is_none(),
            "reading data from a vector of wrong size (too big)"
        );
        out
    }

    pub fn save_param_vec(param_vec: &[i32], path: &str) {
        let mut output = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for param in param_vec {
            std::io::Write::write_all(&mut output, format!("{param},").as_bytes()).unwrap();
        }
        std::io::Write::flush(&mut output).unwrap();
    }

    pub fn load_param_vec(path: &str) -> Result<Vec<i32>, Box<dyn Error>> {
        let mut params = Vec::new();
        let input = std::io::BufReader::new(std::fs::File::open(path)?);
        for param in std::io::BufRead::split(input, b',') {
            let param = param?;
            let param = String::from_utf8(param)?;
            let param: i32 = param.parse()?;
            params.push(param);
        }
        Ok(params)
    }

    pub fn from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let vec = Self::load_param_vec(path)?;
        Ok(Self::devectorise(&vec))
    }
}