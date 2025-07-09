use crate::{
    chess::{squareset::SquareSet, types::Square},
    rng::XorShiftState,
};

macro_rules! cfor {
    ($init: stmt; $cond: expr; $step: expr; $body: block) => {
        {
            $init
            while $cond {
                $body;

                $step;
            }
        }
    }
}

/// The number of relevant bits in the bitboard representation of the attack squareset.
/// For example, on a1, a bishop has 7 squares it can move to
/// along the long diagonal - but the contents of h8 are irrelevant to movegen,
/// so the number of relevant bits on the board is 6.
pub const BISHOP_REL_BITS: u64 = 9;

/// The number of relevant bits in the bitboard representation of the attack squareset.
/// For example, on a1, a rook has 14 squares it can move to
/// along the file or rank - but the contents of a8 and h1 are irrelevant to movegen,
/// so the number of relevant bits on the board is 12.
pub const ROOK_REL_BITS: u64 = 12;

pub const fn set_occupancy(
    index: usize,
    bits_in_mask: u64,
    mut attack_mask: SquareSet,
) -> SquareSet {
    let mut occupancy = SquareSet::EMPTY;

    let mut count = 0;
    while count < bits_in_mask {
        let square = attack_mask.first().unwrap();
        attack_mask = attack_mask.remove_square(square);
        // this bitwise AND seems really weird
        if index & (1 << count) != 0 {
            occupancy = occupancy.add_square(square);
        }
        count += 1;
    }

    occupancy
}

const fn mask_bishop_attacks(sq: Square) -> SquareSet {
    let mut attacks = 0;

    // file and rank
    let (mut f, mut r);

    // target files and ranks
    let tr = sq.rank() as i32;
    let tf = sq.file() as i32;

    cfor!((r, f) = (tr + 1, tf + 1); r <= 6 && f <= 6; (r, f) = (r + 1, f + 1); {
        attacks |= 1 << (r * 8 + f);
    });
    cfor!((r, f) = (tr + 1, tf - 1); r <= 6 && f >= 1; (r, f) = (r + 1, f - 1); {
        attacks |= 1 << (r * 8 + f);
    });
    cfor!((r, f) = (tr - 1, tf + 1); r >= 1 && f <= 6; (r, f) = (r - 1, f + 1); {
        attacks |= 1 << (r * 8 + f);
    });
    cfor!((r, f) = (tr - 1, tf - 1); r >= 1 && f >= 1; (r, f) = (r - 1, f - 1); {
        attacks |= 1 << (r * 8 + f);
    });

    SquareSet::from_inner(attacks)
}

const fn mask_rook_attacks(sq: Square) -> SquareSet {
    let mut attacks = 0;

    // file and rank
    let (mut f, mut r);

    // target files and ranks
    let tr = sq.rank() as i32;
    let tf = sq.file() as i32;

    cfor!(r = tr + 1; r <= 6; r += 1; {
        attacks |= 1 << (r * 8 + tf);
    });
    cfor!(r = tr - 1; r >= 1; r -= 1; {
        attacks |= 1 << (r * 8 + tf);
    });
    cfor!(f = tf + 1; f <= 6; f += 1; {
        attacks |= 1 << (tr * 8 + f);
    });
    cfor!(f = tf - 1; f >= 1; f -= 1; {
        attacks |= 1 << (tr * 8 + f);
    });

    SquareSet::from_inner(attacks)
}

pub const fn bishop_attacks_on_the_fly(square: Square, block: SquareSet) -> SquareSet {
    let mut attacks = 0;

    // so sue me
    let block = block.inner();

    // file and rank
    let (mut f, mut r);

    // target files and ranks
    let tr = square.rank() as i32;
    let tf = square.file() as i32;

    cfor!((r, f) = (tr + 1, tf + 1); r <= 7 && f <= 7; (r, f) = (r + 1, f + 1); {
        let sq_bb = 1 << (r * 8 + f);
        attacks |= sq_bb;
        if block & sq_bb != 0 {
            break;
        }
    });
    cfor!((r, f) = (tr + 1, tf - 1); r <= 7 && f >= 0; (r, f) = (r + 1, f - 1); {
        let sq_bb = 1 << (r * 8 + f);
        attacks |= sq_bb;
        if block & sq_bb != 0 {
            break;
        }
    });
    cfor!((r, f) = (tr - 1, tf + 1); r >= 0 && f <= 7; (r, f) = (r - 1, f + 1); {
        let sq_bb = 1 << (r * 8 + f);
        attacks |= sq_bb;
        if block & sq_bb != 0 {
            break;
        }
    });
    cfor!((r, f) = (tr - 1, tf - 1); r >= 0 && f >= 0; (r, f) = (r - 1, f - 1); {
        let sq_bb = 1 << (r * 8 + f);
        attacks |= sq_bb;
        if block & sq_bb != 0 {
            break;
        }
    });

    SquareSet::from_inner(attacks)
}

pub const fn rook_attacks_on_the_fly(square: Square, block: SquareSet) -> SquareSet {
    let mut attacks = 0;

    // so sue me
    let block = block.inner();

    // file and rank
    let (mut f, mut r);

    // target files and ranks
    let tr = square.rank() as i32;
    let tf = square.file() as i32;

    cfor!(r = tr + 1; r <= 7; r += 1; {
        let sq_bb = 1 << (r * 8 + tf);
        attacks |= sq_bb;
        if block & sq_bb != 0 {
            break;
        }
    });
    cfor!(r = tr - 1; r >= 0; r -= 1; {
        let sq_bb = 1 << (r * 8 + tf);
        attacks |= sq_bb;
        if block & sq_bb != 0 {
            break;
        }
    });
    cfor!(f = tf + 1; f <= 7; f += 1; {
        let sq_bb = 1 << (tr * 8 + f);
        attacks |= sq_bb;
        if block & sq_bb != 0 {
            break;
        }
    });
    cfor!(f = tf - 1; f >= 0; f -= 1; {
        let sq_bb = 1 << (tr * 8 + f);
        attacks |= sq_bb;
        if block & sq_bb != 0 {
            break;
        }
    });

    SquareSet::from_inner(attacks)
}

/**************************************\
|       Generating magic numbers       |
|                :3                    |
\**************************************/

fn find_magic(square: Square, relevant_bits: u64, is_bishop: bool) -> u64 {
    // occupancies array
    let mut occupancies = vec![SquareSet::EMPTY; 4096];

    // attacks array
    let mut attacks = vec![SquareSet::EMPTY; 4096];

    // used indices array
    let mut used_indices = vec![SquareSet::EMPTY; 4096];

    // mask piece attack
    let mask_attack = if is_bishop {
        mask_bishop_attacks(square)
    } else {
        mask_rook_attacks(square)
    };

    // occupancy variations
    let occupancy_variations = 1 << relevant_bits;

    // loop over all occupancy variations
    cfor!(let mut count = 0; count < occupancy_variations; count += 1; {
        // init occupancies
        occupancies[count] = set_occupancy(count, relevant_bits, mask_attack);

        // init attacks
        attacks[count] = if is_bishop {
            bishop_attacks_on_the_fly(square, occupancies[count])
        } else {
            rook_attacks_on_the_fly(square, occupancies[count])
        };
    });

    let mut rng = XorShiftState::new();
    // test the magic!
    cfor!(let mut random_count = 0; random_count < 100_000_000; random_count += 1; {
        let magic = rng.random_few_bits();

        if ((mask_attack.inner().wrapping_mul(magic)) & 0xFF00_0000_0000_0000).count_ones() < 6 {
            continue;
        }

        // reset used attacks
        cfor!(let mut i = 0; i < used_indices.len(); i += 1; {
            used_indices[i] = SquareSet::EMPTY;
        });

        // init count & fail flag
        let (mut count, mut fail);

        // test magic index
        cfor!((count, fail) = (0, false); !fail && count < occupancy_variations; count += 1; {
            #[allow(clippy::cast_possible_truncation)]
            let magic_index = ((occupancies[count].inner().wrapping_mul(magic)) >> (64 - relevant_bits)) as usize;

            // if got free index
            if used_indices[magic_index] == SquareSet::EMPTY {
                // assign attack map
                used_indices[magic_index] = attacks[count];
            }

            // otherwise, fail if we have a collision
            else if used_indices[magic_index] != attacks[count] {
                fail = true;
            }
        });

        if !fail {
            return magic;
        }
    });

    // on fail
    panic!("magic number failed");
}

#[allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
pub fn init_magics() {
    println!("Generating bishop magics...");
    println!("static BISHOP_MAGICS: [u64; 64] = [");
    for square in Square::all() {
        let magic = find_magic(square, BISHOP_REL_BITS, true);
        let magic_str = format!("{magic:016X}");
        // split into blocks of four
        let magic_str = magic_str.chars().collect::<Vec<char>>();
        let magic_str = magic_str
            .chunks(4)
            .map(|chunk| chunk.iter().collect::<String>())
            .collect::<Vec<String>>()
            .join("_");
        println!("    0x{magic_str},");
    }
    println!("];");

    println!("Generating rook magics...");
    println!("static ROOK_MAGICS: [u64; 64] = [");
    for square in Square::all() {
        let magic = find_magic(square, ROOK_REL_BITS, false);
        let magic_str = format!("{magic:016X}");
        // split into blocks of four
        let magic_str = magic_str.chars().collect::<Vec<char>>();
        let magic_str = magic_str
            .chunks(4)
            .map(|chunk| chunk.iter().collect::<String>())
            .collect::<Vec<String>>()
            .join("_");
        println!("    0x{magic_str},");
    }
    println!("];");
    println!("Done!");
}

macro_rules! init_masks_with {
    ($attack_function:ident) => {{
        let mut masks = [SquareSet::EMPTY; 64];
        cfor!(let mut square = 0; square < 64; square += 1; {
            masks[square as usize] = $attack_function(Square::new_clamped(square));
        });
        masks
    }};
}

#[derive(Clone, Copy)]
pub struct MagicEntry {
    pub mask: SquareSet,
    pub magic: u64,
}

pub static BISHOP_TABLE: [MagicEntry; 64] = {
    let mut table = [MagicEntry {
        mask: SquareSet::EMPTY,
        magic: 0,
    }; 64];

    cfor!(let mut square = 0; square < 64; square += 1; {
        table[square] = MagicEntry {
            mask: BISHOP_MASKS[square],
            magic: BISHOP_MAGICS[square],
        };
    });

    table
};

pub static ROOK_TABLE: [MagicEntry; 64] = {
    let mut table = [MagicEntry {
        mask: SquareSet::EMPTY,
        magic: 0,
    }; 64];

    cfor!(let mut square = 0; square < 64; square += 1; {
        table[square] = MagicEntry {
            mask: ROOK_MASKS[square],
            magic: ROOK_MAGICS[square],
        };
    });

    table
};

const BISHOP_MASKS: [SquareSet; 64] = init_masks_with!(mask_bishop_attacks);
const ROOK_MASKS: [SquareSet; 64] = init_masks_with!(mask_rook_attacks);

// SAFETY: All bitpatterns of u64 are valid, and SquareSet is repr(transparent) around u64.
pub static BISHOP_ATTACKS: [[SquareSet; 512]; 64] =
    unsafe { std::mem::transmute(*include_bytes!("../../embeds/diagonal_attacks.bin")) };
// SAFETY: All bitpatterns of u64 are valid, and SquareSet is repr(transparent) around u64.
#[allow(clippy::large_stack_arrays)]
pub static ROOK_ATTACKS: [[SquareSet; 4096]; 64] =
    unsafe { std::mem::transmute(*include_bytes!("../../embeds/orthogonal_attacks.bin")) };

const BISHOP_MAGICS: [u64; 64] = [
    0x0080_8104_1082_0200,
    0x2010_5204_2240_1000,
    0x88A0_1411_A008_1800,
    0x1001_0500_0261_0001,
    0x9000_9082_8000_0000,
    0x2008_0442_A000_0001,
    0x0221_A800_4508_0800,
    0x0000_6020_0A40_4000,
    0x0020_1008_9440_8080,
    0x0800_0840_2140_4602,
    0x0040_8041_0029_8014,
    0x5080_2010_6040_0011,
    0x4900_0620_A000_0000,
    0x8000_0012_0030_0000,
    0x4000_0082_4110_0060,
    0x0000_0409_2016_0200,
    0x0042_0020_0024_0090,
    0x0004_8410_0420_A804,
    0x0008_0001_0200_0910,
    0x0488_0010_A810_0202,
    0x0004_0188_0404_0402,
    0x0202_1001_0828_1120,
    0xC201_1620_1010_1042,
    0x0240_0880_2201_0B80,
    0x0083_0160_0C24_0814,
    0x0000_2810_0E14_2050,
    0x0020_8800_0083_8110,
    0x0041_0800_0402_04A0,
    0x2012_0022_0600_8040,
    0x0044_0288_1900_A008,
    0x14A8_0004_804C_1080,
    0xA004_8144_0480_0F02,
    0x00C0_1802_3010_1600,
    0x000C_9052_0002_0080,
    0x0604_0008_0010_404A,
    0x0004_0401_080C_0100,
    0x0020_1210_1014_0040,
    0x0000_5000_8000_0861,
    0x8202_0902_4100_2020,
    0x2008_0220_0800_2108,
    0x0200_4024_0104_2000,
    0x0002_E032_1004_2000,
    0x0110_0400_8042_2400,
    0x9084_04C0_5840_40C0,
    0x1000_2042_0224_0408,
    0x8002_0022_0020_0200,
    0x2002_0081_0108_1414,
    0x0002_0800_2109_8404,
    0x0060_1100_8068_0000,
    0x1080_0481_0842_0000,
    0x0400_1840_1410_0000,
    0x0080_81A0_0401_2240,
    0x0011_0080_4481_82A0,
    0xA400_2000_604A_4000,
    0x0004_0028_1104_9020,
    0x0002_4A04_10A1_0220,
    0x0808_0900_8901_3000,
    0x0C80_8004_0080_5800,
    0x0001_0201_0006_1618,
    0x1202_8200_4050_1008,
    0x4130_1005_0C10_0405,
    0x0004_2482_0404_2020,
    0x0044_0044_0828_0110,
    0x6010_2200_8060_0502,
];

const ROOK_MAGICS: [u64; 64] = [
    0x8A80_1040_0080_0020,
    0x0084_0201_0080_4000,
    0x0080_0A10_0004_8020,
    0xC410_0020_B100_0200,
    0x0400_4400_0208_0420,
    0x0A80_0400_2A80_1200,
    0x0840_140C_8040_0100,
    0x0100_0082_0C41_2300,
    0x0010_8002_1240_0820,
    0x0008_0501_9000_2800,
    0x0001_0808_0010_2000,
    0x0041_0800_8020_1001,
    0x0208_2004_0800_890A,
    0x0010_8002_0000_8440,
    0x0320_0800_418A_0022,
    0x0250_0606_0020_1100,
    0x4440_0024_0086_0020,
    0x1004_4028_0008_4000,
    0x0004_1404_C014_0004,
    0x5000_4009_0800_1400,
    0x0000_0208_4100_0830,
    0x0083_0A01_0100_0500,
    0x0140_40A0_0280_4040,
    0x4400_1010_0885_4220,
    0xE008_0252_2002_2600,
    0x0440_2440_0860_3000,
    0x0008_0240_0400_9000,
    0x0801_0090_0210_0002,
    0x0400_2002_0001_0811,
    0x3204_0200_4401_2400,
    0x0002_1000_8820_0100,
    0x0208_00A0_0409_1041,
    0x0002_10C2_2420_0241,
    0x0020_0A0C_0204_0080,
    0x004D_8028_104C_0800,
    0x813C_0A00_0290_0012,
    0x0008_1042_0020_8020,
    0x2404_00A0_00A0_4080,
    0x0802_1991_0010_0042,
    0x062C_4C00_2010_0280,
    0x0020_1042_8080_0820,
    0x20C8_0100_80A8_0200,
    0x1114_0840_8046_4008,
    0x2000_0254_3000_1805,
    0x1404_C4A1_0011_0008,
    0x0000_0084_0001_2008,
    0x3045_1400_8002_2010,
    0x8040_0284_1008_0100,
    0x0220_2003_1020_4820,
    0x0200_0822_4404_8202,
    0x0009_0984_C020_8022,
    0x8000_1101_2004_0900,
    0x9000_4024_0008_0084,
    0x2402_1001_0003_8020,
    0x0098_4006_0000_8028,
    0x0001_1100_0040_200C,
    0x0102_4022_0810_8102,
    0x0440_0414_8220_4101,
    0x4004_4020_0004_0811,
    0x804A_0008_1040_2002,
    0x0008_0002_0902_0401,
    0x0440_3411_0800_9002,
    0x0000_0088_2508_4204,
    0x2084_0021_1242_8402,
];
