use std::mem::offset_of;

use crate::{
    chess::{board::Board, piece::Piece, types::Square},
    nnue::{
        geometry,
        network::{ThreatFeatureUpdate, ThreatUpdateBuffer},
    },
};

const _: () = const {
    assert!(size_of::<ThreatFeatureUpdate>() == size_of::<u32>());
    assert!(offset_of!(ThreatFeatureUpdate, attacker) == 0);
    assert!(offset_of!(ThreatFeatureUpdate, from) == 1);
    assert!(offset_of!(ThreatFeatureUpdate, victim) == 2);
    assert!(offset_of!(ThreatFeatureUpdate, to) == 3);
};

pub trait AddSub {
    const ADD: bool;
}

pub struct Add;
pub struct Sub;

impl AddSub for Add {
    const ADD: bool = true;
}
impl AddSub for Sub {
    const ADD: bool = false;
}

pub trait Direction {
    const OUTGOING: bool;
}

pub struct Outgoing;
pub struct Incoming;

impl Direction for Outgoing {
    const OUTGOING: bool = true;
}
impl Direction for Incoming {
    const OUTGOING: bool = false;
}

#[cfg(target_feature = "avx512vbmi")]
mod vbmi {
    use std::arch::x86_64::*;

    use crate::{
        chess::{piece::Piece, types::Square},
        nnue::{
            geometry,
            network::{
                ThreatUpdateBuffer,
                threat_updates::{AddSub, Direction},
            },
        },
    };

    pub fn push_focus<Op: AddSub, Dir: Direction>(
        updates: &mut ThreatUpdateBuffer,
        indices: geometry::Vector, // square indices
        rays: geometry::Vector,    // pieces on said squares
        br: geometry::BitRays,     // bitrays where set bit sees focus square
        piece: Piece,              // piece on the focus square
        sq: Square,                // the focus square
    ) {
        // Safety: TODO
        unsafe {
            #[rustfmt::skip]
            let pair2_shuffle = _mm512_set_epi8(
                79, 15, 79, 15, 78, 14, 78, 14,
                77, 13, 77, 13, 76, 12, 76, 12,
                75, 11, 75, 11, 74, 10, 74, 10,
                73,  9, 73,  9, 72,  8, 72,  8,
                71,  7, 71,  7, 70,  6, 70,  6,
                69,  5, 69,  5, 68,  4, 68,  4,
                67,  3, 67,  3, 66,  2, 66,  2,
                65,  1, 65,  1, 64,  0, 64,  0,
            );

            // piece-square pair under focus
            // todo: i hate trying to think about signed shifts
            let pair1 = _mm512_set1_epi16(piece as i16 | ((sq as i16) << 8));

            // non-focus:
            let pair2_sq = _mm512_maskz_compress_epi8(br, indices.raw);
            let pair2_piece = _mm512_maskz_compress_epi8(br, rays.raw);
            let pair2 = _mm512_permutex2var_epi8(pair2_piece, pair2_shuffle, pair2_sq);

            let mask = if Dir::OUTGOING {
                0xCCCC_CCCC_CCCC_CCCC
            } else {
                0x3333_3333_3333_3333
            };

            let vector = _mm512_mask_mov_epi8(pair1, mask, pair2);

            let buffer = if Op::ADD {
                &mut updates.add
            } else {
                &mut updates.sub
            };

            let ptr = buffer.as_mut_ptr().add(buffer.len());

            _mm512_storeu_si512(ptr.cast(), vector);

            buffer.set_len(buffer.len() + br.count_ones() as usize);
        }
    }

    pub fn push_discovered<Op: AddSub>(
        updates: &mut ThreatUpdateBuffer,
        idxs: geometry::Vector,
        rays: geometry::Vector,
        sliders: geometry::BitRays,
        victims: geometry::BitRays,
    ) {
        // Safety: TODO
        #[expect(clippy::cast_ptr_alignment)]
        unsafe {
            let count = victims.count_ones();

            debug_assert_eq!(count, sliders.count_ones());

            let p1 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(sliders, rays.raw));
            let sq1 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(sliders, idxs.raw));
            let rays = rays.flip();
            let idxs = idxs.flip();
            let p2 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(victims, rays.raw));
            let sq2 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(victims, idxs.raw));

            let pair1 = _mm_unpacklo_epi8(p1, sq1);
            let pair2 = _mm_unpacklo_epi8(p2, sq2);

            let tuple1 = _mm_unpacklo_epi16(pair1, pair2);
            let tuple2 = _mm_unpackhi_epi16(pair1, pair2);

            // this is flipped because adding a piece undiscovers threats,
            // whereas removing a piece adds discovered threats.
            let buffer = if Op::ADD {
                &mut updates.sub
            } else {
                &mut updates.add
            };

            let ptr = buffer.as_mut_ptr().add(buffer.len());

            _mm_storeu_si128(ptr.cast::<__m128i>().add(0), tuple1);
            _mm_storeu_si128(ptr.cast::<__m128i>().add(1), tuple2);

            buffer.set_len(buffer.len() + count as usize);
        }
    }
}

#[expect(clippy::useless_let_if_seq)]
#[cfg(not(target_feature = "avx512vbmi"))]
mod generic {
    use crate::{
        chess::{piece::Piece, squareset::SquareSet, types::Square},
        nnue::{
            geometry,
            network::{
                ThreatFeatureUpdate, ThreatUpdateBuffer,
                threat_updates::{AddSub, Direction},
            },
        },
    };

    pub fn push_focus<Op: AddSub, Dir: Direction>(
        updates: &mut ThreatUpdateBuffer,
        indices: geometry::Vector, // square indices
        rays: geometry::Vector,    // pieces on said squares
        br: geometry::BitRays,     // bitrays where set bit sees focus square
        piece: Piece,              // piece on the focus square
        sq: Square,                // the focus square
    ) {
        // Safety: ehhhhhhhh get back to me on that, geometry::Vector needs to be
        // inconstructible w/out unsafe :(
        unsafe {
            let others = std::mem::transmute::<geometry::Vector, [Piece; 64]>(rays);
            let other_sqs = std::mem::transmute::<geometry::Vector, [Square; 64]>(indices);

            for i in SquareSet::from_inner(br) {
                let other = others[i];
                let other_sq = other_sqs[i];

                let attacker;
                let from;
                let victim;
                let to;
                if Dir::OUTGOING {
                    attacker = piece;
                    from = sq;
                    victim = other;
                    to = other_sq;
                } else {
                    attacker = other;
                    from = other_sq;
                    victim = piece;
                    to = sq;
                }

                let feature = ThreatFeatureUpdate {
                    attacker,
                    from,
                    victim,
                    to,
                };

                if Op::ADD {
                    updates.add.push(feature);
                } else {
                    updates.sub.push(feature);
                }
            }
        }
    }

    pub fn push_discovered<Op: AddSub>(
        updates: &mut ThreatUpdateBuffer,
        idxs: geometry::Vector,
        rays: geometry::Vector,
        sliders: geometry::BitRays,
        victims: geometry::BitRays,
    ) {
        // Safety: ehhhhhhhh get back to me on that, geometry::Vector needs to be
        // inconstructible w/out unsafe :(
        unsafe {
            debug_assert_eq!(victims.count_ones(), sliders.count_ones());

            let pieces = std::mem::transmute::<geometry::Vector, [Piece; 64]>(rays);
            let squares = std::mem::transmute::<geometry::Vector, [Square; 64]>(idxs);

            let sliders = SquareSet::from_inner(sliders);
            let victims = SquareSet::from_inner(victims);

            for (slider_idx, victim_idx) in sliders.into_iter().zip(victims) {
                let attacker = pieces[slider_idx];
                let from = squares[slider_idx];
                let victim = pieces[victim_idx.wrapping_add(32)];
                let to = squares[victim_idx.wrapping_add(32)];

                let feature = ThreatFeatureUpdate {
                    attacker,
                    from,
                    victim,
                    to,
                };

                // this is flipped because adding a piece undiscovers threats,
                // whereas removing a piece adds discovered threats.
                if Op::ADD {
                    updates.sub.push(feature);
                } else {
                    updates.add.push(feature);
                }
            }
        }
    }
}

#[cfg(target_feature = "avx512vbmi")]
pub use vbmi::*;

#[cfg(not(target_feature = "avx512vbmi"))]
pub use generic::*;

pub fn update_piece_threats_on_change<Op: AddSub>(
    updates: &mut ThreatUpdateBuffer,
    board: &Board,
    piece: Piece,
    sq: Square,
) {
    // make an index-list & rays for `sq`, the focus-square.
    let perm = geometry::permutation_for(sq);
    let (rays, bits) = geometry::permute_mailbox(&perm, &board.state.mailbox);

    // focus-square relative threats
    let closest = geometry::closest_occupied(bits);
    let outgoing_threats = geometry::outgoing_threats(piece, closest);
    let incoming_attackers = geometry::incoming_attackers(bits, closest);
    let incoming_sliders = geometry::incoming_attackers(bits, closest);

    // push focus-square relative threats
    push_focus::<Op, Outgoing>(updates, perm.indices, rays, outgoing_threats, piece, sq);
    push_focus::<Op, Incoming>(updates, perm.indices, rays, incoming_attackers, piece, sq);

    // find discovered threats, from sliders looking through the focus square.
    // this is somewhat arcane, but one has it on good authority that it finds
    // all valid discovered threats ^^
    let victim_mask = (closest & 0xFEFE_FEFE_FEFE_FEFE).rotate_right(32);
    let valid = geometry::ray_fill(victim_mask) & geometry::ray_fill(incoming_sliders);

    push_discovered::<Op>(
        updates,
        perm.indices,
        rays,
        incoming_sliders & valid,
        victim_mask & valid,
    );
}

pub fn update_piece_threats_on_mutate(
    updates: &mut ThreatUpdateBuffer,
    board: &Board,
    old_piece: Piece,
    new_piece: Piece,
    sq: Square,
) {
    // make an index-list & rays for `sq`, the focus-square.
    let perm = geometry::permutation_for(sq);
    let (rays, bits) = geometry::permute_mailbox(&perm, &board.state.mailbox);

    // focus-square relative threats
    let closest = geometry::closest_occupied(bits);
    let old_outgoing = geometry::outgoing_threats(old_piece, closest);
    let new_outgoing = geometry::outgoing_threats(new_piece, closest);
    let incoming = geometry::incoming_attackers(bits, closest);

    push_focus::<Sub, Outgoing>(updates, perm.indices, rays, old_outgoing, old_piece, sq);
    push_focus::<Add, Outgoing>(updates, perm.indices, rays, new_outgoing, new_piece, sq);
    push_focus::<Sub, Incoming>(updates, perm.indices, rays, incoming, old_piece, sq);
    push_focus::<Add, Incoming>(updates, perm.indices, rays, incoming, new_piece, sq);
}

pub fn update_piece_threats_on_move(
    updates: &mut ThreatUpdateBuffer,
    board: &Board,
    old_piece: Piece,
    src: Square,
    new_piece: Piece,
    dst: Square,
) {
    let src_perm = geometry::permutation_for(src);
    let dst_perm = geometry::permutation_for(dst);
    let (src_rays, src_bits) =
        geometry::permute_mailbox_ignoring(&src_perm, &board.state.mailbox, dst);
    let (dst_rays, dst_bits) = geometry::permute_mailbox(&dst_perm, &board.state.mailbox);

    let src_closest = geometry::closest_occupied(src_bits);
    let dst_closest = geometry::closest_occupied(dst_bits);
    let src_outgoing = geometry::outgoing_threats(old_piece, src_closest);
    let dst_outgoing = geometry::outgoing_threats(new_piece, dst_closest);
    let src_incoming = geometry::incoming_attackers(src_bits, src_closest);
    let dst_incoming = geometry::incoming_attackers(dst_bits, dst_closest);
    let src_sliders = geometry::incoming_sliders(src_bits, src_closest);
    let dst_sliders = geometry::incoming_sliders(dst_bits, dst_closest);

    let src_idxs = src_perm.indices;
    let dst_idxs = dst_perm.indices;
    push_focus::<Sub, Outgoing>(updates, src_idxs, src_rays, src_outgoing, old_piece, src);
    push_focus::<Add, Outgoing>(updates, dst_idxs, dst_rays, dst_outgoing, new_piece, src);
    push_focus::<Sub, Incoming>(updates, src_idxs, src_rays, src_incoming, old_piece, src);
    push_focus::<Add, Incoming>(updates, dst_idxs, dst_rays, dst_incoming, new_piece, src);

    let src_victim_mask = (src_closest & 0xFEFE_FEFE_FEFE_FEFE).rotate_right(32);
    let dst_victim_mask = (dst_closest & 0xFEFE_FEFE_FEFE_FEFE).rotate_right(32);
    let src_valid = geometry::ray_fill(src_victim_mask) & geometry::ray_fill(src_sliders);
    let dst_valid = geometry::ray_fill(dst_victim_mask) & geometry::ray_fill(dst_sliders);

    push_discovered::<Sub>(
        updates,
        src_idxs,
        src_rays,
        src_sliders & src_valid,
        src_victim_mask & src_valid,
    );
    push_discovered::<Add>(
        updates,
        dst_idxs,
        dst_rays,
        dst_sliders & dst_valid,
        dst_victim_mask & dst_valid,
    );
}
