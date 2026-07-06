// SPDX-License-Identifier: AGPL-3.0-only

use std::mem::offset_of;

use crate::{
    chess::{
        board::Board,
        piece::{Piece, PieceType},
        types::Square,
    },
    nnue::{
        geometry::{self, BitRays},
        network::{AuxUpdateBuffer, ThreatFeatureUpdate},
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

#[cfg(debug_assertions)]
#[expect(clippy::zero_prefixed_literal)]
fn check_rays(rays: geometry::Vector) {
    #[rustfmt::skip]
    const VALID: [u8; 13] = [
        00, 01, 02, 03, 04, 05,
        06, 07, 08, 09, 10, 11,
        // Safety: All bitpatterns of `u8` are valid.
        unsafe { std::mem::transmute::<Option<Piece>, u8>(None) }
    ];

    // Safety: All bitpatterns of `u8` are valid.
    let ray_bytes: [u8; size_of::<geometry::Vector>()] = unsafe { std::mem::transmute(rays) };

    for byte in ray_bytes {
        assert!(
            VALID.contains(&byte),
            "Got invalid byte for variants of Option<Piece>: {byte}"
        );
    }
}

#[cfg(debug_assertions)]
fn check_indexes(squares: geometry::Vector) {
    // Safety: All bitpatterns of `u8` are valid.
    let square_bytes: [u8; size_of::<geometry::Vector>()] = unsafe { std::mem::transmute(squares) };

    for byte in square_bytes {
        // 0×80 is the sentinel for unused permutation slots.
        // This value is selected s.t. a _mm256_shuffle_epi8
        // zeroes containing lanes, as 0×80’s top bit is one.
        // <3 https://en.algorithmica.org/hpc/simd/shuffling/
        assert!(
            byte < 64 || byte == 0x80,
            "Got invalid byte for square index: {byte}"
        );
    }
}

#[cfg(target_feature = "avx512vbmi")]
mod vbmi {
    use std::arch::x86_64::{
        __m128i, _mm_storeu_si128, _mm_unpackhi_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
        _mm512_castsi512_si128, _mm512_mask_mov_epi8, _mm512_maskz_compress_epi8,
        _mm512_permutex2var_epi8, _mm512_set_epi8, _mm512_set1_epi16, _mm512_storeu_si512,
    };

    use crate::{
        chess::{piece::Piece, types::Square},
        nnue::{
            geometry,
            network::{
                AuxUpdateBuffer,
                threat_updates::{AddSub, Direction},
            },
        },
    };

    /// Add threats directly related to a piece’s presence
    /// on some square. These updates could be additive or
    /// subtractive, and the threats are either inbound or
    /// outbound from the piece. Also see `push_discovered`.
    pub unsafe fn push_focus<Op: AddSub, Dir: Direction>(
        updates: &mut AuxUpdateBuffer,
        indexes: geometry::Vector, // square indexes
        rays: geometry::Vector,    // pieces on said squares
        br: geometry::BitRays,     // bitrays where set bit sees focus square
        piece: Piece,              // piece on the focus square
        sq: Square,                // the focus square
    ) {
        #[cfg(debug_assertions)]
        super::check_rays(rays);
        #[cfg(debug_assertions)]
        super::check_indexes(indexes);

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
            let pair2_sq = _mm512_maskz_compress_epi8(br.inner(), indexes.raw);
            let pair2_piece = _mm512_maskz_compress_epi8(br.inner(), rays.raw);
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

    /// Add threats revealed by removal of some piece from
    /// some square, or blocked by arrival of such a piece
    /// on such a square. For direct threats → `push_focus`.
    pub unsafe fn push_discovered<Op: AddSub>(
        updates: &mut AuxUpdateBuffer,
        idxs: geometry::Vector,
        rays: geometry::Vector,
        sliders: geometry::BitRays,
        victims: geometry::BitRays,
    ) {
        #[cfg(debug_assertions)]
        super::check_rays(rays);
        #[cfg(debug_assertions)]
        super::check_indexes(idxs);

        // Safety: TODO
        #[expect(clippy::cast_ptr_alignment)]
        unsafe {
            let count = victims.count_ones();

            debug_assert_eq!(count, sliders.count_ones());

            let p1 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(sliders.inner(), rays.raw));
            let sq1 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(sliders.inner(), idxs.raw));
            let rays = rays.flip();
            let idxs = idxs.flip();
            let p2 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(victims.inner(), rays.raw));
            let sq2 = _mm512_castsi512_si128(_mm512_maskz_compress_epi8(victims.inner(), idxs.raw));

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
        chess::{piece::Piece, types::Square},
        nnue::{
            geometry,
            network::{
                AuxUpdateBuffer, ThreatFeatureUpdate,
                threat_updates::{AddSub, Direction},
            },
        },
    };

    /// Add threats directly related to a piece’s presence
    /// on some square. These updates could be additive or
    /// subtractive, and the threats are either inbound or
    /// outbound from the piece. Also see `push_discovered`.
    pub unsafe fn push_focus<Op: AddSub, Dir: Direction>(
        updates: &mut AuxUpdateBuffer,
        indexes: geometry::Vector, // square indexes
        rays: geometry::Vector,    // pieces on said squares
        br: geometry::BitRays,     // bitrays where set bit sees focus square
        piece: Piece,              // piece on the focus square
        sq: Square,                // the focus square
    ) {
        #[cfg(debug_assertions)]
        super::check_rays(rays);
        #[cfg(debug_assertions)]
        super::check_indexes(indexes);

        // Safety: Caller must ensure that the contents of `rays`
        // and `indexes` map to valid variants of `Option<Piece>`
        // and `Square` (respectively), to license transmutation.
        unsafe {
            let others = std::mem::transmute::<geometry::Vector, [Option<Piece>; 64]>(rays);
            let other_sqs = std::mem::transmute::<geometry::Vector, [u8; 64]>(indexes);

            for i in br {
                let other = others[i].unwrap_unchecked();
                // Safety: `br` only has bits set for occupied slots,
                // where the value is a valid square index in (0..64).
                // The 0×80 sentinels in empty slots aren’t selected.
                let other_sq = Square::new_unchecked(other_sqs[i]);

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

    /// Add threats revealed by removal of some piece from
    /// some square, or blocked by arrival of such a piece
    /// on such a square. For direct threats → `push_focus`.
    pub unsafe fn push_discovered<Op: AddSub>(
        updates: &mut AuxUpdateBuffer,
        idxs: geometry::Vector,
        rays: geometry::Vector,
        sliders: geometry::BitRays,
        victims: geometry::BitRays,
    ) {
        #[cfg(debug_assertions)]
        super::check_rays(rays);
        #[cfg(debug_assertions)]
        super::check_indexes(idxs);

        // Safety: Caller must ensure that the contents of `rays`
        // and `indexes` map to valid variants of `Option<Piece>`
        // and `Square` (respectively), to license transmutation.
        unsafe {
            debug_assert_eq!(victims.count_ones(), sliders.count_ones());

            let pieces = std::mem::transmute::<geometry::Vector, [Option<Piece>; 64]>(rays);
            let squares = std::mem::transmute::<geometry::Vector, [u8; 64]>(idxs);

            for (slider_idx, victim_idx) in sliders.into_iter().zip(victims) {
                let attacker = pieces[slider_idx].unwrap_unchecked();
                // Safety: `br` only has bits set for occupied slots,
                // where the value is a valid square index in (0..64).
                // The 0×80 sentinels in empty slots aren’t selected.
                let from = Square::new_unchecked(squares[slider_idx]);
                let victim = pieces[(victim_idx + 32) % 64].unwrap_unchecked();
                let to = Square::new_unchecked(squares[(victim_idx + 32) % 64]);

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

/// Given the addition or subtraction of a piece on a square,
/// append the difference in threats induced into the update buffer.
///
/// This includes threats directly emitted by or inbound to the piece,
/// and discovered/blocked threats from sliders through the piece.
pub fn on_change<Op: AddSub>(
    updates: &mut AuxUpdateBuffer,
    board: &Board,
    piece: Piece,
    sq: Square,
) {
    // make an index-list & rays for `sq`, the focus-square.
    let perm = geometry::permutation_for(sq);
    let (rays, bits) = geometry::permute_mailbox(&perm, &board.state.mailbox);
    let non_king = !geometry::test_bit(bits, geometry::Bit::KING);

    // focus-square relative threats
    let closest = geometry::closest_occupied(bits);
    let outgoing_threats = geometry::outgoing_threats(piece, closest & non_king);
    let incoming_attackers = geometry::incoming_attackers(bits, closest);
    let incoming_sliders = geometry::incoming_sliders(bits, closest);

    // push focus-square relative threats (kings are neither attackers nor victims)
    if piece.piece_type() != PieceType::King {
        // Safety: `perm.indexes` & rays have valid bit-patterns.
        unsafe {
            push_focus::<Op, Outgoing>(updates, perm.indexes, rays, outgoing_threats, piece, sq);
            push_focus::<Op, Incoming>(updates, perm.indexes, rays, incoming_attackers, piece, sq);
        }
    }

    // find discovered threats, from sliders looking through the focus square.
    //
    // ray_fill detects if there is a piece anywhere within the 8 bits that make up a ray.
    // Rotation of the bitrays by 32 causes a 180 degree rotation of the rays, so the south ray
    // becomes the north ray, the south east ray becomes the north west ray, etc.
    //
    // A discovered threat exists if a victim exists on a ray opposite a slider, for example,
    // if we have a pawn on the North ray and a rook on the South ray, then a discovered
    // threat exists from the rook to the pawn.
    let victim_mask = (closest & non_king & BitRays::NON_KNIGHT).flip();
    // answer the question: do the sliders have a victim on the opposite ray?
    let valid = geometry::ray_fill(victim_mask) & geometry::ray_fill(incoming_sliders);

    // Safety: idxs & rays have valid bit-patterns.
    unsafe {
        push_discovered::<Op>(
            updates,
            perm.indexes,
            rays,
            incoming_sliders & valid,
            victim_mask & valid,
        );
    }
}

/// Given the transformation of a piece from one type into another,
/// append the difference in threats induced into the update buffer.
///
/// Because the piece is not moving, no threats are discovered or blocked.
pub fn on_mutate(
    updates: &mut AuxUpdateBuffer,
    board: &Board,
    old_piece: Piece,
    new_piece: Piece,
    sq: Square,
) {
    // make an index-list & rays for `sq`, the focus-square.
    let perm = geometry::permutation_for(sq);
    let (rays, bits) = geometry::permute_mailbox(&perm, &board.state.mailbox);
    let non_king = !geometry::test_bit(bits, geometry::Bit::KING);

    // focus-square relative threats
    let closest = geometry::closest_occupied(bits);
    let old_outgoing = geometry::outgoing_threats(old_piece, closest & non_king);
    let new_outgoing = geometry::outgoing_threats(new_piece, closest & non_king);
    let incoming = geometry::incoming_attackers(bits, closest);

    debug_assert!(old_piece.piece_type() != PieceType::King);
    // Safety: `perm.indexes` & rays have valid bit-patterns.
    unsafe {
        push_focus::<Sub, Outgoing>(updates, perm.indexes, rays, old_outgoing, old_piece, sq);
        push_focus::<Sub, Incoming>(updates, perm.indexes, rays, incoming, old_piece, sq);
        if new_piece.piece_type() != PieceType::King {
            push_focus::<Add, Outgoing>(updates, perm.indexes, rays, new_outgoing, new_piece, sq);
            push_focus::<Add, Incoming>(updates, perm.indexes, rays, incoming, new_piece, sq);
        }
    }
}

/// Given the movement of a piece from a square to another,
/// append the difference in threats induced into the update buffer.
///
/// This includes threats directly emitted by or inbound to the piece,
/// and discovered/blocked threats from sliders through the piece.
///
/// This function takes piece-types for old & new, in case you wish to
/// transform the type of the piece as it moves (e.g. in case of promotion).
pub fn on_move(
    updates: &mut AuxUpdateBuffer,
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
    let src_non_king = !geometry::test_bit(src_bits, geometry::Bit::KING);
    let dst_non_king = !geometry::test_bit(dst_bits, geometry::Bit::KING);

    let src_closest = geometry::closest_occupied(src_bits);
    let dst_closest = geometry::closest_occupied(dst_bits);
    let src_outgoing = geometry::outgoing_threats(old_piece, src_closest & src_non_king);
    let dst_outgoing = geometry::outgoing_threats(new_piece, dst_closest & dst_non_king);
    let src_incoming = geometry::incoming_attackers(src_bits, src_closest);
    let dst_incoming = geometry::incoming_attackers(dst_bits, dst_closest);
    let src_sliders = geometry::incoming_sliders(src_bits, src_closest);
    let dst_sliders = geometry::incoming_sliders(dst_bits, dst_closest);

    let src_idxs = src_perm.indexes;
    let dst_idxs = dst_perm.indexes;
    debug_assert!(
        old_piece.piece_type() != PieceType::King || old_piece == new_piece,
        "cannot transmute to/from king"
    );
    if old_piece.piece_type() != PieceType::King {
        // Safety: `perm.indexes` & rays have valid bit-patterns.
        unsafe {
            push_focus::<Sub, Outgoing>(updates, src_idxs, src_rays, src_outgoing, old_piece, src);
            push_focus::<Add, Outgoing>(updates, dst_idxs, dst_rays, dst_outgoing, new_piece, dst);
            push_focus::<Sub, Incoming>(updates, src_idxs, src_rays, src_incoming, old_piece, src);
            push_focus::<Add, Incoming>(updates, dst_idxs, dst_rays, dst_incoming, new_piece, dst);
        }
    }

    let src_victim_mask = (src_closest & src_non_king & BitRays::NON_KNIGHT).flip();
    let dst_victim_mask = (dst_closest & dst_non_king & BitRays::NON_KNIGHT).flip();
    let src_valid = geometry::ray_fill(src_victim_mask) & geometry::ray_fill(src_sliders);
    let dst_valid = geometry::ray_fill(dst_victim_mask) & geometry::ray_fill(dst_sliders);

    // Safety: idxs & rays have valid bit-patterns.
    unsafe {
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
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use arrayvec::ArrayVec;

    use crate::{
        chess::{
            board::{Board, movegen::attacks_by_type},
            piece::PieceType,
        },
        nnue::{accumulator, network::UpdateBuffer},
    };

    use super::*;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    /// Collect threats using the simple attack-table method.
    /// Each threat appears exactly once.
    fn collect_threats_simple(board: &Board) -> Vec<ThreatFeatureUpdate> {
        let mut threats = Vec::new();
        let bbs = &board.state.bbs;
        let occ = bbs.occupied();
        let non_kings = occ & !bbs.pieces[PieceType::King];

        for from in non_kings {
            let attacker = board.state.mailbox[from].unwrap();
            let targets = occ & attacks_by_type(attacker, from, occ) & !bbs.pieces[PieceType::King];
            for to in targets {
                let victim = board.state.mailbox[to].unwrap();
                threats.push(ThreatFeatureUpdate {
                    attacker,
                    from,
                    victim,
                    to,
                });
            }
        }
        threats.sort();
        threats
    }

    fn assert_threats_eq(
        expected: &[ThreatFeatureUpdate],
        actual: &[ThreatFeatureUpdate],
        context: impl Display,
    ) {
        use std::cmp::Ordering::{Equal, Greater, Less};

        if expected == actual {
            return;
        }

        let mut missing = Vec::new();
        let mut extra = Vec::new();
        let (mut ei, mut ai) = (0, 0);
        while ei < expected.len() && ai < actual.len() {
            match expected[ei].cmp(&actual[ai]) {
                Equal => {
                    ei += 1;
                    ai += 1;
                }
                Less => {
                    missing.push(expected[ei]);
                    ei += 1;
                }
                Greater => {
                    extra.push(actual[ai]);
                    ai += 1;
                }
            }
        }
        missing.extend_from_slice(&expected[ei..]);
        extra.extend_from_slice(&actual[ai..]);

        panic!(
            "threat mismatch: {context}\n\
             missing ({count_m}): {missing:?}\n\
             extra ({count_e}): {extra:?}",
            count_m = missing.len(),
            count_e = extra.len(),
        );
    }

    fn check_incremental_quiet(fen: &str, uci_move: &str) {
        let board_before = Board::from_fen(fen).unwrap();
        let m = board_before.parse_uci(uci_move).unwrap();

        let threats_before = collect_threats_simple(&board_before);

        let mut buf = AuxUpdateBuffer::default();
        let piece = board_before.state.mailbox[m.from()].unwrap();

        // Remove threats before the move
        on_change::<Sub>(&mut buf, &board_before, piece, m.from());

        let mut board_after = board_before.clone();
        board_after.make_move_simple(m);

        // Add threats after the move
        on_change::<Add>(&mut buf, &board_after, piece, m.to());

        // Apply diff: remove subs, add adds, filtering king threats
        let mut result = threats_before;
        for &sub in &buf.sub {
            let idx = result.binary_search(&sub).unwrap();
            result.remove(idx);
        }
        for &add in &buf.add {
            let idx = result.binary_search(&add).unwrap_err();
            result.insert(idx, add);
        }

        let expected = collect_threats_simple(&board_after);
        assert_threats_eq(
            &expected,
            &result,
            format!("incremental after {uci_move} from {fen}"),
        );
    }

    #[test]
    fn startpos_threats_after_moves() {
        check_move(STARTPOS, "e2e4");
        check_move(STARTPOS, "g1f3");
        check_move(STARTPOS, "d2d4");
    }

    #[test]
    fn kiwipete_threats_after_moves() {
        check_move(KIWIPETE, "e5d3");
        check_move(KIWIPETE, "f3f5");
        check_move(KIWIPETE, "e5f7");
        check_move(KIWIPETE, "d5e6");
    }

    #[test]
    fn startpos_incremental_quiet() {
        check_incremental_quiet(STARTPOS, "e2e4");
        check_incremental_quiet(STARTPOS, "g1f3");
        check_incremental_quiet(STARTPOS, "b1c3");
    }

    #[test]
    fn kiwipete_incremental_quiet() {
        check_incremental_quiet(KIWIPETE, "e5d3");
        check_incremental_quiet(KIWIPETE, "f3f5");
        check_incremental_quiet(KIWIPETE, "a1b1");
    }

    #[test]
    fn kiwipete_c3b1_king_victim() {
        check_incremental_quiet(KIWIPETE, "c3b1");
    }

    #[test]
    fn kiwipete_e1d1_king_victim() {
        check_move(KIWIPETE, "e1d1");
    }

    fn apply_threat_diff(
        mut before: Vec<ThreatFeatureUpdate>,
        subs: &[ThreatFeatureUpdate],
        adds: &[ThreatFeatureUpdate],
    ) -> Vec<ThreatFeatureUpdate> {
        before.extend_from_slice(adds);
        before.sort_unstable();
        for sub in subs {
            if let Ok(idx) = before.binary_search(sub) {
                before.remove(idx);
            } else {
                panic!("attempted to remove non-existent threat {sub:?}");
            }
        }
        before
    }

    fn check_move(fen: &str, uci_move: &str) {
        let board = Board::from_fen(fen).unwrap();
        let m = board.parse_uci(uci_move).unwrap();
        let threats_before = collect_threats_simple(&board);

        let mut board_after = board;
        let mut buffer = UpdateBuffer::default();
        board_after.make_move_base(m, &mut buffer);

        let simple = collect_threats_simple(&board_after);

        let incremental = apply_threat_diff(threats_before, &buffer.aux.sub, &buffer.aux.add);
        assert_threats_eq(
            &simple,
            &incremental,
            format!("incremental after {uci_move} from {fen}"),
        );
    }

    /// Exhaustive: for every legal move, verify geometry matches simple.
    fn check_all_moves(fen: &str) {
        let board = Board::from_fen(fen).unwrap();
        for m in board.legal_moves() {
            let uci = format!("{}", m.display(board.rules()));
            check_move(fen, &uci);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // too slow.
    fn startpos_all_moves() {
        check_all_moves(STARTPOS);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // too slow.
    fn kiwipete_all_moves() {
        check_all_moves(KIWIPETE);
    }

    // ---- active index tests ----

    use crate::{
        chess::piece::Colour,
        nnue::network::{
            L1_SIZE, NNUEParams, PAWN_TUPLE_FEATURES,
            feature::{pawn_pawn_index, threat_index},
            pawn_updates::PAWN_PAWN_MASKS,
        },
        util::Align,
    };

    /// Compute sorted auxiliary feature indexes (threats
    /// and pawn-pawn relations) for a given perspective.
    fn aux_indexes(board: &Board, colour: Colour) -> ArrayVec<u32, 256> {
        let bbs = &board.state.bbs;
        let king = bbs.king_sq(colour);

        // Threat features are offset to make space for the
        // pawn-pawn features that occupy 0..PAWN_TUPLE_FEATURES.
        let pawn_offset = u32::try_from(PAWN_TUPLE_FEATURES).unwrap();
        let threats = collect_threats_simple(board);
        let mut indexes = threats
            .iter()
            .filter_map(|t| {
                let (good, idx) = threat_index(colour, king, t.attacker, t.victim, t.from, t.to);
                good.then_some(pawn_offset + idx)
            })
            .collect::<ArrayVec<u32, 256>>();

        // Pawn-pawn relational features, occupying 0..PAWN_TUPLE_FEATURES.
        let our_pawns = bbs.pieces[PieceType::Pawn] & bbs.colours[colour];
        let their_pawns = bbs.pieces[PieceType::Pawn] & bbs.colours[!colour];

        let mut our_pawns_iter = our_pawns.into_iter();
        let mut their_pawns_iter = their_pawns.into_iter();

        while let Some(a) = our_pawns_iter.next() {
            let mask = PAWN_PAWN_MASKS[a.file()];
            for b in our_pawns_iter.remaining() & mask {
                indexes.push(u32::from(pawn_pawn_index(
                    colour, king, colour, a, colour, b,
                )));
            }
            for b in their_pawns & mask {
                indexes.push(u32::from(pawn_pawn_index(
                    colour, king, colour, a, !colour, b,
                )));
            }
        }

        while let Some(a) = their_pawns_iter.next() {
            let mask = PAWN_PAWN_MASKS[a.file()];
            for b in their_pawns_iter.remaining() & mask {
                indexes.push(u32::from(pawn_pawn_index(
                    colour, king, !colour, a, !colour, b,
                )));
            }
        }

        indexes.sort_unstable();
        indexes
    }

    #[test]
    fn startpos_threat_indexes() {
        let board = Board::from_fen(STARTPOS).unwrap();
        let expected = [
            0, 2, 5, 9, 14, 20, 27, 3828, 3829, 3916, 3917, 3918, 4004, 4006, 4007, 4008, 4094,
            4097, 4098, 4099, 4185, 4189, 4190, 4191, 4277, 4282, 4283, 4284, 4370, 4376, 4377,
            4378, 4464, 4471, 4472, 4559, 4898, 4917, 8270, 8271, 8291, 8292, 12743, 12841, 13632,
            13736, 19995, 19996, 19997, 22904, 36794, 36813, 40923, 40924, 40944, 40945, 47014,
            47112, 47911, 48015, 58471, 58472, 58473, 61390,
        ];
        // symmetric
        assert_eq!(
            &aux_indexes(&board, Colour::White),
            &expected[..],
            "white threats"
        );
        assert_eq!(
            &aux_indexes(&board, Colour::Black),
            &expected[..],
            "black threats"
        );
    }

    #[test]
    fn kiwipete_threat_indexes() {
        let board = Board::from_fen(KIWIPETE).unwrap();

        let white_expected = [
            0, 2, 20, 27, 173, 383, 397, 1540, 1541, 2420, 2421, 2422, 3240, 3241, 3242, 3296,
            3405, 3422, 3431, 4006, 4007, 4024, 4086, 4088, 4191, 4205, 4214, 4269, 4283, 4284,
            4306, 4348, 4370, 4471, 4472, 4535, 4571, 4997, 4998, 5000, 5668, 6426, 6766, 6768,
            6769, 8909, 12743, 12841, 20299, 20300, 20311, 21762, 23213, 27582, 29051, 34503,
            34712, 34715, 35031, 35033, 35064, 36715, 36745, 37755, 39710, 41409, 42530, 47010,
            47112, 58269, 58274, 58278, 58279, 59729,
        ];
        let black_expected = [
            14, 38, 57, 59, 440, 442, 505, 2282, 2283, 2289, 2777, 2779, 2786, 2843, 3837, 3860,
            3918, 3925, 3948, 4004, 4007, 4014, 4016, 4080, 4094, 4282, 4283, 4308, 4346, 4376,
            4378, 4401, 4464, 4472, 4495, 4559, 4570, 4573, 4781, 4973, 5004, 6011, 6655, 6657,
            6685, 8882, 9999, 12747, 12841, 20144, 20145, 20150, 20157, 21605, 34713, 34934, 34935,
            34936, 36734, 36736, 36739, 37410, 37964, 39741, 41436, 47014, 47112, 50784, 52232,
            58063, 58071, 58072, 59524, 60985,
        ];
        assert_eq!(
            &aux_indexes(&board, Colour::White),
            &white_expected[..],
            "white threats"
        );
        assert_eq!(
            &aux_indexes(&board, Colour::Black),
            &black_expected[..],
            "black threats"
        );
    }

    // ---- accumulator correctness tests ----

    /// manually sum the `l0_threat` weight rows for each active threat feature,
    /// and verify that `refresh_threats` produces the same accumulator.
    fn check_threat_accumulator(fen: &str) {
        let nnue_params = NNUEParams::decompress_and_alloc().unwrap();
        let board = Board::from_fen(fen).unwrap();

        for colour in [Colour::White, Colour::Black] {
            // compute expected accumulator by summing weight rows
            let indexes = aux_indexes(&board, colour);
            let mut expected = Align([0i16; L1_SIZE]);
            for &idx in &indexes {
                let start = idx as usize * L1_SIZE;
                let row = &nnue_params.l0_aux[start..start + L1_SIZE];
                for (e, &r) in expected.0.iter_mut().zip(row) {
                    *e += i16::from(r);
                }
            }

            // compute actual accumulator via refresh_threats
            let mut acc = crate::nnue::accumulator::Accumulator {
                halves: [Align([0i16; L1_SIZE]), Align([0i16; L1_SIZE])],
            };
            accumulator::refresh_aux(&nnue_params.l0_aux, &mut acc.halves[colour], &board, colour);

            assert_eq!(
                acc.halves[colour].0, expected.0,
                "threat accumulator mismatch for {colour:?} in {fen}"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // too slow.
    fn startpos_threat_accumulator() {
        check_threat_accumulator(STARTPOS);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // too slow.
    fn kiwipete_threat_accumulator() {
        check_threat_accumulator(KIWIPETE);
    }
}
