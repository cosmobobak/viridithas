#![allow(unused_imports)]

use crate::{
    chess::{
        board::Board,
        piece::{Colour, Piece},
        types::{Rank, Square},
    },
    nnue::network::NNUEState,
    searchinfo::SearchInfo,
};

impl Board {
    #[cfg(debug_assertions)]
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    pub fn check_validity(&self) {
        #![allow(clippy::similar_names, clippy::cast_possible_truncation)]

        // check turn
        assert!(
            !(self.side != Colour::White && self.side != Colour::Black),
            "invalid side: {:?}",
            self.side
        );

        // check square-set / piece array coherency
        for sq in Square::all() {
            let piece_ss = self.state.bbs.piece_at(sq);
            let piece_mb = self.state.mailbox[sq];
            assert_eq!(
                piece_ss, piece_mb,
                "square-set / piece array coherency corrupt: expected square {sq} to be '{piece_ss:?}' but was '{piece_mb:?}'",
            );
        }

        assert!(
            self.side == Colour::White || self.side == Colour::Black,
            "side is corrupt: expected WHITE or BLACK, got {:?}",
            self.side
        );
        assert!(
            self.state.generate_pos_keys(self.side) == self.state.keys,
            "key is corrupt: expected {:?}, got {:?}",
            self.state.generate_pos_keys(self.side),
            self.state.keys
        );

        if let Some(ep_square) = self.state.ep_square
            && !(ep_square.rank() == Rank::Six && self.side == Colour::White)
            && !(ep_square.rank() == Rank::Three && self.side == Colour::Black)
        {
            panic!(
                "en passant square is corrupt: expected square to be None or to be on ranks 6 or 3, got {ep_square} ({:?})",
                ep_square.rank()
            );
        }

        // the fifty-move counter is allowed to be *exactly* 100, to allow a finished game to be
        // created.
        assert!(
            self.state.fifty_move_counter <= 100,
            "fifty move counter is corrupt: expected 0-100, got {}",
            self.state.fifty_move_counter
        );

        // check there are the correct number of kings for each side
        assert!(
            self.state.bbs.piece_bb(Piece::WK).count() == 1,
            "white king count is corrupt: expected 1, got {}",
            self.state.bbs.piece_bb(Piece::WK).count()
        );
        assert!(
            self.state.bbs.piece_bb(Piece::BK).count() == 1,
            "black king count is corrupt: expected 1, got {}",
            self.state.bbs.piece_bb(Piece::BK).count()
        );

        assert!(
            self.state.mailbox[self.state.bbs.king_sq(Colour::White)] == Some(Piece::WK),
            "white king square is corrupt: expected white king, got {:?}",
            self.state.mailbox[self.state.bbs.king_sq(Colour::White)]
        );
        assert!(
            self.state.mailbox[self.state.bbs.king_sq(Colour::Black)] == Some(Piece::BK),
            "black king square is corrupt: expected black king, got {:?}",
            self.state.mailbox[self.state.bbs.king_sq(Colour::Black)]
        );
    }
}
