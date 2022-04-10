#![allow(dead_code)]

use std::mem;

use crate::{definitions::{BOARD_N_SQUARES, MAX_GAME_MOVES, Piece, Undo, OFF_BOARD, NO_SQUARE}, lookups::SQ64_TO_SQ120};
use crate::{definitions::Colour, lookups::{PIECE_KEYS, SIDE_KEY, CASTLE_KEYS}};

#[derive(Clone, Eq, PartialEq)]
pub struct Board {
    pieces: [u8; BOARD_N_SQUARES],
    pawns: [u64; 3],
    king_sq: [u8; 2],
    side: u8,
    ep_sq: u8,
    fifty_move_counter: u8,
    ply: usize,
    hist_ply: usize,
    key: u64,
    piece_counts: [u8; 13],
    big_piece_counts: [u8; 3],
    major_piece_counts: [u8; 3],
    minor_piece_counts: [u8; 3],
    castle_perm: u8,
    history: [Undo; MAX_GAME_MOVES],
    p_list: [[u8; 10]; 13], // p_list[piece][N]
}

impl Board {
    pub fn generate_pos_key(&self) -> u64 {
        let mut key = 0;
        for sq in 0..BOARD_N_SQUARES {
            let piece = self.pieces[sq];
            if piece != Piece::Empty as u8 {
                debug_assert!(piece >= Piece::WP as u8 && piece <= Piece::BK as u8);
                key ^= PIECE_KEYS[piece as usize][sq];
            }
        }

        if self.side == Colour::White as u8 {
            key ^= SIDE_KEY;
        }

        if self.ep_sq != 0 {
            debug_assert!((self.ep_sq as usize) < BOARD_N_SQUARES);
            key ^= PIECE_KEYS[Piece::Empty as usize][self.ep_sq as usize];
        }

        debug_assert!(self.castle_perm <= 15);

        key ^= CASTLE_KEYS[self.castle_perm as usize];

        key
    }

    pub fn reset(&mut self) {
        self.pieces.fill(OFF_BOARD);
        for &i in &SQ64_TO_SQ120 {
            self.pieces[i as usize] = Piece::Empty as u8;
        }
        self.pawns.fill(0);
        self.big_piece_counts.fill(0);
        self.major_piece_counts.fill(0);
        self.minor_piece_counts.fill(0);
        self.piece_counts.fill(0);
        self.king_sq.fill(NO_SQUARE);
        self.side = Colour::Both as u8;
        self.ep_sq = NO_SQUARE;
        self.fifty_move_counter = 0;
        self.ply = 0;
        self.hist_ply = 0;
        self.key = 0;
        self.castle_perm = 0;
    }
}