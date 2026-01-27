#![expect(clippy::identity_op)]

use crate::{
    chess::{
        piece::{Colour, Piece, PieceType},
        piecelayout::PieceLayout,
        types::{CastlingRights, File, Square},
    },
    errors::QuickParseError,
};

/// A successfully-parsed quick format position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quick {
    pub board: PieceLayout,
    pub rights: CastlingRights,
    pub turn: Colour,
    pub score: i16,
    pub outcome: QuickOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuickOutcome {
    Unknown,
    BlackWin,
    Draw,
    WhiteWin,
}

impl Quick {
    pub fn parse(record: &str) -> Result<Self, QuickParseError> {
        // Positions in the quick format have a fixed length of 40 bytes:
        //   AAA M KQRRBBNNPPPPPPPP kqrrbbnnpppppppp VVVV
        //
        // Each byte is encoded as one of the characters [0-9A-Za-z&$:].
        //   After decoding, its value is in the range 0−63 or is 255.
        //   Using 65 encoding values per byte means that U+000A LINE FEED can
        //   appear in files of quick format data (so that traditional line-based
        //   utilities are still usable) and, more generally, records are valid
        //   UTF-8 so that other text-based utilities are easily usable.
        //
        // The first three bytes encode scoring information (annotations):
        //
        //   AAA = 00ss ssss  (byte 0)
        //         00ss ssss  (byte 1)
        //         00oo ssss  (byte 2)
        //
        //   oo = outcome (0 incomplete, 1 drawn, 2 white win, 3 black win)
        //   ssss ssss ssss ssss = score as an i16
        //
        // The next byte encodes two fields:
        //
        //   M = 000t rrrr
        //
        //   rrrr = castling rights
        //   t = 0 when white has the move and 1 when black has the move
        //
        // The next sixteen bytes encode the square indices (locations) of white’s
        //   pieces. When a piece is not present on the board, its index is 255.
        //   For example, white’s pieces in "8/8/8/8/8/8/4PPPP/4KBNR w K - 0 1"
        //   would be coded
        //
        //   [4, 255, 7, 255, 5, 255, 6, 255, 12, 13, 14, 15, 255, 255, 255, 255]
        //
        // The next sixteen bytes encode the square indices of black’s pieces.
        //
        // The last four bytes encode two pairs. The first value of a pair is in the
        //   range 0–5, 8–13, or 255 and indicates the kind of a piece. The second
        //   value of a pair is in the range 0−63 or 255 and indicates a square
        //   index. These two pairs are used for promotions.
        //
        // Note that the en passant target, halfmove clock, and fullmove number are
        //   not stored in this format.
        //
        // Perhaps the most significant defect of this format is depth from zeroing
        //   is not included. A future variant may include such information.

        use crate::chess::piece::PieceType::{Bishop, King, Knight, Pawn, Queen, Rook};

        const PIECETYPE: [PieceType; 16] = [
            King, Queen, Rook, Rook, Bishop, Bishop, Knight, Knight, Pawn, Pawn, Pawn, Pawn, Pawn,
            Pawn, Pawn, Pawn,
        ];

        let ary = record
            .as_bytes()
            .as_array::<40>()
            .ok_or(QuickParseError::InvalidLength(record.len()))?;

        let a0 = decode(ary[0]);
        let a1 = decode(ary[1]);
        let a2 = decode(ary[2]);
        if a0 == 255 || a1 == 255 || a2 == 255 {
            return Err(QuickParseError::InvalidAnnotation);
        }

        // Positive scores favour white and negative scores favour black
        //   (the scores are not from the perspective of the side to move).
        let score = (u16::from(a0) | u16::from(a1) << 6 | u16::from(a2 & 15) << 12).cast_signed();

        let outcome = match a2 >> 4 {
            0 => QuickOutcome::Unknown,
            1 => QuickOutcome::BlackWin,
            2 => QuickOutcome::Draw,
            3 => QuickOutcome::WhiteWin,
            _ => return Err(QuickParseError::InvalidOutcome),
        };

        let misc = decode(ary[3]);
        if misc >= 32 {
            return Err(QuickParseError::InvalidMiscellania);
        }

        let rights = CastlingRights::new(
            (misc & 1 != 0).then_some(File::H), // white_kingside
            (misc & 2 != 0).then_some(File::A), // white_queenside
            (misc & 4 != 0).then_some(File::H), // black_kingside
            (misc & 8 != 0).then_some(File::A), // black_queenside
        );

        // TODO check that the castling rights are set properly

        let turn = match misc & 16 {
            0 => Colour::White,
            _ => Colour::Black,
        };

        let mut layout = PieceLayout::default();

        for colour in Colour::all() {
            for offset in 0..16 {
                let location = decode(ary[4 + colour.index() * 16 + offset]);
                let Some(square) = Square::new(location) else {
                    continue;
                };
                if layout.occupied().contains_square(square) {
                    return Err(QuickParseError::DuplicateLocation(square));
                }
                let piece = Piece::new(colour, PIECETYPE[offset]);
                layout.set_piece_at(square, piece);
            }
        }

        // TODO check that the kings are present

        for offset in [36, 38] {
            let piece_u8 = decode(ary[offset + 0]);
            let location = decode(ary[offset + 1]);
            let Some(square) = Square::new(location) else {
                continue;
            };
            if layout.occupied().contains_square(square) {
                return Err(QuickParseError::DuplicateLocation(square));
            }
            let colour = match piece_u8 & 8 {
                0 => Colour::White,
                _ => Colour::Black,
            };
            let piece_type = match piece_u8 & !8 {
                0 => King,
                1 => Queen,
                2 => Rook,
                3 => Bishop,
                4 => Knight,
                5 => Pawn,
                _ => return Err(QuickParseError::InvalidVariable),
            };
            layout.set_piece_at(square, Piece::new(colour, piece_type));
        }

        // TODO check that the position is legal

        Ok(Self {
            board: layout,
            rights,
            turn,
            score,
            outcome,
        })
    }
}

#[inline]
fn decode(c: u8) -> u8 {
    DECODE[c as usize]
}

#[rustfmt::skip]
static DECODE: [u8; 256] = [
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255,  63, 255,  62, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9, 255, 255, 255, 255, 255, 255,
  255,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
   25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35, 255, 255, 255, 255, 255,
  255,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,
   51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];
