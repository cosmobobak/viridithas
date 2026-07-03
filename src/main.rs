// SPDX-License-Identifier: AGPL-3.0-only
//
// Viridithas.
// Copyright (C) 2022-2026 Cosmo Bobak
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::missing_const_for_fn)]
#![deny(missing_docs, clippy::undocumented_unsafe_blocks)]

//! Viridithas, a UCI chess engine written in Rust.

#[macro_use]
mod macros;

#[cfg(feature = "datagen")]
mod datagen;

#[cfg(feature = "stats")]
pub mod stats;

mod bench;
mod chess;
mod cli;
mod cuckoo;
mod errors;
mod evaluation;
mod history;
mod historytable;
mod image;
mod lookups;
mod movepicker;
mod nnue;
mod perft;
mod rng;
mod search;
mod searchinfo;
mod stack;
mod tablebases;
mod term;
mod threadlocal;
mod threadpool;
mod timemgmt;
mod transpositiontable;
mod uci;
mod util;

#[cfg(feature = "datagen")]
use cli::Subcommands::{Analyse, CountPositions, Datagen, Relabel, Rescale, Splat};
use cli::Subcommands::{
    Bench, EvalStats, License, Merge, NNUEDryRun, Perft, Quantise, Spsa, Verbatim, VisNNUE,
};

/// The name of the engine.
pub static NAME: &str = "Viridithas";
/// The version of the engine.
pub static VERSION: &str = env!("CARGO_PKG_VERSION");

/// The full text of the GNU Affero General Public License v3.0,
/// under which Viridithas is distributed.
pub const AGPL_LICENSE_TEXT: &str = include_str!("../LICENSE");
/// Notices for third-party components bundled into Viridithas under their own terms.
pub const THIRD_PARTY_NOTICES: &str = include_str!("../THIRD-PARTY-NOTICES.md");
/// The git commit this binary was built from ("unknown" if unavailable at build time).
pub const GIT_HASH: &str = env!("VIRIDITHAS_GIT_HASH");

/// Print Viridithas's copyright banner and third-party
/// notices and optionally the full text of the licence.
pub fn print_license(full: bool) {
    println!("Viridithas, a superhuman chess engine.");
    println!("Copyright (C) 2022-2026 Cosmo Bobak");
    println!();
    println!("This program is free software: you can redistribute it and/or modify it");
    println!("under the terms of the GNU Affero General Public License, version 3.");
    println!("This program comes with ABSOLUTELY NO WARRANTY.");
    println!();
    println!("{}", corresponding_source_notice());
    println!();
    print!("{THIRD_PARTY_NOTICES}");
    if full {
        println!();
        print!("{AGPL_LICENSE_TEXT}");
    } else {
        println!(
            "\nRun `viridithas license --full` (CLI) or `license full` (UCI) to print the full \
             GNU AGPL v3, or see <https://www.gnu.org/licenses/agpl-3.0.html>."
        );
    }
}

fn corresponding_source_notice() -> String {
    const REPO: &str = "https://github.com/cosmobobak/viridithas";
    if GIT_HASH == "unknown" {
        format!("Built from Viridithas {VERSION}. Complete corresponding source: <{REPO}>.")
    } else {
        let modified = if env!("VIRIDITHAS_GIT_DIRTY") == "1" {
            " plus uncommitted local modifications"
        } else {
            ""
        };
        format!(
            "Built from commit {GIT_HASH}{modified}.\n\
             Complete corresponding source: <{REPO}/tree/{GIT_HASH}>."
        )
    }
}

fn main() -> anyhow::Result<()> {
    if std::env::args_os().len() == 1 {
        // fast path to UCI:
        return Ok(uci::main_loop()?);
    }

    let cli = <cli::Cli as clap::Parser>::parse();

    match cli.subcommand {
        Some(Bench { depth, threads }) => {
            let nnue_params = nnue::network::NNUEParams::decompress_and_alloc()?;
            let stopped = std::sync::atomic::AtomicBool::new(false);
            let nodes = std::sync::atomic::AtomicU64::new(0);
            let tbhits = std::sync::atomic::AtomicU64::new(0);
            let control = searchinfo::Control::default();
            let info = searchinfo::SearchInfo::new(&stopped, &nodes, &tbhits, &control);
            Ok(uci::bench(
                "openbench",
                &info.conf,
                nnue_params,
                depth,
                threads,
            )?)
        }
        Some(Perft) => perft::gamut(),
        Some(License { full }) => {
            print_license(full);
            Ok(())
        }
        Some(Quantise { input, output }) => nnue::network::quantise(&input, &output),
        Some(Merge { input, output }) => nnue::network::merge(&input, &output),
        Some(Verbatim { output }) => nnue::network::dump_verbatim(&output),
        Some(VisNNUE) => nnue::network::visualise_nnue(),
        Some(NNUEDryRun) => nnue::network::dry_run(),
        Some(Spsa { json }) => {
            if json {
                println!(
                    "{}",
                    search::parameters::Config::default().emit_json_for_spsa()
                );
            } else {
                println!(
                    "{}",
                    search::parameters::Config::default().emit_csv_for_spsa()
                );
            }
            Ok(())
        }
        Some(EvalStats {
            input,
            output,
            bucket,
        }) => evaluation::eval_stats(&input, output.as_deref(), bucket),
        #[cfg(feature = "datagen")]
        Some(Analyse { input }) => datagen::dataset_stats(&input),
        #[cfg(feature = "datagen")]
        Some(CountPositions { input }) => datagen::dataset_count(&input),
        #[cfg(feature = "datagen")]
        Some(Rescale {
            scale,
            input,
            output,
        }) => datagen::run_rescale(&input, &output, scale),
        #[cfg(feature = "datagen")]
        Some(Relabel { input, output }) => datagen::run_relabel(&input, &output),
        #[cfg(feature = "datagen")]
        Some(Splat {
            input,
            marlinformat,
            pgn,
            output,
            limit,
            cfg_path,
            annotate,
        }) => {
            if pgn {
                datagen::run_topgn(&input, &output, limit, annotate)
            } else {
                datagen::run_splat(&input, &output, cfg_path.as_deref(), marlinformat, limit)
            }
        }
        #[cfg(feature = "datagen")]
        Some(Datagen {
            games,
            threads,
            tbs,
            book,
            nodes,
            dfrc,
        }) => datagen::gen_data_main(datagen::DataGenOptionsBuilder {
            games,
            threads,
            tbs,
            book,
            nodes,
            dfrc,
        }),
        None => Ok(uci::main_loop()?),
    }
}
