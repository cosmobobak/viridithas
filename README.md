<div align="center">

  ![Logo](assets/images/logo2.png)
  
  [![Build][build-badge]][build-link]
  [![License][license-badge]][license-link]
  [![Release][release-badge]][release-link]
  [![Commits][commits-badge]][commits-link]
  
</div>

Viridithas is a very strong chess playing program, developed in his current form since 2022-04-10.

Viridithas communicates over the Universal Chess Interface, which means you can talk to him in the terminal,
or install him into a UCI-compatible graphical user interface, like [En Croissant](https://encroissant.org/download) or [Nibbler](https://github.com/rooklift/nibbler).

Some writing on the internals and development of Viridithas can be found on [his author's website](https://cosmo.tardis.ac).

The Viridithas project prides itself on using original training data for its neural networks.

PS: Viridithas is enormously fond of the [Stormphrax](https://github.com/Ciekce/Stormphrax) chess engine.

## Building Viridithas

If you just want one of the official releases of Viridithas, check out the **Releases** tab on the right.
If you want to build a specific version of Viridithas from source (say, the latest and greatest master commit) then follow these instructions:

0. Before following any of these instructions, make sure you have [Rust](https://www.rust-lang.org/tools/install) installed.
   You may also need to install the `clang` C compiler, as Viridithas relies on [Fathom](https://github.com/jdart1/Fathom) for tablebase probing support.
1. Clone this repository to your machine via `git clone git@github.com:cosmobobak/viridithas.git`.
2. Enter the source directory via `cd viridithas`.
3. Download the corresponding neural network for the version of Viridithas that you are compiling and save it in the source root as `viridithas.nnue.zst`.
   All of Viridithas's neural networks can be found in the releases of the [viridithas-networks](https://github.com/cosmobobak/viridithas-networks) repo.
   Networks are stored seperately from this repo due to their considerable size.

    **If you just want the latest neural net**, you can download it with the command `curl -s "https://api.github.com/repos/cosmobobak/viridithas-networks/releases/latest" | grep -o '"browser_download_url": "[^"]*' | awk -F'"' '{print $4}' | xargs -L 1 wget -O viridithas.nnue.zst`.

4. Build Viridithas.
   
   **On Windows**, run 
   ```
   > $env:RUSTFLAGS="-C target-cpu=native"
   > cargo b -r --features syzygy,bindgen
   ``` 
   **On Linux**, run
   ```
   > RUSTFLAGS="-C target-cpu=native" cargo b -r --features syzygy,bindgen
   ```
   You now have a fully-functional version of Viridithas at the path `target/release/viridithas`.

## Evaluation Development History/Originality (HCE/NNUE)

- First evaluation was a simple material / PSQT approach, using values from PeSTO.
- Early in development, a local-search [Texel tuning](https://www.chessprogramming.org/Texel%27s_Tuning_Method) module was developed,
  and Viridithas was retrained on the Lichess Elite dataset, removing the PeSTO values with increased strength.
- Many new evaluation terms were added during development, and were continually re-tuned on Viridithas's self-play games.
- The first NNUE was trained on a dataset of games played by Viridithas 2.7.0, 2.6.0, and 2.5.0, all rescored with a development
  version of Viridithas 2.7.0 at low depth.
- Subsequent networks were trained on additional self-play games by subsequent versions of Viridithas.
  The 13th-generation network, and many since, include positions from the Lichess Elite dataset rescored by Viridithas.
- Between versions 7.0.0 and 8.0.0, original datagen code was written that allows Viridithas to generate data without need for an
  opening book to ensure game variety, resulting in even greater strength of play.

All neural networks currently used in the development of Viridithas are trained exclusively on its own self-play games,
and no network has ever been trained on the output of an engine other than Viridithas.

## Thanks and Acknowledgements

[python-chess](https://github.com/niklasf/python-chess), without which I would never have been able to gain a love for chess programming.

The [VICE](https://www.youtube.com/playlist?list=PLZ1QII7yudbc-Ky058TEaOstZHVbT-2hg) video series, which was invaluable for explaining how various chess engine concepts can be implemented.

Andrew Grant's [Ethereal](https://github.com/AndyGrant/Ethereal), the exceedingly clear code of which helped me realise several horrible flaws in Viridithas.

[weather-factory](https://github.com/dsekercioglu/weather-factory), which I used only minimally, but which is still responsible for about ~10 elo in Viridithas.

[marlinflow](https://github.com/dsekercioglu/marlinflow), was responsible for neural network training for Viridithas up until version 11.0.0.

[bullet](https://github.com/jw1912/bullet), which I have used for neural network training in the development of Viridithas since version 11.0.0.

[The SweHosting OpenBench Instance](https://chess.swehosting.se/), which is invaluable in testing new patches and features.

[build-badge]:https://img.shields.io/github/actions/workflow/status/cosmobobak/viridithas/rust.yml?branch=master&logo=github&style=for-the-badge
[build-link]:https://github.com/cosmobobak/viridithas/actions/workflows/rust.yml
[commits-badge]:https://img.shields.io/github/commits-since/cosmobobak/viridithas/latest?style=for-the-badge
[commits-link]:https://github.com/cosmobobak/viridithas/commits/master
[release-badge]:https://img.shields.io/github/v/tag/cosmobobak/viridithas?style=for-the-badge&label=official%20release
[release-link]:https://github.com/cosmobobak/viridithas/releases/latest
[license-badge]:https://img.shields.io/github/license/cosmobobak/viridithas?style=for-the-badge&label=license&color=success
[license-link]:https://github.com/cosmobobak/viridithas/blob/master/LICENSE

