# Viridithas, a UCI chess engine written in Rust.

<div align="center">

  ![Visualisation of Neuron 0 in the 21th-gen Viridithas NNUE](images/logo.png)
  
  [![Build][build-badge]][build-link]
  [![License][license-badge]][license-link]
  [![Release][release-badge]][release-link]
  [![Commits][commits-badge]][commits-link]
  
</div>

Viridithas is a free and open source chess engine and the successor to the original [Viridithas](https://github.com/cosmobobak/viridithas-chess).

Viridithas is not a complete chess program and requires a UCI-compatible graphical user interface in order to be used comfortably. Read the documentation for your GUI of choice for information about how to use Viridithas with it.

For an overview of the features of Viridithas, see the [viri-wiki](wiki.md).

# Evaluation Development History/Originality (HCE/NNUE)
- First evaluation was a simple piece value / psqt table approach, using values from PeSTO. (as far as I remember)
- Early in development, a local-search texel tuning module was developed, and trained on the Lichess Elite dataset, removing the PeSTO values with increased strength.
- Many new evaluation terms were added during development, and were continually re-tuned on Viridithas's self-play games.
- The first NNUE was trained on a dataset of games played by Viridithas 2.7.0, 2.6.0, and 2.5.0, all rescored with a development version of Viridithas 2.7.0 at low depth.
- Subsequent networks were trained on additional self-play games by subsequent versions of Viridithas. The 13th-generation network, and many since, include positions from the Lichess Elite dataset rescored by Viridithas.

In summary, the originality of the data for training Viridithas's NNUE is strong - it is about ~60% games played by Viridithas against himself, 35% positions from human games, and something like ~5% games played in a Viridithas 2.7.0 vs. StockNemo 5.0.0.0 test match I ran on my own hardware. Most importantly, I have never trained on the evaluation output of any engine other than Viridithas.

# Thanks and Acknowledgements
[python-chess](https://github.com/niklasf/python-chess), without which I would never have been able to gain a love for chess programming.

The [VICE](https://www.youtube.com/playlist?list=PLZ1QII7yudbc-Ky058TEaOstZHVbT-2hg) video series, which was invaluable for explaining how various chess engine concepts can be implemented.

Andrew Grant's [Ethereal](https://github.com/AndyGrant/Ethereal), the exceedingly clear code of which helped me realise several horrible flaws in Viridithas.

[Shaheryar Sohail](https://github.com/TheBlackPlague), whose (very strong) StockNemo engine evolved alongside Viridithas, and who helped me significantly with NNUE.

[weather-factory](https://github.com/dsekercioglu/weather-factory), which I used only minimally, but which is still responsible for about ~10 elo in Viridithas.

[marlinflow](https://github.com/dsekercioglu/marlinflow), which is responsible for all neural network training for Viridithas.

[build-badge]:https://img.shields.io/github/actions/workflow/status/cosmobobak/virtue/rust.yml?branch=master&logo=github&style=for-the-badge
[build-link]:https://github.com/cosmobobak/virtue/actions/workflows/rust.yml
[commits-badge]:https://img.shields.io/github/commits-since/cosmobobak/virtue/latest?style=for-the-badge
[commits-link]:https://github.com/cosmobobak/virtue/commits/master
[release-badge]:https://img.shields.io/github/v/release/cosmobobak/virtue?style=for-the-badge&label=official%20release
[release-link]:https://github.com/cosmobobak/virtue/releases/latest
[license-badge]:https://img.shields.io/github/license/cosmobobak/virtue?style=for-the-badge&label=license&color=success
[license-link]:https://github.com/cosmobobak/virtue/blob/master/LICENSE
