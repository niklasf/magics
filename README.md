Finding magic factors for bitboard based move generation
========================================================

Tool to find factors for [Magic bitboards](https://chessprogramming.wikispaces.com/Magic+Bitboards).

Tricks for fast testing
-----------------------

1. Avoid zero filling table (like Stockfish).
2. Dynamically reorder relevant occupancies to fail earlier on average.

![Performance comparison](/fig-benchmark-rf8.png)

Usage
-----

Select `BISHOP` or `ROOK`, `SQUARE` and `EASY_SHIFT` in `Makefile`, `make` and run `./magics <chunk>`.
