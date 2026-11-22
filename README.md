# Luminex-NNUE

The training side of the [Luminex](https://github.com/changcheng967/Luminex) chess engine: the trainer that
makes its NNUE networks, the data tools that feed it, network forensics, and the development log.

The engine repository keeps what builds and ships the engine: the search, the evaluation, the network reader,
and the specification of every network format (its `docs/NNUE.md`). Networks trained here are written in the
formats those specifications define. `src/...` in comments here means the engine repository's `src/`.

## Layout

| Path | What it is |
| :--- | :--- |
| `trainer/resident_train.py` | The trainer, for one CUDA GPU: float training and quantization-aware training (QAT) on the engine's grid |
| `trainer/luminex_nnue_train.py` | The network's model and the training helpers the trainer imports |
| `trainer/resident_train_dh_rel.py` | The V5 trainer (relation lanes, the draw head): the float phase of the runs since G6, with the exact speed-ups of DEVLOG section 25 |
| `trainer/rel_gate.py`, `trainer/rel_kernels.py`, `trainer/rel_rows.py` | V5's relation inputs: the rows straight from the records, their sums and gradient, the tables' fused optimizer pass |
| `trainer/ft_fused.py` | Fused Triton kernels for the feature layer (`NNUE_FT_DENSE=3`) |
| `trainer/quantize_i8.py` | A float network to the int8 network the engine loads |
| `trainer/lnt8_from_gate.py` | A gen768t (LNT8) network from a V5 gate arm's files |
| `trainer/feeder.cpp`, `trainer/binfeed.cpp` | The trainer's C++ batch feeders, built next to it on first use |
| `trainer/binpack2lcr.cpp`, `trainer/lgs2lcr.cpp` | Training binpacks and Leela LGS shards to LCR1 records |
| `trainer/sf_binpack_dump.cpp` | A Stockfish-relabelled binpack to `fen<TAB>cp` text |
| `trainer/pfetch.py` | One file downloaded over several connections, resumably |
| `trainer/check_*.py` | Checks: the grid against the engine's, the fused kernels and the two feeders against their reference paths |
| `forensics/probe_net.py` | How much of a network's capacity real positions use |
| `bot/` | The lichess bot's start-up script and config template (the live token stays on the bot host) |
| `runs/` | The V5 runs' code byte for byte as the GPU boxes ran it (`g7/`, `tools/`) and their cells: a record, never edited (DEVLOG section 24 maps every file by hash) |
| `docs/DEVLOG.md` | The training log: every network generation, its runs and its measurements |
| `docs/HCE_TUNING_RESULTS.md` | The hand-crafted evaluation's tuning record, from before the networks |

The C++ data tools build against nnue-pytorch's binpack reader at commit `ed84cc1f`, the one `binpack2lcr` was
verified with. The box cells and CI fetch that commit.

## The grid check

QAT rounds the feature layer to the trainer's `FT_GRID`, which has to equal the engine's `NNUE_FT_SCALE`
(`src/nnue_kernels.h`). A network trained on another grid still loads and plays, but it is not the network QAT
trained. `trainer/check_grid.py` compares the two:

- CI runs it against Luminex main's header on every push and pull request here, and once a day, so a grid change
  on the engine side fails here by the next day at the latest.
- A QAT run checks it at the start, against `NNUE_ENGINE_HEADER` or a Luminex checkout beside this one.

## History

The files came from the engine repository with their history (git filter-repo). Commit messages from before the
move refer to that repository: its pull requests are written `changcheng967/Luminex#N`, and the commit hashes
they quote are its commits. Tools retired before the move are in its history.

## License

GPL-3.0, like the engine. See [LICENSE](LICENSE).
