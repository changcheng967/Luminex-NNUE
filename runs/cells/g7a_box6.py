# G7 (variant A, the user's choice: G6's recipe on FRESH Leela data, longer, a 2.4e9 reservoir): THE V5 RUN gen768t_03 (the V5 session, 2026-11-20 ~17:10 UTC, GPU box 6 sb-3e41f7e3a3a2c574, up since 16:51:37
# UTC; molab's 10 hours end it about 02:51 UTC). G6's recipe, longer, on the faster trainer:
# - start: gen768t_02 (G6's float phase, 1,142,724 steps from gen768t_01; its SPRT 176: +28.93 +- 9.30 over gen768t_01q)
#   loaded whole (model, draw head, relation table: ckpt/luminex_gen768t_02_model.pt + _rel.pt, joined here into a start
#   file with no optimizer, step 0), a fresh cosine from lr 5e-5 on G6's 17 binpacks, seed 11; the length fitted at the
#   training start to end at 01:25 UTC (94 steps/s). VARIANT A: t91's 11 months as G6 + 12 Leela BT4-tf13tune corpus files
#   G6 never used (107.8 GB: test80-2024 Feb/Mar/May/Jun histfix, test80-2023 Sep/Dec, test80-2022 Aug, test79-2022 Apr/May,
#   test60-2021 Nov/Dec, leela96 split_1; Leela only, the user's rule) in place of G6's six, 134.6 GB in all (the user's
#   cap 150); NNUE_RESERVOIR 2.4e9 (G6: 1e9): 77 GB + the trainer's ~7.6 GB (the smoke test's peak) = ~84 of 96 GB;
# - the float phase on the speed-up proven on box 5 (notes speed/README.md): /marimo/g7/resident_train_dh_rel.py = G6's
#   trainer + patch_fastrows (the one-pass feature-layer glue) + patch_drawmerge (NNUE_DRAW_MERGE=1), prof14's build byte
#   for byte (rebuilt from G6's files and compared); g7/fast/ft_fused.py = G6's + the glue (prof10's); g7/v5 = notes
#   speed/v5 (NNUE_REL_IMPL=trirows, the int16 sort); NNUE_REL_TABLE16=1. Exact, or proven loss-equal (prof14);
# - then, in this cell, G5's QAT unchanged on G6's tools (tools/, NNUE_REL_IMPL=triseg), and G6's conversion (quantize_i8.py
#   --rows -> LNR8; lnt8_from_qat.py twice, byte-identical -> LNT8) and out/SHA256SUMS_03.
# Log /marimo/luminex_gen768t_03.log. The data came from the same cell with DATA_ONLY = True (17:00 UTC).
# (G6's cell, from here on:)
# THE V5 FULL RUN gen768t_02 (main session, 2026-11-19 ~20:20 UTC, box 1's notebook sb-ee6dc6a2f4fb5f36): the V5 session's
# recipe. gen768t_01 at 450K loaded whole (model, draw head, relation table: ckpt/luminex_gen768t_01_model.pt +
# _rel.pt, joined here into a start file with no optimizer, step 0), a fresh cosine from lr 5e-5 on G5's 17 binpacks
# (11 t91 BT4 months + 6 BT4-tf13tune corpus files), seed 7. The cosine's length is fitted at the training start to
# end 50 min before 02:05 UTC (the box closes with the user's laptop, ~6 h after it started), capped at the recipe's
# 1.6M; never cut once running. Saves every 50K, .nnue copies every 400K. The QAT is the next cell (G5's recipe).
# (G5's V5 cell, from here on:)
# THE V5 FULL PAIR (G5), V5 ARM (gen768t_01: A256, the full relation set, pawn pairs in contact), box 1, 2026-11-18 evening UTC (main session): run 4's recipe (data, length,
# schedule, QAT, heads) on the trainer the pair shares: trainer-stack d0ac5af (PR #105) + fused-rel-lanes (PR #104,
# 4e76774), QAT on the engine's x2048 grid (FT_GRID, checked against src/nnue_kernels.h). Start: gen768s_03q's float
# file (2df94155, the V5 gate's start), a fresh draw head and optimizer, as the control. The relation rows: 55,960 into
# lanes 0..255 (rel_gate.py + rel_kernels.py + patch_rel.py, sums by triseg, lanes inside the fused kernels), lr x4,
# clipped to +-127/128 after every step. FLOAT PHASE ONLY: its QAT (int8 rows, lane clamp, LNT8) comes with that code,
# from the model saved at the end (out/luminex_gen768t_01_model.pt and _rel.pt, small enough to outlive the machine).
# (run 4's own cell, from here on:) V5 run 4 (data, long, from run 3): gen768s_04, ~6 passes over more and more diverse data from gen768s_03 (its
# value and draw head, a fresh optimizer) continued on every t91 BT4 month plus ~91 GB of the public BT4-tf13tune
# corpus (Leela test78, test80 2023 and 2024, leela96; Stockfish's wrongIsRight self-play), all held as binpacks in
# host RAM by the binpack feeder (PR #85 at 786821a); the draw head in the loss; then QAT with one int8 scale per
# tail row, then the error and draw heads. Every corpus file passes the material check on a sample before training.
# 1.6M more steps after run 3's 450K, on PR #88's fused activation. SPRT [0, 5] of gen768s_04q against gen768s_03q (LNR8,
# main + PR #89): data and length change together. Made by make_v5r4e.py from make_v5r4d.py's cell.
import collections
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import marimo as mo

REFRESH_SECONDS = 5   # how often the panel redraws
SHOW_LINES = 20000    # log lines kept in the panel (the log file has every line)
SHOW_STEP_EVERY = 1000   # the trainer logs every 50 steps; the panel and console show one line per 1000
DATA_ONLY = False     # G7: the data is on the box already (the data cell); True: prepare the data and stop; False: prepare (or reuse) the data, then train
SMOKE = False         # True: a 1,500-step test of every phase (into out/smoke/), then stop
MONTHS = "2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06 2026-07 2026-08 2026-10"   # every t91 BT4 month (8.56B usable): the feeder has no record cap; January's tail held out
CONNECTIONS = 8      # parallel download connections: one gave about 7 MB/s, eight 21-25 MB/s a file (two files download at once)
# The BT4-tf13tune corpus files (dataset/path on Hugging Face), the same net's labels as t91's binpacks:
EXTRA = " ".join((   # G7 variant A, Leela only (the user): BT4-tf13tune files G6 never used (G6's six are set aside)
    "xushawn/test80-bt4-relabel/test80-2024-02-feb-2tb7p.min-v2.v6.relabel-BT4-tf13tune-histfix.binpack",   # 8.3 GB
    "xushawn/test80-bt4-relabel/test80-2024-03-mar-2tb7p.min-v2.v6.relabel-BT4-tf13tune-histfix.binpack",   # 10.6 GB
    "xushawn/test80-bt4-relabel/test80-2024-05-may-2tb7p.min-v2.v6.relabel-BT4-tf13tune-histfix.binpack",   # 10.6 GB
    "xushawn/test80-bt4-relabel/test80-2024-06-jun-2tb7p.min-v2.v6.relabel-BT4-tf13tune-histfix.binpack",   # 9.5 GB
    "vondele/linrock_relabel_2/test80-2023-09-sep-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack",           # 8.6 GB
    "vondele/linrock_relabel_2/test80-2023-12-dec-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack",           # 8.0 GB
    "vondele/linrock_relabel_1/test80-2022-08-aug-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack",          # 10.9 GB
    "vondele/linrock_relabel_1/test79-2022-04-apr-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack",          # 8.5 GB
    "vondele/linrock_relabel_1/test60-2021-11-nov-12tb7p.min-v2.relabel-BT4-tf13tune.binpack",             # 3.4 GB
    "vondele/linrock_relabel_1/test60-2021-12-dec-12tb7p.min-v2.relabel-BT4-tf13tune.binpack",             # 3.2 GB
    "vondele/from_kaggle_1_relabel/leela96-filt-v2.min.split_1.relabel-BT4-tf13tune.binpack",              # 19.5 GB
    "vondele/linrock_relabel_1/test79-2022-05-may-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack",          # 6.5 GB
))
SAMPLE_BYTES = 1_500_000_000   # the head of each corpus file that the material check reads

HOME = "/marimo"
RUN = "luminex_gen768t_03"   # G7: V5's run after G6; was luminex_gen768t_02, the V5 full run (was G5's arm luminex_gen768t_01); G5's V5 arm; was gen768s_04: gen768s_02 continued on t91 + the BT4 corpus; QAT (per-row tail scales) writes RUN + "q", the heads RUN + "q_heads.pt"
STEPS = 3000000       # G7: a cap only, the cell fits the cosine to end at 01:25 UTC; was the recipe's 1.6M cap: the cell fits the cosine to the window at the training start; was G5 tonight: section 5's pair length (450K float + 30K QAT), to finish before 22:30 Toronto; was ~210B samples: ~5.9 passes over ~35.6B positions, after run 3's 450K steps on t91; the budget guard cuts it only if it cannot fit
HOURS = (1795228800 - time.time()) / 3600   # G7: the cell's budget ends at 02:40 UTC 2026-11-21 (box 6 ends ~02:51); was 6.0, the cell's budget guard, wider than the window so the fitted cosine is never cut; was G5 tonight: from ~00:08 to at most ~03:08 UTC, before the user's 22:30 Toronto (03:30 UTC); was the whole cell, to the user's 22:30 EST limit with a margin: data (on disk: ~2 min), the float phase, one QAT phase (~30 min), the heads
LOSS_POWER = 3.0      # NNUE_LOSS_POWER: 3.0 beat 2.6 by +2.95 (test 67)
KEEP_EVERY = 400000   # the recipe: a copy every 400K; was 100000; steps between numbered copies of the net (out/<RUN>_s<step>.nnue)
LR = 5e-5             # peak learning rate of the new cosine (NNUE_LR), as run05; the tail trains at 4x
COMMIT = "4e76774debf540906f8c80f004caa670666e0551"   # fused-rel-lanes (#104) on trainer-stack (#105); was branch fused-act (PR #88) head: #88's NNUE_FT_ACT + #86 + #85's feeder and stall fix
VRAM_POS = 2_300_000_000   # positions on the card; the rest in host RAM
QAT_STEPS = 30000     # then QAT fine-tunes from the float phase's final checkpoint (as 03q, 04q, 05q)
QAT_LR = 2e-5         # their peak learning rate
QAT_FT8 = 0           # no int8-grid QAT phase: test 137 settled the 16-bit table
DRAW_W = 1e-3         # weight of the draw head's loss in the float phase (the gate: 3e-4 and 3e-3 both left the value unchanged)
QAT_RESERVE = 2700    # G7: 45 min for this cell's QAT and conversion after the float phase; was 600: G5's V5 arm: no QAT tonight, 600 s to save the model in small files; was seconds kept for the QAT phase (reload of the binpacks into RAM, compile, 30K steps, export) and the heads
JAN_TAIL_MB = 32      # January's last whole chunks, at least this many MiB, held out: the BT4-labelled game-disjoint check (it covers base's jantail games)
NP_COMMIT = "ed84cc1fd2255faa0f33dcfa7839c9b6b4bb49b8"  # nnue-pytorch: the binpack reader binpack2lcr was verified with

# Each phase checks what it needs and stops with FAIL on anything wrong; finished
# months are skipped when the cell runs again. No heredocs in SCRIPT, and no
# indentation that matters: the notebook file re-indents the cell.
SCRIPT = r"""
set -euo pipefail
HOME_DIR="$1"; RUN="$2"; STEPS="$3"; TOTAL="$4"; COMMIT="$5"; DATA_ONLY="$6"; PY="$7"; MONTHS="$8"; NP_COMMIT="$9"; CONN="${10}"; SMOKE="${11}"; LOSS_POW="${12}"; KEEP_N="${13}"; LR_V="${14}"; VRAM_POS="${15}"; QAT_STEPS="${16}"; QAT_LR="${17}"; QAT_RES="${18}"; QAT_FT8="${19}"; JAN_TAIL="${20}"; DRAW_W="${21}"; EXTRA="${22}"; SAMPLE_B="${23}"
cd "$HOME_DIR"
T_CELL=$(date +%s)
GH=https://raw.githubusercontent.com/changcheng967/Luminex/$COMMIT/nnue/openi_upload
GH8=https://raw.githubusercontent.com/changcheng967/Luminex/e2bac968298160870165677d48cf075bcbc61ace/nnue/openi_upload
S02="$HOME_DIR/out/luminex_gen768s_03.pt"   # run 3's float checkpoint on this box: the value and the draw head
L1=768
DATA="$HOME_DIR/data"; TOOLS="$HOME_DIR/tools"
mkdir -p "$DATA" "$TOOLS"
sha() { sha256sum "$1" | cut -d' ' -f1; }
ok() { [ -f "$1" ] && [ "$(sha "$1")" = "$2" ]; }

echo "== PHASE 1/4: environment check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "FAIL: no NVIDIA GPU visible"; exit 1; }
echo "  CPU threads: $(nproc), RAM: $(free -g | awk '/^Mem/ {print $2}') GB, $(g++ --version | head -n 1)"
FREE_GB=$(df -BG --output=avail "$HOME_DIR" | tail -n 1 | tr -dc 0-9)
echo "  disk free in $HOME_DIR: ${FREE_GB} GB (the run's binpacks take ~118 GB)"
for t in curl g++ sha256sum tar; do command -v $t > /dev/null || { echo "FAIL: $t is missing (install it in the terminal)"; exit 1; }; done
"$PY" -c 'import sys, numpy, torch; print("  python", sys.version.split()[0], "| numpy", numpy.__version__, "| torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU:", torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)' || { echo "FAIL: PyTorch cannot see the GPU"; exit 1; }
if command -v zstd > /dev/null; then ZST=cli; echo "  zstd: command line"
elif "$PY" -c 'import zstandard' 2> /dev/null; then ZST=py; echo "  zstd: python zstandard"
else ZST=0; echo "  zstd: none (the base binpacks are downloaded plain either way)"; fi
echo "  environment: ok"

echo "== PHASE 2/4: code, tools and the checkpoint (Luminex ${COMMIT:0:7}, nnue-pytorch ${NP_COMMIT:0:7})"
for pair in "resident_train.py 210e59727d6c166c88f3b6a338e9d84d0a88879340f591c50e0ebeb8b34b4ae0" "binfeed.cpp 6b13e1644706405e43e18484f42f5b4d2fa3754fc6466e7a150adea1400644c7" "ft_fused.py 638f81988214a9460e2b6736c9ed2ea597f6d04144ff65601b2b6f2652561db8" "check_fused.py e5807730bdffd29393115292981a4ea07db69988e3c12af766e223b8f0f083e4" "feeder.cpp e01bbbf99c8bd712b3fe28cdb90287e91f310b3eba8ebd922ece92aabfed8779" "check_feeder.py ece75d313cd902b8b5800215f1f99873f67c85f7bc516a53bfd606fd87e26805" "luminex_nnue_train.py ea5b94f2026c82b7a111d54d75fcf87fc30a8986290332b3dd446bb64f1222a8" "binpack2lcr.cpp 28f7ec0d009bd401429b3db85f014abebfdeb6997457af5e015e40836cfe36ab" "quantize_i8.py 52811a38aa1631f80d0f16b875ec93d9117c73766ab479f08249cb4621b40b9d" "pfetch.py cf24acc2e7482c8afb2862528e2aed6a1b7e4ce8e0e5f7e244e3e98e117a0ccb" "check_grid.py 09dc24898745937a404b7ff5b26d6e7dac9ee921e6fe17cd626be8cb3f2cd097"; do
    set -- $pair
    ok "$TOOLS/$1" "$2" || curl -fsSL --retry 5 -o "$TOOLS/$1" "$GH/$1" || curl -fsSL --retry 5 -o "$TOOLS/$1" "https://gh-proxy.com/$GH/$1" || true
    ok "$TOOLS/$1" "$2" || { echo "FAIL: $1 from GitHub does not match sha256 $2"; exit 1; }
    echo "  $1: sha256 ok"
done
curl -fsSL --retry 5 -o "$TOOLS/nnue_kernels.h" "https://raw.githubusercontent.com/changcheng967/Luminex/$COMMIT/src/nnue_kernels.h" || { echo "FAIL: the engine header (src/nnue_kernels.h)"; exit 1; }
grep -q "NNUE_FT_SCALE = 2048.0f" "$TOOLS/nnue_kernels.h" || { echo "FAIL: the engine header is not on the x2048 grid"; exit 1; }
echo "  nnue_kernels.h: NNUE_FT_SCALE = 2048"
Q8=627687a0da5e8baa3560885bcde76bce2dd5c1fa12c806d6003d76c819348c46   # quantize_i8.py of branch nnue-ft-int8 (e2bac96): --ft8 / --ft-grid
ok "$TOOLS/quantize_ft8.py" "$Q8" || curl -fsSL --retry 5 -o "$TOOLS/quantize_ft8.py" "$GH8/quantize_i8.py" || curl -fsSL --retry 5 -o "$TOOLS/quantize_ft8.py" "https://gh-proxy.com/$GH8/quantize_i8.py" || true
ok "$TOOLS/quantize_ft8.py" "$Q8" || { echo "FAIL: quantize_ft8.py (nnue-ft-int8's quantize_i8.py) does not match sha256 $Q8"; exit 1; }
echo "  quantize_ft8.py (the LNF8 exporter): sha256 ok"
if [ ! -f "$TOOLS/nnue-pytorch/data_loader/cpp/lib/binpack.h" ]; then
    mkdir -p "$TOOLS/nnue-pytorch"
    curl -fsSL --retry 5 "https://codeload.github.com/official-stockfish/nnue-pytorch/tar.gz/$NP_COMMIT" | tar xz -C "$TOOLS/nnue-pytorch" --strip-components=1
fi
[ -f "$TOOLS/nnue-pytorch/data_loader/cpp/lib/binpack.h" ] || { echo "FAIL: nnue-pytorch headers missing"; exit 1; }
if [ ! -x "$TOOLS/binpack2lcr" ] || [ "$TOOLS/binpack2lcr.cpp" -nt "$TOOLS/binpack2lcr" ]; then
    g++ -O2 -march=native -std=c++20 -pthread -I "$TOOLS/nnue-pytorch/data_loader/cpp" -o "$TOOLS/binpack2lcr" "$TOOLS/binpack2lcr.cpp" || { echo "FAIL: binpack2lcr build"; exit 1; }
fi
echo "  binpack2lcr: built"
"$PY" "$TOOLS/pfetch.py" --selftest || { echo "FAIL: pfetch self-test (see above)"; exit 1; }
START="$HOME_DIR/ckpt/luminex_gen768t_02_start.pt"   # G7: gen768t_02 (G6's float phase) whole, no optimizer, step 0
if [ "$DATA_ONLY" != 1 ]; then
    [ "$(sha "$HOME_DIR/ckpt/luminex_gen768t_02_model.pt" | cut -c1-16)" = "e1381c69d1685a35" ] || { echo "FAIL: ckpt/luminex_gen768t_02_model.pt is not G6's (sha256 e1381c69...)"; exit 1; }
    [ "$(sha "$HOME_DIR/ckpt/luminex_gen768t_02_rel.pt" | cut -c1-16)" = "4fb7680089744958" ] || { echo "FAIL: ckpt/luminex_gen768t_02_rel.pt is not G6's (sha256 4fb76800...)"; exit 1; }
    "$PY" -c 'import sys, torch; m = torch.load(sys.argv[1], map_location="cpu", weights_only=False); r = torch.load(sys.argv[2], map_location="cpu", weights_only=False); sd = dict(m["model"]); sd.update(r); assert "rel" in sd and tuple(sd["rel"].shape) == (55960, 256), "no relation table"; torch.save({"model": sd, "gstep": 0}, sys.argv[3]); print("  start:", sys.argv[3], "| gen768t_02 at step", m.get("gstep"), "|", len(sd), "tensors, rel", tuple(sd["rel"].shape), "| no optimizer, step 0")' "$HOME_DIR/ckpt/luminex_gen768t_02_model.pt" "$HOME_DIR/ckpt/luminex_gen768t_02_rel.pt" "$START" || { echo "FAIL: building the start file"; exit 1; }
fi
"$PY" "$TOOLS/patch_draw.py" "$TOOLS/resident_train.py" "$TOOLS/resident_train_dh.py" || { echo "FAIL: the draw-head patch"; exit 1; }
for pair in "rel_gate.py 4a7076fc7a789fd5c814e16735f56addd209f10fcb3cd195e2015fe9292d35fc" "rel_kernels.py ccff545cde723667f78009c35d5494a3531754a4366dfabfb26d7c8625f7e857" "patch_rel.py 904db2f83527b4e5afa39c55d3db9539c8c922647f1d1eddefa0055e76b4109b"; do
    set -- $pair
    ok "$TOOLS/$1" "$2" || { echo "FAIL: $TOOLS/$1 is not the relation code this run pins (sha256 $2): upload it"; exit 1; }
    echo "  $1: sha256 ok"
done
"$PY" "$TOOLS/patch_rel.py" "$TOOLS/resident_train_dh.py" "$TOOLS/resident_train_dh_rel.py" || { echo "FAIL: the relation patch"; exit 1; }
G7="$HOME_DIR/g7"   # G7's float-phase code (uploaded): prof14's trainer, prof10's ft_fused.py, notes speed/v5
for pair in "resident_train_dh_rel.py d8088c2b74640a8ef210f9b82fb23b42bbac4addb0f024ecc154a091c65377b2" "fast/ft_fused.py 568a2483dc9b1f7b879e63052571fdd4489f4a13032d6b202ea949d7557554d8" "v5/rel_gate.py a83ec76dcefa852e7a9af80aca7bd4d3c9105353272354ca140b5ab83a5e3125" "v5/rel_kernels.py 19a1f736a7d2df076f6f717de84297face199a2f8ddb2e566e22c8841a68d1d7" "v5/rel_rows.py dfe8b37988cc2e0f3cef92c930e8568c5c139e1a5112f4d189ee1c7e47633c5d"; do
    set -- $pair
    ok "$G7/$1" "$2" || { echo "FAIL: $G7/$1 is not G7's pinned code (sha256 $2): upload it"; exit 1; }
    echo "  g7/$1: sha256 ok"
done
# the trainer builds its feeders from its own folder (HERE): G6's pinned sources beside it, as prof14 had them
cp "$TOOLS/binfeed.cpp" "$TOOLS/feeder.cpp" "$G7/" || { echo "FAIL: copying the feeders beside G7's trainer"; exit 1; }
for pair in "binfeed.cpp 6b13e1644706405e43e18484f42f5b4d2fa3754fc6466e7a150adea1400644c7" "feeder.cpp e01bbbf99c8bd712b3fe28cdb90287e91f310b3eba8ebd922ece92aabfed8779"; do
    set -- $pair
    ok "$G7/$1" "$2" || { echo "FAIL: $G7/$1 is not G6's (sha256 $2)"; exit 1; }
    echo "  g7/$1: sha256 ok (G6's, beside the trainer)"
done
for pair in "make_qatstart_v5.py e78b6a75cb11386276e60348d196e2e316208edc600022f2f65e2f2e3b75b6ec" "lnt8_from_qat.py 2d29301f331d6e1e64f9e6630f8545a714084a45d3733cf4404ae0ebd680e1d9" "quantize_i8.py 52811a38aa1631f80d0f16b875ec93d9117c73766ab479f08249cb4621b40b9d"; do
    set -- $pair
    ok "$TOOLS/$1" "$2" || { echo "FAIL: $TOOLS/$1 is not the QAT and conversion code G6 ran (sha256 $2): upload it"; exit 1; }
    echo "  $1: sha256 ok"
done
"$PY" "$TOOLS/patch_rows.py" "$TOOLS/resident_train.py" "$TOOLS/resident_train_qr.py" || { echo "FAIL: the per-row patch"; exit 1; }

echo "== PHASE 3/4: data: t91's BT4 months ($MONTHS) and the BT4 corpus, kept as binpacks for the feeder (NNUE_BINPACKS)"
T_DATA=$(date +%s)
BP="$DATA/bp"; CHK="$DATA/chk"; mkdir -p "$BP" "$CHK"
rm -f "$DATA"/t91_*.lcr "$DATA"/t91_*.done "$DATA"/t91_*.log   # the record path's months (runs 1 to 3): this run reads binpacks
meta() {   # DATASET PATH -> "size sha256" (Hugging Face paths-info)
    "$PY" -c "import json, sys, urllib.parse, urllib.request; ds, p = sys.argv[1], sys.argv[2]; e = json.load(urllib.request.urlopen(urllib.request.Request('https://huggingface.co/api/datasets/' + ds + '/paths-info/main', data=urllib.parse.urlencode([('paths', p), ('expand', 'true')]).encode()), timeout=60))[0]; print(e['size'], e['lfs']['oid'])" "$1" "$2"
}
get() {   # DATASET PATH DEST: download (resumable, CONN connections), check the sha256, then mark DEST.ok
    local ds="$1" p="$2" dest="$3" m size oid t0 t1
    if [ -f "$dest.ok" ]; then echo "  $(basename "$dest"): already here ($(cat "$dest.ok"))"; return 0; fi
    m=$(meta "$ds" "$p") || { echo "FAIL: no Hugging Face metadata for $ds/$p"; return 1; }
    read -r size oid <<< "$m"
    [ -n "${oid:-}" ] || { echo "FAIL: no size and sha256 for $ds/$p"; return 1; }
    t0=$(date +%s)
    if [ "$(stat -c %s "$dest" 2> /dev/null || echo 0)" != "$size" ]; then
        "$PY" "$TOOLS/pfetch.py" "https://huggingface.co/datasets/$ds/resolve/main/$p" "$dest" "$size" "$CONN" > "$dest.fetch.log" 2>&1 || { tail -n 5 "$dest.fetch.log"; echo "FAIL: download $ds/$p"; return 1; }
        rm -f "$dest.pieces"
    fi
    t1=$(date +%s)
    [ "$(sha "$dest")" = "$oid" ] || { rm -f "$dest"; echo "FAIL: $ds/$p differs from Hugging Face's sha256 (deleted it: run the cell again)"; return 1; }
    echo "$size bytes, sha256 ${oid:0:16}, $ds" > "$dest.ok"
    echo "  $(basename "$dest"): $((size / 1000000)) MB in $((t1 - t0)) s ($((size / 1000000 / (t1 - t0 + 1))) MB/s), sha256 ok"
}
JOBS=()
for ym in $MONTHS; do
    case ${ym#*-} in 01) mn=jan;; 02) mn=feb;; 03) mn=mar;; 04) mn=apr;; 05) mn=may;; 06) mn=jun;; 07) mn=jul;; 08) mn=aug;; 09) mn=sep;; 10) mn=oct;; 11) mn=nov;; 12) mn=dec;; *) echo "FAIL: unknown month $ym"; exit 1;; esac
    JOBS+=("xushawn/t91-lgs|binpacks/v6-bt4/test91-$ym-$mn-150gb6p.v6.relabel-BT4-tf13tune-histfix.binpack|$BP/t91_$ym.binpack")
done
for e in $EXTRA; do
    ds=$(echo "$e" | cut -d/ -f1-2); p=$(echo "$e" | cut -d/ -f3-)
    JOBS+=("$ds|$p|$BP/x_$(basename "$p")")
done
ASIDE="$DATA/bp_aside"; mkdir -p "$ASIDE"; KEEP=" "   # G7: only this run's mix stays in $BP, which the feeder reads whole
for j in "${JOBS[@]}"; do
    IFS='|' read -r ds p dest <<< "$j"; b=$(basename "$dest"); KEEP="$KEEP$b "
    if [ ! -f "$dest" ] && [ -f "$ASIDE/$b" ]; then for x in "$ASIDE/$b" "$ASIDE/$b".*; do [ -e "$x" ] && mv "$x" "$BP/"; done; echo "  back from $ASIDE: $b"; fi
done
FAILS=0; RUNNING=()
for j in "${JOBS[@]}"; do
    IFS='|' read -r ds p dest <<< "$j"
    get "$ds" "$p" "$dest" &
    RUNNING+=("$!")
    if [ "${#RUNNING[@]}" -ge 2 ]; then
        wait "${RUNNING[0]}" || FAILS=$((FAILS + 1))
        RUNNING=("${RUNNING[@]:1}")
    fi
done
[ "${#RUNNING[@]}" = 0 ] || for pid in "${RUNNING[@]}"; do wait "$pid" || FAILS=$((FAILS + 1)); done
[ "$FAILS" = 0 ] || { echo "FAIL: $FAILS download(s) failed (see above; the cell resumes them when it runs again)"; exit 1; }
for f in "$BP"/*.binpack; do
    b=$(basename "$f")
    case "$KEEP" in *" $b "*) ;; *) for x in "$f" "$f".*; do [ -e "$x" ] && mv "$x" "$ASIDE/"; done; echo "  set aside in $ASIDE (not in this run's mix): $b";; esac
done
J="$BP/t91_2026-01.binpack"   # the game-disjoint check: January's last chunks, never trained on (cut once; the marker keeps it)
if [ ! -f "$J.split" ]; then
    rm -f "$DATA"/jantail.*
    "$PY" "$TOOLS/binpack_split.py" "$J" "$JAN_TAIL" "$DATA/jantail.binpack" || { echo "FAIL: split January"; exit 1; }
    echo "$(stat -c %s "$J") bytes kept to train on" > "$J.split"
fi
if [ ! -f "$DATA/jantail.done" ]; then
    [ -f "$DATA/jantail.binpack" ] || { echo "FAIL: January's tail binpack is missing (delete $J, $J.ok and $J.split, then run the cell again)"; exit 1; }
    "$TOOLS/binpack2lcr" "$DATA/jantail.binpack" "$DATA/jantail" "$(nproc)" 2> "$DATA/jantail.log" || { cat "$DATA/jantail.log"; echo "FAIL: convert January's held-out tail"; exit 1; }
    rm -f "$DATA/jantail.binpack"
    tail -n 1 "$DATA/jantail.log" > "$DATA/jantail.done"
fi
echo "  jan tail (held out, never trained on): $(cat "$DATA/jantail.done")"
for f in "$BP"/x_*.binpack; do   # the material check reads the head of every corpus file
    n=$(basename "$f" .binpack)
    [ -f "$CHK/$n.done" ] && continue
    rm -f "$CHK/$n".*
    "$PY" "$TOOLS/binpack_head.py" "$f" "$SAMPLE_B" "$CHK/$n.binpack" || { echo "FAIL: the sample of $n"; exit 1; }
    "$TOOLS/binpack2lcr" "$CHK/$n.binpack" "$CHK/$n" "$(nproc)" 2> "$CHK/$n.log" || { cat "$CHK/$n.log"; echo "FAIL: convert the sample of $n"; exit 1; }
    echo "$(stat -c %s "$CHK/$n.binpack") $(stat -c %s "$f") $(tail -n 1 "$CHK/$n.log")" > "$CHK/$n.done"
    rm -f "$CHK/$n.binpack"
done
"$PY" "$TOOLS/check_sources.py" "$CHK" || { echo "FAIL: a corpus file does not value material (see above)"; exit 1; }
echo "  data phase: $(( $(date +%s) - T_DATA )) s; $(du -sh "$DATA" | cut -f1) in $DATA"
if [ "$DATA_ONLY" = 1 ]; then echo "== DATA ONLY: done; set DATA_ONLY = False to train on it"; exit 0; fi

# The float phase trains on the fused feature layer (NNUE_FT_DENSE=3, run05's); its batches come from the
# binpack feeder (binfeed.cpp into a VRAM reservoir), so feeder.cpp's host-RAM batches are not used.
FT=3; FEED=8

if [ "$SMOKE" = 1 ]; then   # every part of a full run (load, compile, checks, saves, step copies, both QATs) in a few minutes
    STEPS=1500; EVERY=500; KEEP=500; BUDGET=0; OUTD="$HOME_DIR/out/smoke"; QAT_STEPS=300; QEVERY=300
    echo "== PHASE 4/4: SMOKE TEST of the training phases ($STEPS steps into $OUTD)"
else
    BUDGET=$(( TOTAL - ($(date +%s) - T_CELL) - 300 - QAT_RES ))   # 300 s for the final save and export, QAT_RES for the QAT phases
    QEVERY=5000
    [ "$BUDGET" -ge 1800 ] || { echo "FAIL: only $BUDGET s of the cell's $TOTAL s are left for training"; exit 1; }
    EVERY=50000; KEEP="$KEEP_N"; OUTD="$HOME_DIR/out"   # G5: evaluations and saves every 50K steps in both arms (run 4: 20K)
    echo "== PHASE 4/4: training $RUN ($STEPS steps, budget $BUDGET s of the cell's $TOTAL s; then QAT, $QAT_STEPS steps, as the engine runs it$( [ "$QAT_FT8" = 0 ] || echo " and again with the feature layer on the int8 grid (shift $QAT_FT8)"))"
fi
mkdir -p "$OUTD"
NEED=0; NF=0
for f in "$BP"/*.binpack; do NEED=$(( NEED + $(stat -c %s "$f") )); NF=$((NF + 1)); done
MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
echo "  binpacks for the feeder: $NF files, $((NEED / 1000000000)) GB, held in host RAM ($((MEM_KB / 1048576)) GiB in all; 35 GB kept free)"
[ $(( MEM_KB * 1024 - NEED )) -ge 35000000000 ] || { echo "FAIL: the binpacks ($((NEED / 1000000000)) GB) and 35 GB of headroom do not fit $((MEM_KB / 1048576)) GiB of host RAM"; exit 1; }
[ -n "$START" ] || { echo "FAIL: no checkpoint to start from"; exit 1; }
[ -f "$DATA/tree.000.lcr" ] || { echo "FAIL: no $DATA/tree.000.lcr (the tree held-out set)"; exit 1; }
END_UTC=1795224300   # G7: the float phase ends by 01:25 UTC 2026-11-21 (then QAT, conversion and copies; box 6 ends ~02:51)
FIT=$(( (END_UTC - $(date +%s)) * 94 ))   # G7: 94 steps/s (box 6's smoke test: 96 on all 17 binpacks; box 5: 93.5; G6 fitted 66 and ran 66.1)
if [ "$SMOKE" != 1 ] && [ "$FIT" -lt "$STEPS" ]; then STEPS=$FIT; fi
[ "$STEPS" -ge 200000 ] || [ "$SMOKE" = 1 ] || { echo "FAIL: only $STEPS steps fit before 01:25 UTC"; exit 1; }
echo "  the cosine fitted to the window: $STEPS steps, ending about $(date -u -d @$(( $(date +%s) + STEPS / 94 )) +%H:%M) UTC at 94 steps/s (then QAT and the conversion in this cell)"
echo "  starting from $START: a cosine of $STEPS steps from lr $LR_V on BT4's labels (NNUE_CONTINUE=0)"
env PYTHONUNBUFFERED=1 PYTHONPATH="$G7/v5:$G7/fast:$TOOLS" NNUE_PYTORCH="$TOOLS/nnue-pytorch" NNUE_BINPACKS="$BP/*.binpack" NNUE_VAL_DATA="$DATA/jantail.*.lcr" NNUE_VAL_TREE="$DATA/tree.000.lcr" \
    NNUE_RESUME="$START" NNUE_CONTINUE=0 NNUE_FEED_THREADS="$FEED" \
    NNUE_BUCKETS=8 NNUE_TAIL_LR_MULT=4 NNUE_L1="$L1" NNUE_LR="$LR_V" NNUE_BS=131072 \
    NNUE_T_MAX_STEPS="$STEPS" NNUE_BUDGET_SEC="$BUDGET" NNUE_LOSS_POWER="$LOSS_POW" NNUE_QUIET_LOSS=1,0.3,0 \
    NNUE_GRAD_CLIP=1.0 NNUE_TARGET_CLIP=1500 NNUE_VAL_FRAC=0.002 \
    NNUE_EVAL_EVERY="$EVERY" NNUE_SAVE_EVERY="$EVERY" NNUE_KEEP_EVERY="$KEEP" \
    NNUE_COMPILE=1 NNUE_QAT=0 NNUE_FT_DENSE="$FT" NNUE_FT_ACT=1 NNUE_VRAM_POS="$VRAM_POS" NNUE_RESERVOIR=2400000000 NNUE_SEED=11 NNUE_OUT_DIR="$OUTD" NNUE_OUT_NAME="$RUN.nnue" \
    NNUE_REL_T=256 NNUE_REL_SET=full NNUE_REL_PPM=0 NNUE_REL_LR_MULT=4 NNUE_REL_IMPL=trirows NNUE_REL_TABLE16=1 NNUE_REL_FUSED=1 NNUE_REL_JOIN=inplace \
    NNUE_DRAW_MERGE=1 NNUE_DRAW_W="$DRAW_W" "$PY" -u "$G7/resident_train_dh_rel.py"
echo "== float phase done: $OUTD/$RUN.pt (model, relation table and optimizer); $OUTD/$RUN.nnue holds no relation rows"
"$PY" -c 'import sys, torch; ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); sd = ck["model"]; rel = {k: v for k, v in sd.items() if k.startswith("rel")}; rest = {k: v for k, v in sd.items() if not k.startswith("rel")}; assert rel, "no relation table in the checkpoint"; torch.save({"model": rest, "gstep": ck.get("gstep", 0)}, sys.argv[2]); torch.save(rel, sys.argv[3]); print("  saved", sys.argv[2], "and", sys.argv[3], {k: tuple(v.shape) for k, v in rel.items()}, "step", ck.get("gstep"))' "$OUTD/$RUN.pt" "$OUTD/${RUN}_model.pt" "$OUTD/${RUN}_rel.pt" || { echo "FAIL: saving the model in small files"; exit 1; }
ls -la "$OUTD/${RUN}_model.pt" "$OUTD/${RUN}_rel.pt"
echo "== FINISHED (V5 arm, float phase): $OUTD/${RUN}_model.pt + ${RUN}_rel.pt; QAT and the heads come with the relation QAT code"
# G7: G5's QAT unchanged (qat_v5_box1.py), on G6's tools (tools/: rel_gate.py 4a7076fc, NNUE_REL_IMPL=triseg)
Q="${RUN}q"
echo "== QAT: $Q, $QAT_STEPS steps from $OUTD/$RUN.pt (lr 2e-5, per-row tail scales, seed 4, int8 relation rows, lane clamp)"
"$PY" "$TOOLS/make_qatstart_v5.py" "$OUTD/$RUN.pt" "$OUTD/${RUN}_qatstart.pt" 2e-5 4 || { echo "FAIL: the QAT start file"; exit 1; }
"$PY" "$TOOLS/patch_rel.py" "$TOOLS/resident_train_qr.py" "$TOOLS/resident_train_qr_rel.py" || { echo "FAIL: the relation patch of the QAT trainer"; exit 1; }
env PYTHONUNBUFFERED=1 NNUE_PYTORCH="$TOOLS/nnue-pytorch" NNUE_BINPACKS="$BP/*.binpack" NNUE_VAL_DATA="$DATA/jantail.*.lcr" NNUE_QAT_ROWS=1 \
    NNUE_RESUME="$OUTD/${RUN}_qatstart.pt" NNUE_CONTINUE=0 NNUE_FEED_THREADS=8 \
    NNUE_BUCKETS=8 NNUE_TAIL_LR_MULT=4 NNUE_L1=768 NNUE_LR=2e-5 NNUE_BS=131072 \
    NNUE_T_MAX_STEPS="$QAT_STEPS" NNUE_BUDGET_SEC=3600 NNUE_LOSS_POWER=3.0 NNUE_QUIET_LOSS=1,0.3,0 \
    NNUE_GRAD_CLIP=1.0 NNUE_TARGET_CLIP=1500 NNUE_VAL_FRAC=0.002 \
    NNUE_EVAL_EVERY="$QEVERY" NNUE_SAVE_EVERY="$QEVERY" NNUE_KEEP_EVERY="$QEVERY" \
    NNUE_ENGINE_HEADER="$TOOLS/nnue_kernels.h" NNUE_COMPILE=1 NNUE_QAT=1 NNUE_QAT_FT8=0 NNUE_FT_DENSE=2 NNUE_VRAM_POS=2300000000 NNUE_RESERVOIR=1000000000 NNUE_SEED=4 NNUE_OUT_DIR="$OUTD" NNUE_OUT_NAME="$Q.nnue" \
    NNUE_REL_T=256 NNUE_REL_SET=full NNUE_REL_PPM=0 NNUE_REL_LR_MULT=4 NNUE_REL_IMPL=triseg NNUE_REL_FUSED=1 NNUE_REL_JOIN=inplace \
    "$PY" -u "$TOOLS/resident_train_qr_rel.py" || { echo "FAIL: the QAT phase"; exit 1; }
"$PY" -c 'import sys, torch; ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); sd = ck["model"]; rel = sd["rel"]; rest = {k: v for k, v in sd.items() if k != "rel"}; k = torch.round(rel * 128.0).clamp(-127, 127).to(torch.int8); torch.save({"model": rest, "gstep": ck.get("gstep", 0)}, sys.argv[2]); torch.save({"rel": rel}, sys.argv[3]); torch.save({"rel_i8": k, "step": 1.0 / 128.0}, sys.argv[4]); print("  saved", sys.argv[2], sys.argv[3], sys.argv[4], "rel", tuple(rel.shape), "int8 |k| max", int(k.abs().max()), "step", ck.get("gstep"))' \
    "$OUTD/$Q.pt" "$OUTD/${Q}_model.pt" "$OUTD/${Q}_rel.pt" "$OUTD/${Q}_rel_i8.pt" || { echo "FAIL: the QAT model's small files"; exit 1; }
echo "== V5 QAT FINISHED: $OUTD/$Q.nnue, ${Q}_model.pt, ${Q}_rel.pt, ${Q}_rel_i8.pt"
# G7: G6's conversion (conv_v5_02_box5.py): LNR8 from the QAT net, then the LNT8 twice, byte-identical
"$PY" "$TOOLS/quantize_i8.py" "$OUTD/$Q.nnue" "$OUTD/${Q}_lnr8.nnue" --rows || { echo "FAIL: quantize_i8.py --rows"; exit 1; }
"$PY" "$TOOLS/lnt8_from_qat.py" "$OUTD/${Q}_lnr8.nnue" "$OUTD/${Q}_rel.pt" "$OUTD/${Q}_i8.nnue" || { echo "FAIL: lnt8_from_qat.py"; exit 1; }
"$PY" "$TOOLS/lnt8_from_qat.py" "$OUTD/${Q}_lnr8.nnue" "$OUTD/${Q}_rel.pt" "/tmp/${Q}_i8_again.nnue" || { echo "FAIL: lnt8_from_qat.py, the second time"; exit 1; }
cmp "$OUTD/${Q}_i8.nnue" "/tmp/${Q}_i8_again.nnue" || { echo "FAIL: the two LNT8 runs differ"; exit 1; }
rm -f "/tmp/${Q}_i8_again.nnue"
echo "  LNT8 twice, byte-identical: $(sha "$OUTD/${Q}_i8.nnue")"
(cd "$OUTD" && sha256sum "${Q}_i8.nnue" "${Q}_lnr8.nnue" "${Q}_rel_i8.pt" "${Q}_model.pt" "${Q}_rel.pt" "${Q}.nnue" "${RUN}_model.pt" "${RUN}_rel.pt") > "$OUTD/SHA256SUMS_03" || { echo "FAIL: SHA256SUMS_03"; exit 1; }
sed 's/^/  /' "$OUTD/SHA256SUMS_03"
echo "== V5 CONVERSION DONE: $OUTD/SHA256SUMS_03"
"""

# Splits a binpack at a chunk boundary into a training head and a held-out tail
# (tools/binpack_split.py; its test on mscope: head + tail = the file, bit for bit).
SPLIT = textwrap.dedent('''
    import os, struct, sys
    p, tail_bytes, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    size = os.path.getsize(p)
    starts = []
    with open(p, "rb") as f:
        off = 0
        while off + 8 <= size:
            f.seek(off)
            head = f.read(8)
            if head[:4] != b"BINP":
                sys.exit(f"no chunk header at byte {off:,}")
            starts.append(off)
            off += 8 + struct.unpack("<I", head[4:])[0]
        if off != size:
            sys.exit(f"the last chunk ends at byte {off:,}, the file at {size:,}")
        cut = next((s for s in reversed(starts) if size - s >= tail_bytes), None)
        if cut is None or cut == 0:
            sys.exit(f"cannot keep {tail_bytes:,} bytes back from {size:,} in {len(starts)} chunks")
        f.seek(cut)
        with open(out + ".part", "wb") as g:
            left = size - cut
            while left:
                buf = f.read(min(left, 1 << 24))
                if not buf:
                    sys.exit("short read")
                g.write(buf)
                left -= len(buf)
    os.replace(out + ".part", out)
    os.truncate(p, cut)
    tail_chunks = sum(1 for s in starts if s >= cut)
    print(f"  split: {len(starts) - tail_chunks:,} chunks ({cut:,} bytes) to train on, "
          f"the last {tail_chunks:,} chunks ({size - cut:,} bytes) held out")
''')

# Data check, run after the conversion: quiet positions with a piece, rook or
# queen more must mostly be labelled as won. The t91 TSVs that dropped every
# score beyond 800 failed exactly this (a queen up read +14 cp).
CHECK = textwrap.dedent('''
    import glob, sys
    import numpy as np
    d = sys.argv[1]
    parts = sorted(glob.glob(d + "/t91_*.000.lcr"))
    if not parts:
        sys.exit("no converted data found")
    recs = np.concatenate([np.fromfile(p, dtype=np.uint8, count=3_000_000 * 32).reshape(-1, 32) for p in parts])
    cnt = recs[:, 27].astype(np.int64)
    nib = np.stack([recs[:, 8:24] & 15, recs[:, 8:24] >> 4], 2).reshape(-1, 32).astype(np.int64)
    valid = np.arange(32)[None, :] < cnt[:, None]
    val = np.array([1, 3, 3, 5, 9, 0] * 2)[nib] * np.where(nib < 6, 1, -1) * valid
    edge = val.sum(1)
    white = (recs[:, 26] & 1).astype(bool)
    tgt = recs[:, 24:26].copy().view(np.int16).ravel().astype(np.int64)
    wt = np.where(white, tgt, -tgt)
    flag = (recs[:, 26] >> 1) & 3
    shares = {}
    for e, name in ((3, "piece"), (5, "rook"), (9, "queen")):
        m = (np.abs(edge) == e) & (flag == 0)
        v = wt[m] * np.sign(edge[m])
        shares[name] = (int(m.sum()), float(np.mean(v >= 800)) if m.any() else float("nan"))
    total = 0
    for f in sorted(glob.glob(d + "/t91_*.done")):
        total += int(open(f).read().split()[1])
    print("  material check, quiet positions labelled beyond +800: " +
          ", ".join(f"{k} up {s:.0%} (of {c:,})" for k, (c, s) in shares.items()) +
          f" | {total:,} positions in all")
    sys.exit(0 if shares["queen"][1] >= 0.9 else 1)
''')


# The draw head, for the float phase: a patched copy of the pinned trainer (tools/resident_train_dh.py).
# NNUE_DRAW_W > 0 adds one logit per output bucket from the 2*L1 activations and the value's size
# (detached), trained with the value on the game result (records with a known result). The gate's
# version concatenated the activations (46 steps/s against 56.6); this one uses two products.
PATCH_DRAW = textwrap.dedent('''
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    rt = open(src, encoding="utf-8").read()


    def patch(old, new, what):
        global rt
        if rt.count(old) != 1:
            sys.exit(f"draw patch: {what} appears {rt.count(old)} times")
        rt = rt.replace(old, new)


    patch("def forward(model, wf, bf, stm, cnt, acc=None):",
          "def forward(model, wf, bf, stm, cnt, acc=None, want_h=False):", "forward()")
    patch("    return F.linear(tail_act(z), Wo, bo).gather(1, bucket).view(n).float() * 300.0\\n",
          "    _o = F.linear(tail_act(z), Wo, bo).gather(1, bucket).view(n).float() * 300.0\\n"
          "    return (_o, h) if want_h else _o\\n", "the value's return")
    patch('def fwd_loss(model, rec):\\n'
          '    wf, bf, stm, tgt, flag, cnt = expand(rec)\\n'
          '    acc = ft_fused(wf, bf, model.ft.weight) if FT_DENSE == 3 else None\\n'
          '    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\\n'
          '        pred = forward(model, wf, bf, stm, cnt, acc)\\n'
          '    return batch_loss(pred.float(), tgt.clamp(-TARGET_CLIP, TARGET_CLIP), flag)\\n',
          'DRAW_W = float(os.environ.get("NNUE_DRAW_W", "0"))\\n'
          '\\n'
          '\\n'
          'def draw_logit(model, h, pred, cnt):\\n'
          '    """The draw head: one logit per output bucket from the activations and the value\\'s size\\n'
          '    (detached: the draw loss shapes the accumulator, not the value\\'s output directly)."""\\n'
          '    n = h.shape[0]\\n'
          '    bucket = ((cnt - 1) * model.num_buckets // 32).clamp(0, model.num_buckets - 1).view(n, 1)\\n'
          '    z = model.dhead(h).float().gather(1, bucket).view(n)\\n'
          '    return z + model.dhead_p[bucket.view(n)] * (pred.detach().float().abs() / 400.0)\\n'
          '\\n'
          '\\n'
          'def fwd_loss(model, rec):\\n'
          '    wf, bf, stm, tgt, flag, cnt = expand(rec)\\n'
          '    acc = ft_fused(wf, bf, model.ft.weight) if FT_DENSE == 3 else None\\n'
          '    dh = getattr(model, "dhead", None)\\n'
          '    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\\n'
          '        if dh is None:\\n'
          '            pred = forward(model, wf, bf, stm, cnt, acc)\\n'
          '        else:\\n'
          '            pred, h = forward(model, wf, bf, stm, cnt, acc, want_h=True)\\n'
          '            dlogit = draw_logit(model, h, pred, cnt)\\n'
          '    vloss = batch_loss(pred.float(), tgt.clamp(-TARGET_CLIP, TARGET_CLIP), flag)\\n'
          '    if dh is None:\\n'
          '        return vloss\\n'
          '    fl = rec[:, 26].to(torch.int64)\\n'
          '    known = ((fl >> 3) & 1).float()\\n'
          '    drawn = (((fl >> 4) & 3) == 1).float()\\n'
          '    bce = F.binary_cross_entropy_with_logits(dlogit, drawn, reduction="none")\\n'
          '    return vloss + DRAW_W * (bce * known).sum() / known.sum().clamp(min=1.0)\\n', "fwd_loss()")
    patch("    model = LNNUE(L1=L1).to(dev)\\n",
          "    model = LNNUE(L1=L1).to(dev)\\n"
          "    if DRAW_W > 0:\\n"
          "        model.dhead = torch.nn.Linear(2 * L1, model.num_buckets).to(dev)\\n"
          "        torch.nn.init.zeros_(model.dhead.weight)\\n"
          "        torch.nn.init.zeros_(model.dhead.bias)\\n"
          "        model.dhead_p = torch.nn.Parameter(torch.zeros(model.num_buckets, device=dev))\\n"
          '        print(f"[draw] draw head on, loss weight {DRAW_W:g}", flush=True)\\n', "model construction")
    patch("    opt = torch.optim.AdamW(groups, lr=LR, amsgrad=True, fused=True)\\n",
          '    if hasattr(model, "dhead"):   # its own group, the last: no weight decay, the tail learning rate\\n'
          '        groups.append({"params": list(model.dhead.parameters()) + [model.dhead_p], "weight_decay": 0.0,\\n'
          '                       "lr": LR * TAIL_LR_MULT})\\n'
          "    opt = torch.optim.AdamW(groups, lr=LR, amsgrad=True, fused=True)\\n", "the optimizer line")
    open(dst, "w", encoding="utf-8").write(rt)
    print(f"  draw head: patched trainer {dst}")
''')

# QAT trains the value alone (the pinned trainer): its start is the float phase's checkpoint without the
# draw head, and without the head's optimizer group (the last of four), so the optimizer state loads.
STRIP_HEADS = textwrap.dedent('''
    import sys
    import torch
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("dhead")}
    dropped = len(ck["model"]) - len(sd)
    opt = ck.get("opt")
    if opt is not None and len(opt["param_groups"]) == 4:
        gone = set(opt["param_groups"][3]["params"])
        opt["param_groups"] = opt["param_groups"][:3]
        opt["state"] = {k: v for k, v in opt["state"].items() if k not in gone}
    torch.save({"model": sd, "gstep": ck.get("gstep", 0), "opt": opt, "swa": None}, dst)
    print(f"  draw head: {dropped} tensors left out of the QAT start ({dst}); optimizer groups: "
          f"{len(opt['param_groups']) if opt else 0}")
''')

# The heads on the final value: ridge per output bucket on the January tail's first 16 files (3M positions),
# scored on its last 4 (1M, other games); features: the 2*L1 activations, |pred|/400 and 1.
FIT_HEADS = textwrap.dedent('''
    import glob, os, sys, time
    import numpy as np
    import torch
    qat_pt, float_pt, data, out, L1 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    here = os.path.dirname(os.path.abspath(__file__))
    os.environ.update(NNUE_BUCKETS="8", NNUE_L1=str(L1), NNUE_FT_DENSE="0", NNUE_QAT="0", NNUE_BS="131072")
    sys.path.insert(0, here)
    import resident_train as RT
    RT.FT_DENSE = 0
    dev, NB, D = "cuda", 8, 2 * L1 + 2
    t0 = time.time()
    model = RT.LNNUE(L1=L1).to(dev)
    model.load_state_dict(torch.load(qat_pt, map_location=dev, weights_only=False)["model"], strict=False)
    model.eval()
    joint = {k: v.to(dev) for k, v in torch.load(float_pt, map_location=dev, weights_only=False)["model"].items()
             if k.startswith("dhead")}
    files = sorted(glob.glob(data + "/jantail.*.lcr"))
    if len(files) < 20:
        sys.exit(f"heads: only {len(files)} January-tail files")


    def batches(paths, cap):
        left = cap
        for p in paths:
            a = np.fromfile(p, dtype=np.uint8)
            a = a[: (len(a) // 32) * 32].reshape(-1, 32)
            for i in range(0, len(a), 65536):
                if left <= 0:
                    return
                chunk = a[i:i + min(65536, left)]
                left -= len(chunk)
                yield torch.from_numpy(chunk).to(dev)


    @torch.no_grad()
    def features(rec):
        wf, bf, stm, tgt, flag, cnt = RT.expand(rec)
        acc_w = model.ft(wf) + model.ft_bias
        acc_b = model.ft(bf) + model.ft_bias
        s = stm.view(-1, 1) > 0.5
        h = torch.clamp(torch.cat([torch.where(s, acc_w, acc_b), torch.where(s, acc_b, acc_w)], 1), 0.0, 1.0).square()
        pred = RT.forward(model, wf, bf, stm, cnt)
        x = torch.cat([h, (pred.abs() / 400.0).view(-1, 1), torch.ones(len(h), 1, device=dev)], 1)
        err = (torch.sigmoid(pred / 400.0) - torch.sigmoid(tgt.clamp(-1500, 1500) / 400.0)).abs()
        fl = rec[:, 26].to(torch.int64)
        known = ((fl >> 3) & 1).bool()
        draw = (((fl >> 4) & 3) == 1).float()
        bucket = ((cnt - 1) * NB // 32).clamp(0, NB - 1)
        return x, h, pred, err, draw, known, bucket


    XtX = torch.zeros(NB, D, D, device=dev, dtype=torch.float64)
    XtXk = torch.zeros(NB, D, D, device=dev, dtype=torch.float64)
    Xe = torch.zeros(NB, D, device=dev, dtype=torch.float64)
    Xd = torch.zeros(NB, D, device=dev, dtype=torch.float64)
    n_fit = 0
    for rec in batches(files[:16], 3_000_000):
        x, h, pred, err, draw, known, bucket = features(rec)
        for b in range(NB):
            m = bucket == b
            if m.any():
                xb = x[m].double()
                XtX[b] += xb.T @ xb
                Xe[b] += xb.T @ err[m].double()
                mk = m & known
                if mk.any():
                    xk = x[mk].double()
                    XtXk[b] += xk.T @ xk
                    Xd[b] += xk.T @ draw[mk].double()
        n_fit += len(x)


    def solve(A, y, lam=1e-3):
        return torch.linalg.solve(A + lam * (torch.trace(A) / A.shape[0]) * torch.eye(A.shape[0], device=dev, dtype=A.dtype), y)


    W_err = torch.stack([solve(XtX[b], Xe[b]) for b in range(NB)]).float()
    W_draw = torch.stack([solve(XtXk[b], Xd[b]) for b in range(NB)]).float()
    print(f"heads: fitted on {n_fit:,} positions in {time.time() - t0:.0f} s", flush=True)

    E, P, Eh, Dh, Dj, Dy, K, Bk = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for rec in batches(files[-4:], 1_000_000):
            x, h, pred, err, draw, known, bucket = features(rec)
            E.append(err); P.append(pred); Dy.append(draw); K.append(known); Bk.append(bucket)
            Eh.append((x * W_err[bucket]).sum(1))
            Dh.append((x * W_draw[bucket]).sum(1))
            if joint:
                z = (h @ joint["dhead.weight"].T + joint["dhead.bias"]).gather(1, bucket.view(-1, 1)).view(-1)
                Dj.append(z + joint["dhead_p"][bucket] * (pred.abs() / 400.0))
    E, P, Eh, Dh, Dy, K, Bk = (torch.cat(v) for v in (E, P, Eh, Dh, Dy, K, Bk))
    Dj = torch.cat(Dj) if Dj else None


    def ranks(v):
        r = torch.empty_like(v, dtype=torch.float64)
        r[torch.argsort(v)] = torch.arange(len(v), device=v.device, dtype=torch.float64)
        return r


    def spearman(a, b):
        ra, rb = ranks(a.double()), ranks(b.double())
        ra, rb = ra - ra.mean(), rb - rb.mean()
        return float((ra * rb).sum() / (ra.norm() * rb.norm()))


    def auc(score, label):
        r = ranks(score.double()) + 1
        pos = label > 0.5
        n1, n0 = int(pos.sum()), int((~pos).sum())
        return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


    sc = {"err_head": spearman(Eh, E), "err_abs_pred": spearman(P.abs(), E)}
    per = "  ".join(f"b{b}:{spearman(Eh[Bk == b], E[Bk == b]):.2f}/{spearman(P[Bk == b].abs(), E[Bk == b]):.2f}"
                    for b in range(NB) if int((Bk == b).sum()) > 2000)
    print(f"  error head: Spearman with the actual error {sc['err_head']:.3f} (the value's size alone {sc['err_abs_pred']:.3f}); "
          f"per bucket head/|pred|: {per}", flush=True)
    near = K & (P.abs() < 50)
    sc.update(draw_head=auc(Dh[K], Dy[K]), draw_head_near=auc(Dh[near], Dy[near]),
              draw_abs_pred=auc(-P[K].abs(), Dy[K]), draw_abs_pred_near=auc(-P[near].abs(), Dy[near]))
    line = (f"  draw head: AUC {sc['draw_head']:.3f} (-|pred| {sc['draw_abs_pred']:.3f}); among |pred| < 50 cp "
            f"({int(near.sum()):,}): {sc['draw_head_near']:.3f} (-|pred| {sc['draw_abs_pred_near']:.3f})")
    if Dj is not None:
        sc.update(draw_joint=auc(Dj[K], Dy[K]), draw_joint_near=auc(Dj[near], Dy[near]))
        line += f"; the jointly trained head on this value: {sc['draw_joint']:.3f} / {sc['draw_joint_near']:.3f}"
    print(line, flush=True)
    torch.save({"err": W_err.cpu(), "draw": W_draw.cpu(), "L1": L1, "buckets": NB,
                "features": "per output bucket: the 2*L1 activations (side to move first), |pred|/400, 1",
                "err_label": "|sigmoid(pred/400) - sigmoid(target/400)|", "draw_label": "the game was drawn",
                "fit": "BT4 January tail, files[:16], 3M positions; scored on files[-4:]", "scores": sc,
                "joint_draw_head": {k: v.cpu() for k, v in joint.items()}}, out)
    print(f"  heads: wrote {out} in {time.time() - t0:.0f} s", flush=True)
''')


# Per-row tail scales for QAT: fq_w rounds each ROW (output neuron) of L2, L3 and out to int8 with its own
# scale when NNUE_QAT_ROWS=1 (the engine's LNR8 format), instead of one scale per bucket block. Measured on
# run 1's net: the engine's loss +1.72% over float with the block scale, +0.22% with QAT per row.
PATCH_ROWS = textwrap.dedent('''
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    rt = open(src, encoding="utf-8").read()
    old = ("    g = w.view(groups, -1)\\n"
           "    s = 127.0 / g.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)\\n"
           "    q = torch.round(g * s).clamp(-127.0, 127.0) / s\\n"
           "    return (g + (q - g).detach()).view_as(w)\\n")
    new = ("    if os.environ.get(\\"NNUE_QAT_ROWS\\") == \\"1\\":   # one int8 scale per row (output neuron)\\n"
           "        s = 127.0 / w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)\\n"
           "        q = torch.round(w * s).clamp(-127.0, 127.0) / s\\n"
           "        return w + (q - w).detach()\\n" + old)
    if rt.count(old) != 1:
        sys.exit(f"per-row patch: fq_w's body appears {rt.count(old)} times")
    open(dst, "w", encoding="utf-8").write(rt.replace(old, new))
    print(f"  per-row tail scales: patched trainer {dst} (NNUE_QAT_ROWS=1)")
''')


# The head of a binpack, cut at a chunk boundary (tools/binpack_head.py): the material check's sample.
BPHEAD = textwrap.dedent('''
    import os, struct, sys
    src, want, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    size = os.path.getsize(src)
    off = 0
    with open(src, "rb") as f:
        while off < want and off + 8 <= size:
            f.seek(off)
            h = f.read(8)
            if h[:4] != b"BINP":
                sys.exit(f"no chunk header at byte {off:,}")
            n = struct.unpack("<I", h[4:])[0]
            if off + 8 + n > size:
                sys.exit(f"the chunk at byte {off:,} runs past the end of the file")
            off += 8 + n
        f.seek(0)
        with open(out + ".part", "wb") as g:
            left = off
            while left:
                b = f.read(min(left, 1 << 24))
                if not b:
                    sys.exit("short read")
                g.write(b)
                left -= len(b)
    os.replace(out + ".part", out)
    print(f"  sample: the first {off:,} bytes of {os.path.basename(src)} ({size:,})")
''')

# The corpus check (tools/check_sources.py), per file on its sample: positions per byte (and so the file's
# estimated positions), the flag shares, and the material check of check_data.py: quiet positions a piece, rook or
# queen up must mostly be labelled beyond +800 (queen up >= 90%, as t91's months).
CHECK_SRC = textwrap.dedent('''
    import glob, os, re, sys
    import numpy as np
    d = sys.argv[1]
    bad, tot = [], 0
    for done in sorted(glob.glob(d + "/*.done")):
        name = os.path.basename(done)[:-5]
        sb, fb, rest = open(done).read().split(" ", 2)
        sb, fb = int(sb), int(fb)
        m = re.search(r"DONE (\\d+) positions", rest)
        n = int(m.group(1)) if m else 0
        parts = sorted(glob.glob(f"{d}/{name}.*.lcr"))
        if not parts or not n:
            print(f"  {name}: no converted sample")
            bad.append(name)
            continue
        recs = np.concatenate([np.fromfile(p, dtype=np.uint8, count=400_000 * 32).reshape(-1, 32) for p in parts])
        cnt = recs[:, 27].astype(np.int64)
        nib = np.stack([recs[:, 8:24] & 15, recs[:, 8:24] >> 4], 2).reshape(-1, 32).astype(np.int64)
        valid = np.arange(32)[None, :] < cnt[:, None]
        val = np.array([1, 3, 3, 5, 9, 0] * 2)[nib] * np.where(nib < 6, 1, -1) * valid
        edge = val.sum(1)
        white = (recs[:, 26] & 1).astype(bool)
        tgt = recs[:, 24:26].copy().view(np.int16).ravel().astype(np.int64)
        wt = np.where(white, tgt, -tgt)
        flag = (recs[:, 26] >> 1) & 3
        shares = {}
        for e, nm in ((3, "piece"), (5, "rook"), (9, "queen")):
            k = (np.abs(edge) == e) & (flag == 0)
            v = wt[k] * np.sign(edge[k])
            shares[nm] = (int(k.sum()), float(np.mean(v >= 800)) if k.any() else float("nan"))
        quiet = flag == 0
        med = float(np.median(np.abs(tgt[quiet]))) if quiet.any() else float("nan")
        est = fb * n // sb
        tot += est
        print(f"  {name}: {n:,} positions in a {sb / 1e9:.2f} GB sample ({sb / n:.2f} bytes each), so ~{est / 1e9:.2f}B in "
              f"its {fb / 1e9:.1f} GB; quiet/capture/check {np.mean(flag == 0):.0%}/{np.mean(flag == 1):.0%}/{np.mean(flag == 2):.0%}; "
              f"quiet |score| median {med:.0f}; material check: " +
              ", ".join(f"{k} up {s:.0%} (of {c:,})" for k, (c, s) in shares.items()))
        if not shares["queen"][1] >= 0.9:
            bad.append(name)
    print(f"  the corpus: ~{tot / 1e9:.1f}B positions estimated from the samples" + (f"; FAILED: {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)
''')


def kill_old_run():
    """Stop a run left over from an earlier run of this cell, so two jobs never
    share the GPU or the data directory."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        if any(a.endswith((b"resident_train.py", b"c500_train_stream.py")) for a in argv) or \
                any(a.endswith((b"binpack2lcr", b"luminex-featurize")) for a in argv[:1]):
            try:
                os.kill(int(pid), signal.SIGKILL)
            except OSError:
                pass


LOG = os.path.join(HOME, RUN + ".log")
HW_SECONDS = 15       # hardware sample interval: GPU use, VRAM, power, CPU
state = {"lines": 0, "last": time.time(), "phase": "starting", "step": "", "health": "", "disjoint": "",
         "guard": "", "hw": "", "guard_shown": 0}
recent = collections.deque(maxlen=SHOW_LINES)
console = collections.deque()   # key lines, printed to the cell console (a paired agent reads that)
KEY_MARKS = ("== ", "[HEALTH", "[DISJOINT", "budget guard", "FAIL", "DONE", ">>> ", "[data]", "[DATA]", "[train]",
             "[RESUME]", "[save] final", "[keep]", "jan tail", "split:", "removed the files", "checkpoint:",
             "Traceback", "Error", "material check", "  ... ", "sha256 ok", "binpack2lcr", "pfetch", "zstd:", "data phase",
             "  2025-", "  2026-", "wrote ", "[draw]", "HEADS", "heads:", "  error head", "  draw head", "per-row",
             "  t91_", "  x_", "sample:", "corpus", "binpacks for the feeder", "STALL", "stall", "[feed", "[bin")
lock = threading.Lock()


def sample_hw(stop):
    """Hardware sampler: every HW_SECONDS, GPU utilization, VRAM, power and
    temperature (nvidia-smi), whole-machine CPU use (/proc/stat) and RAM in use
    (/proc/meminfo: it also shows whether the data files are held in memory),
    as one "[hw]" line in the log."""
    def cpu_times():
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + (v[4] if len(v) > 4 else 0)
    log = open(LOG, "a", buffering=1, errors="replace")
    prev = cpu_times()
    while not stop.wait(HW_SECONDS):
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            util, used, total, power, temp = [x.strip() for x in q.strip().split(",")]
            gpu = f"gpu {util}% vram {float(used) / 1024:.1f}/{float(total) / 1024:.0f} GB {float(power):.0f} W {temp}C"
        except Exception as e:
            gpu = f"gpu ? ({e.__class__.__name__})"
        cur = cpu_times()
        dt, di = cur[0] - prev[0], cur[1] - prev[1]
        prev = cur
        with open("/proc/loadavg") as f:
            load = f.read().split()[0]
        try:
            with open("/proc/meminfo") as f:
                mem = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in f)}
            ram = f" | ram {(mem['MemTotal'] - mem['MemAvailable']) / 2**20:.1f}/{mem['MemTotal'] / 2**20:.0f} GB"
        except (OSError, ValueError, KeyError, IndexError):
            ram = ""
        line = f"[hw] {time.strftime('%H:%M:%S')} {gpu} | cpu {100 * (1 - di / max(dt, 1)):.0f}% load {load}{ram}"
        log.write(line + "\n")
        with lock:
            state["hw"] = line
            console.append(line)
    log.close()


def read_output(pipe):
    """Reader thread: every line goes to the log file and to the panel, which
    also keeps the latest phase, training-step, health and budget-guard lines.
    Over a full run the trainer writes about 5,000 step lines (one per 50 steps)
    and a budget-guard line per 200 steps, so the panel and console get a step
    line per SHOW_STEP_EVERY steps and a guard line when the horizon moves by more
    than 2%; the status lines at the top always show the latest of each."""
    log = open(LOG, "a", buffering=1, errors="replace")
    for line in pipe:
        log.write(line)
        text = line.rstrip("\n")
        show = True
        with lock:
            state["lines"] += 1
            state["last"] = time.time()
            if text.startswith("== "):
                state["phase"] = text[3:]
            elif "[HEALTH" in text or "material check" in text:   # before the step test: "[HEALTH step N] val loss="
                state["health"] = text.strip()
            elif "[DISJOINT" in text:   # also before it: "[DISJOINT step N] loss="
                state["disjoint"] = text.strip()
            elif " step " in text and "loss=" in text:
                state["step"] = text.strip()
                try:
                    show = int(text.split(" step ", 1)[1].split(" ", 1)[0]) % SHOW_STEP_EVERY == 0
                except ValueError:
                    pass
            elif "budget guard" in text:
                state["guard"] = text.strip()
                try:
                    horizon = int(text.split("-> ", 1)[1].split(" ", 1)[0].replace(",", ""))
                    show = abs(horizon - state["guard_shown"]) > 0.02 * horizon
                    if show:
                        state["guard_shown"] = horizon
                except (IndexError, ValueError):
                    pass
            if show:
                recent.append(text)
                if any(m in text for m in KEY_MARKS) or (" step " in text and "loss=" in text):
                    console.append(text.strip())
    log.close()


def panel(started, status):
    """The whole cell output: status, the latest key lines, then the log."""
    with lock:
        s, shown = dict(state), list(recent)
    up = int(time.time() - started)
    mode = " (data only)" if DATA_ONLY else " (smoke test)" if SMOKE else ""
    head = (f"{RUN}{mode}: {status} {up // 3600}h {up % 3600 // 60:02d}m · "
            f"refreshes every {REFRESH_SECONDS} s · last output {int(time.time() - s['last'])} s ago\n"
            f"phase:  {s['phase']}\n"
            f"step:   {s['step'] or '-'}\n"
            f"health: {s['health'] or '-'}\n"
            + (f"held:   {s['disjoint']}\n" if s["disjoint"] else "")
            + (f"guard:  {s['guard']}\n" if s["guard"] else "")
            + (f"hw:     {s['hw']}\n" if s["hw"] else "")
            + f"full log: {LOG} · data: {HOME}/data/ · output: {HOME}/out/\n" + "─" * 46 + "\n")
    return mo.plain_text(head + "\n".join(shown))


os.makedirs(os.path.join(HOME, "tools"), exist_ok=True)
with open(os.path.join(HOME, "tools", "check_data.py"), "w") as _f:
    _f.write(CHECK)
with open(os.path.join(HOME, "tools", "binpack_split.py"), "w") as _f:
    _f.write(SPLIT)
for _name, _text in (("patch_draw.py", PATCH_DRAW), ("strip_heads.py", STRIP_HEADS), ("fit_heads.py", FIT_HEADS),
                     ("patch_rows.py", PATCH_ROWS), ("binpack_head.py", BPHEAD), ("check_sources.py", CHECK_SRC)):
    with open(os.path.join(HOME, "tools", _name), "w") as _f:
        _f.write(_text)
kill_old_run()
time.sleep(1)
started = time.time()
proc = subprocess.Popen(
    ["bash", "-c", SCRIPT, "bash", HOME, RUN, str(STEPS), str(int(HOURS * 3600)), COMMIT,
     "1" if DATA_ONLY else "0", sys.executable, MONTHS, NP_COMMIT, str(CONNECTIONS), "1" if SMOKE else "0",
     str(LOSS_POWER), str(KEEP_EVERY), str(LR), str(VRAM_POS), str(QAT_STEPS), str(QAT_LR), str(QAT_RESERVE),
     str(QAT_FT8), str(JAN_TAIL_MB * 1048576), str(DRAW_W), EXTRA, str(SAMPLE_BYTES)],
    env=dict(os.environ, PYTHONUNBUFFERED="1"), stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    errors="replace", start_new_session=True)
reader = threading.Thread(target=read_output, args=(proc.stdout,), daemon=True)
reader.start()
hw_stop = threading.Event()
threading.Thread(target=sample_hw, args=(hw_stop,), daemon=True).start()
try:
    while True:
        mo.output.replace(panel(started, "running"))
        with lock:
            news = list(console)
            console.clear()
        for text in news:
            print(text, flush=True)
        try:
            proc.wait(timeout=REFRESH_SECONDS)
            break
        except subprocess.TimeoutExpired:
            pass
    reader.join(timeout=5)
    with lock:
        news = list(console)
        console.clear()
    for text in news:
        print(text, flush=True)
    mo.output.replace(panel(started, f"finished (exit code {proc.returncode}) after"))
finally:
    hw_stop.set()
    if proc.poll() is None:  # the cell was stopped: stop the run
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        kill_old_run()
        mo.output.replace(panel(started, "stopped by you after"))
