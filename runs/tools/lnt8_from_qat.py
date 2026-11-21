"""Write a gen768t (LNT8) network from the V5 arm's QAT phase: lnt8_from_gate.py (branch v5-engine-path) with the
relation rows taken from the trainer's own float table instead of the gate's x8192 row file.

ARM_i8.nnue  the QAT net's LNR8 file (quantize_i8.py --rows on the QAT phase's float .nnue): the float feature layer
             and the int8 tails with one scale per row, as NNUE_QAT_ROWS=1 trained them
REL.pt       {"rel": float tensor [55960, T]}, the QAT checkpoint's relation table (qat_v5_box1.py's _rel.pt)
OUT.nnue     the LNT8 file (nnue/V5-GEN768T.md, section 1)

The relation rows become q = rint(128 w) clipped to +-127, which is what rel_gate.py's QAT forward read
(torch.round: ties to even, as np.rint); rounding the table once more through a x8192 file, as the gate's export did,
could move a row that sits within 1/16384 of a tie. The piece-square table and bias are rounded onto x2048 half away
from zero, as the engine rounds an LNR8 net at load, so the LNT8 net's piece-square lanes are the ones the engine
computes for the LNR8 file; the tails and any section after them are LNR8's bytes unchanged.
usage: lnt8_from_qat.py ARM_i8.nnue REL.pt OUT.nnue [PPM]"""
import struct
import sys

import numpy as np
import torch

FS, RS, LAYOUT, ROWS = 11, 4, 1, 55960
args = sys.argv[1:]
if len(args) not in (3, 4):
    raise SystemExit(__doc__)
src, rel_path, out = args[:3]
ppm = int(args[3]) if len(args) == 4 else 0

raw = open(src, "rb").read()
if raw[:4] != b"LNR8":
    raise SystemExit(f"FAIL: {src} is not an LNR8 net (quantize_i8.py --rows)")
L1, L2, L3, NI, NB = struct.unpack("<5i", raw[4:24])
off = 24
(n,) = struct.unpack("<i", raw[off:off + 4])
if n != L1 * NI:
    raise SystemExit(f"FAIL: {src}: feature table of {n} values, expected {L1 * NI}")
ft = np.frombuffer(raw, dtype="<f4", count=n, offset=off + 4).reshape(L1, NI)   # lane-major, as LNR8 stores it
off += 4 + 4 * n
(n,) = struct.unpack("<i", raw[off:off + 4])
if n != L1:
    raise SystemExit(f"FAIL: {src}: bias of {n} values, expected {L1}")
bias = np.frombuffer(raw, dtype="<f4", count=n, offset=off + 4)
off += 4 + 4 * n
tails = raw[off:]


def half_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


scale = np.float32(1 << FS)
q = half_away((ft.astype(np.float32) * scale).astype(np.float64))
clipped = int((np.abs(q) > 32767).sum())
q = np.clip(q, -32767, 32767).astype("<i2").T.copy()                             # feature-major [NI][L1]
qb = half_away((bias.astype(np.float32) * scale).astype(np.float64)).astype("<i4")
beyond = int((np.abs(ft) > 3.99).sum())

rel = torch.load(rel_path, map_location="cpu", weights_only=False)["rel"].float().numpy()
rows, T = rel.shape
if rows != ROWS:
    raise SystemExit(f"FAIL: {rel_path}: {rows} rows; LNT8 layout 1 is the full set of {ROWS}")
if T % 64 or T > L1:
    raise SystemExit(f"FAIL: {rel_path}: T = {T}, a multiple of 64 up to L1 = {L1}")
r = np.rint(rel.astype(np.float64) * float(1 << (FS - RS)))                       # steps of 1/128
rel_clipped = int((np.abs(r) > 127).sum())
r8 = np.clip(r, -127, 127).astype(np.int8)

with open(out, "wb") as f:
    f.write(b"LNT8")
    f.write(struct.pack("<5i", L1, L2, L3, NI, NB))
    f.write(struct.pack("<6i", T, rows, FS, RS, ppm, LAYOUT))
    f.write(struct.pack("<i", NI * L1))
    f.write(q.tobytes())
    f.write(struct.pack("<i", L1))
    f.write(qb.tobytes())
    f.write(struct.pack("<i", rows * T))
    f.write(r8.tobytes())
    f.write(tails)
print(f"{out}: L1 {L1}, L2 {L2}, L3 {L3}, {NB} buckets; T {T}, {rows} rows, pawn pairs {'all' if ppm else 'in contact'}, "
      f"grid x{1 << FS}, step 1/{1 << (FS - RS)}; piece-square weights beyond 3.99: {beyond}, clipped to int16: {clipped}; "
      f"relation values clipped to int8: {rel_clipped}; rows |q| max {int(np.abs(r8).max())}, nonzero {float((r8 != 0).mean()):.2%}")
