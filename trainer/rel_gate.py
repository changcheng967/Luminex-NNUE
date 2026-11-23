"""V5 value gate: attack-relation inputs computed from the raw boards, for the trainer only.

No engine code is involved: no incremental accumulator, no change-finder. Each batch's LCR1 records are expanded on
the device into the relation and pawn-pair features of nnue/V5-draft/V5-SPEC.md section 2, for both perspectives,
and their rows are summed into the first T lanes of the feature layer ahead of the bias and SCReLU.

The index rules below are the reference script's (v5-relations-ref.py: block_mask, the blocks' order, PP_INDEX),
copied so that this file stands alone on the box; check_rel_gate.py compares the two on real positions.

Environment:
  NNUE_REL_T [0]         relation lanes. 0 = off: the patched trainer then computes what the unpatched one does
  NNUE_REL_SET [full]    full = the spec's set; reduced = attacks on the other colour, pawn attackers, pawn pairs
  NNUE_REL_PPM [0]       0 = pawn pairs in contact (one attacks the other's square); 1 = every pair on the same or
                         adjacent files
  NNUE_REL_LR_MULT [4]   the relation table's learning rate, as a multiple of NNUE_LR (its own optimizer group,
                         no weight decay)
  NNUE_REL_IMPL [index]  how the rows are summed, four ways to the same fp32 sums: index = index_select +
                         index_add_ (plain autograd); spmm = two sparse-dense products; embag = embedding_bag;
                         triton = rel_kernels.py (a gather forward in embedding_bag's order, an atomic backward; on a
                         GPU, or on the CPU under TRITON_INTERPRET; elsewhere embag); trieb = the same gather forward
                         with embedding_bag's own backward (both bit for bit embag's, deterministic); triseg = the
                         gather forward with rel_kernels.seg_grad's sorted backward (deterministic, no atomics);
                         trirows = triseg's sums and gradient bit for bit, the rows computed from the records by
                         rel_rows.py's kernel (no dense pass, no nonzero and its host sync; pawn pairs in contact,
                         NNUE_REL_PPM=0, else triseg)
  NNUE_REL_TABLE16 [0]   with triton: 1 = the forward reads a bf16 copy of the table (half the bytes; exact under
                         QAT's k / 128 rows), 0 = the fp32 table
  NNUE_REL_CHUNK [2097152]  active rows per index_add_ call (bounds the temporaries)
  NNUE_REL_OPT [torch]   the step: torch = clip_grad_norm_, AdamW and the clamp as three calls; fused = the relation
                         table's share in one pass (rel_kernels.rel_adamw); fused16 = the relation table's and the
                         feature table's each in one pass that also writes the bf16 copy the next forward reads
                         (rel_kernels.adamw16): the same values bit for bit, see step()
  NNUE_REL_SEGLIST [0]   trirows/triseg: 1 = seg_grad's join over the rows with two or more pieces only, found
                         without a host sync (rel_kernels._seg_join_list): the same sums
  NNUE_REL_ROWS2 [0]     trirows: 1 = the rows by rel_rows._rel_rows2, the same rows by fewer instructions
  NNUE_REL_GS16 [0]      trirows, float phase: 1 = the table's gradient from act_bwd's bf16 gs (its lanes 0..T-1 are
                         bf16 of the fp32 lane gradient, which the feature table's gradient already reads) through
                         the bucket sort's inverse: no fp32 lane gradient written, half the bytes in _seg_piece.
                         NOT bit for bit (each lane gradient rounded to bf16 before the table's sums): proven
                         loss-equal on box 10 (DEVLOG section 26) before any run used it
  NNUE_REL_LANE16 [0]    with NNUE_REL_GS16: 1 = the relation rows' lane sums stored in bf16 (each bag's fp32 sum,
                         rounded once), as the feature layer's own sums are: half the bytes _rel_fwd writes and
                         _ft_fwd and _act_bwd read. NOT bit for bit: proven loss-equal on box 10 (DEVLOG section 27)
  NNUE_REL_FUSED [1]     1: on a GPU, indices() as one compiled dense pass (the same rows in the same order);
                         0: the step-by-step version, which the CPU always runs
  NNUE_REL_JOIN [inplace]   how the sums meet the feature layer's lanes: inplace = one autograd function that adds
                         them where they are and passes the gradient through; cat = the first version's torch.cat
                         (the same values, bit for bit, and 8 ms a step slower at the gate's batch)
The float phase: the rows are clipped to +-127/128 after every step (the int8 range at 1/128 a step), and the
lane sum is not clamped (the [REL] line reports how often it would be).
QAT (NNUE_QAT=1, a phase of its own from the float phase's checkpoint; NNUE_QAT_AFTER is refused): the forward reads
the engine's rows, round(128 w) / 128 with |k| <= 127 (one step = 2^RS on the x2048 grid, RS = 4), with the gradient
passed straight through to the float rows, and clamps each lane sum to +-511 steps (LANE_L = 3.99), as V5-SPEC's
combination clamp(rel_i, -L, L) * 2^RS does before adding it to the piece-square lane.
"""
import os

import torch

T = int(os.environ.get("NNUE_REL_T", "0"))
SET = os.environ.get("NNUE_REL_SET", "full")
PPM = int(os.environ.get("NNUE_REL_PPM", "0"))
LR_MULT = float(os.environ.get("NNUE_REL_LR_MULT", "4"))
IMPL = os.environ.get("NNUE_REL_IMPL", "index")
CHUNK = int(os.environ.get("NNUE_REL_CHUNK", str(1 << 21)))
JOIN = os.environ.get("NNUE_REL_JOIN", "inplace")
FUSED = os.environ.get("NNUE_REL_FUSED", "1") == "1"
TABLE16 = os.environ.get("NNUE_REL_TABLE16", "0") == "1"
G16 = os.environ.get("NNUE_REL_G16", "0") == "1"   # trirows: the backward reads the sums' gradient in bf16
CLIP = 127.0 / 128.0
LANE_L = 511.0 / 128.0
if (SET not in ("full", "reduced") or PPM not in (0, 1) or IMPL not in ("index", "spmm", "embag", "triton", "trieb", "triseg", "trirows")
        or JOIN not in ("inplace", "cat")):
    raise SystemExit(f"FAIL: NNUE_REL_SET={SET} NNUE_REL_PPM={PPM} NNUE_REL_IMPL={IMPL} NNUE_REL_JOIN={JOIN}")

# ---------------------------------------------------------------------------
# The index layout (v5-relations-ref.py, verbatim in substance)
# ---------------------------------------------------------------------------
P, N, B, R, Q, K = range(6)
OWN, THEIR = 0, 1
KN_D = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
DIAG = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
ORTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
K_D = DIAG + ORTH


def _step(s, d):
    f, r = (s & 7) + d[0], (s >> 3) + d[1]
    return r * 8 + f if 0 <= f < 8 and 0 <= r < 8 else None


def _bb(s):
    return 1 << s


def _pop(m):
    return bin(m).count("1")


def _bits(m):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


def _jumps(s, ds):
    m = 0
    for d in ds:
        t = _step(s, d)
        if t is not None:
            m |= _bb(t)
    return m


def _lines(s, ds):
    m = 0
    for d in ds:
        t = _step(s, d)
        while t is not None:
            m |= _bb(t)
            t = _step(t, d)
    return m


RANKS_2_7 = sum(_bb(s) for s in range(8, 56))
EB_PUP = [_jumps(s, [(-1, 1), (1, 1)]) if 8 <= s < 56 else 0 for s in range(64)]
EB_PDN = [_jumps(s, [(-1, -1), (1, -1)]) if 8 <= s < 56 else 0 for s in range(64)]
EB_N = [_jumps(s, KN_D) for s in range(64)]
EB_K = [_jumps(s, K_D) for s in range(64)]
EB_D = [_lines(s, DIAG) for s in range(64)]
EB_O = [_lines(s, ORTH) for s in range(64)]
EB_Q = [EB_D[s] | EB_O[s] for s in range(64)]
ABOVE = [~(_bb(s + 1) - 1) & ((1 << 64) - 1) for s in range(64)]


def _pawn_facing(x, sv):
    ds = [(-1, -1), (1, -1)] if sv == OWN else [(-1, 1), (1, 1)]
    return _jumps(x, ds) & RANKS_2_7


def block_mask(sa, ta, sv, tv, x):
    """The oriented victim squares y' for which feature (sa,ta)@x' -> (sv,tv)@y' exists."""
    if tv == K:
        return 0
    if ta == K and sa == OWN:
        return 0
    if ta == P and tv == P:
        return 0
    if ta == P:
        m = EB_PUP[x] if sa == OWN else EB_PDN[x]
    elif ta == N:
        m = EB_N[x]
    elif ta == B:
        m = EB_D[x]
    elif ta == R:
        m = EB_O[x]
    elif ta == Q:
        m = EB_Q[x]
        if tv == B:
            m = EB_O[x]
        elif tv == R:
            m = EB_D[x]
    else:
        m = EB_K[x]
    if tv == P:
        m &= RANKS_2_7
        if ta in (B, Q):
            m &= ~_pawn_facing(x, sv)
    if ta == tv and ta in (N, B, R, Q):
        if sa == sv:
            m &= ABOVE[x]
        elif sa == THEIR:
            return 0
    return m


BLOCKS, N_REL = [], 0
for _sa in (OWN, THEIR):
    for _ta in (P, N, B, R, Q, K):
        for _sv in (OWN, THEIR):
            for _tv in (P, N, B, R, Q):
                _masks = [block_mask(_sa, _ta, _sv, _tv, x) for x in range(64)]
                _size = sum(_pop(m) for m in _masks)
                if _size == 0:
                    continue
                _offs, _acc = [], 0
                for m in _masks:
                    _offs.append(_acc)
                    _acc += _pop(m)
                BLOCKS.append((_sa, _ta, _sv, _tv, N_REL, _masks, _offs))
                N_REL += _size
PP_INDEX = {}
for _i in range(96):
    for _j in range(_i + 1, 96):
        _qi, _qj = _i % 48, _j % 48
        if _qi == _qj or abs(((_qi + 8) & 7) - ((_qj + 8) & 7)) > 1:
            continue
        PP_INDEX[(_i, _j)] = len(PP_INDEX)
N_PP = len(PP_INDEX)
N_ROWS = N_REL + N_PP
assert (N_REL, N_PP, N_ROWS) == (54472, 1488, 55960), (N_REL, N_PP)


def _fid(sa, ta, x, sv, tv, y):
    return ((((sa * 6 + ta) * 64 + x) * 2 + sv) * 6 + tv) * 64 + y


FID_SIZE = 2 * 6 * 64 * 2 * 6 * 64


def _feature_table(reduced):
    """Row index of (attacker side, type, oriented square) -> (victim side, type, oriented square), or -1 when the
    pair is not a feature in that telling. A mutual attack has two tellings and exactly one of them is a feature;
    pawn on pawn is the pawn-pair row, the same for both orders."""
    t = [-1] * FID_SIZE
    for sa, ta, sv, tv, base, masks, offs in BLOCKS:
        if reduced and sa == sv and ta != P:
            continue                               # the reduced set: no piece on its own side's pieces
        for x in range(64):
            m = masks[x]
            for y in _bits(m):
                t[_fid(sa, ta, x, sv, tv, y)] = base + offs[x] + _pop(m & (_bb(y) - 1))
    for (i, j), k in PP_INDEX.items():
        si, qi = divmod(i, 48)
        sj, qj = divmod(j, 48)
        t[_fid(si, P, qi + 8, sj, P, qj + 8)] = N_REL + k
        t[_fid(sj, P, qj + 8, si, P, qi + 8)] = N_REL + k
    return t


# Geometry for the device. A slider's first blocker along a line is the lowest set bit of occupancy & line: for the
# four directions in which square numbers grow (E, N, NE, NW) on the board itself, for the other four (W, S, SW, SE)
# on the board turned by 180 degrees, where they become the first four. _PM[s] holds the four lines from s as
# signed 64-bit masks; row 64 (no piece) is empty. A piece's eight targets are ordered E N NE NW W S SW SE.
POS_DIRS = [(1, 0), (0, 1), (1, 1), (-1, 1)]


def _signed(m):
    return m - (1 << 64) if m >= (1 << 63) else m


_PM = [[_signed(_lines(s, [d])) for d in POS_DIRS] for s in range(64)] + [[0, 0, 0, 0]]
_DBC = 0x03F79D71B4CB0A89                                    # de Bruijn: the index of a single set bit, exactly
_DB = [0] * 64
for _i in range(64):
    _DB[(((1 << _i) * _DBC) & ((1 << 64) - 1)) >> 58] = _i
assert sorted(_DB) == list(range(64))
_KN = [[64] * 8 for _ in range(65)]
_KG = [[64] * 8 for _ in range(65)]
_PA = [[[64] * 8 for _ in range(65)] for _ in range(2)]      # [colour: 0 white, 1 black][square]
for _s in range(64):
    for _k, _dd in enumerate(KN_D):
        _t = _step(_s, _dd)
        _KN[_s][_k] = 64 if _t is None else _t
    for _k, _dd in enumerate(K_D):
        _t = _step(_s, _dd)
        _KG[_s][_k] = 64 if _t is None else _t
    for _c, _ds in ((0, [(-1, 1), (1, 1)]), (1, [(-1, -1), (1, -1)])):
        for _k, _dd in enumerate(_ds):
            _t = _step(_s, _dd)
            _PA[_c][_s][_k] = 64 if _t is None else _t
_DIR_OK = [[False] * 8 for _ in range(6)]
for _d in range(8):                                         # E N NE NW W S SW SE
    _DIR_OK[R][_d] = _d in (0, 1, 4, 5)
    _DIR_OK[B][_d] = _d in (2, 3, 6, 7)
    _DIR_OK[Q][_d] = True

_DEV = {}      # device -> tensors
_CTX = {"rec": None}


def _tables(dev):
    t = _DEV.get(str(dev))
    if t is None:
        L, U = torch.long, torch.uint8
        t = dict(PM=torch.tensor(_PM, dtype=L, device=dev), DB=torch.tensor(_DB, dtype=U, device=dev),
                 KN=torch.tensor(_KN, dtype=U, device=dev),
                 KG=torch.tensor(_KG, dtype=U, device=dev), PA=torch.tensor(_PA, dtype=U, device=dev),
                 DIR_OK=torch.tensor(_DIR_OK, dtype=torch.bool, device=dev),
                 FID=torch.tensor(_feature_table(SET == "reduced"), dtype=L, device=dev),
                 SQ64=torch.arange(64, device=dev),
                 TRI=torch.triu(torch.ones((16, 16), dtype=torch.bool, device=dev), 1))
        _DEV[str(dev)] = t
    return t


# ---------------------------------------------------------------------------
# Records -> active rows, both perspectives
# ---------------------------------------------------------------------------
def _first(m, DB):
    """The square of each mask's lowest set bit, 64 where the mask is empty."""
    low = m & -m
    return torch.where(m != 0, DB[((low * _DBC) >> 58) & 63].long(), 64)


def _rev64(x):
    """The 64 bits in reverse order: the occupancy of the board turned by 180 degrees. Arithmetic right shifts are
    harmless here: every mask is zero where a shift brings in sign bits."""
    x = ((x >> 1) & 0x5555555555555555) | ((x & 0x5555555555555555) << 1)
    x = ((x >> 2) & 0x3333333333333333) | ((x & 0x3333333333333333) << 2)
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0F) | ((x & 0x0F0F0F0F0F0F0F0F) << 4)
    x = ((x >> 8) & 0x00FF00FF00FF00FF) | ((x & 0x00FF00FF00FF00FF) << 8)
    x = ((x >> 16) & 0x0000FFFF0000FFFF) | ((x & 0x0000FFFF0000FFFF) << 16)
    return ((x >> 32) & 0x00000000FFFFFFFF) | (x << 32)


def _dense_rows(occ, nb, PM, DB, KN, KG, PA, DIR_OK, FID, TRI):
    """indices()'s rules on every (piece, target) slot at once: all 32 pieces of a position and their 8 targets,
    for both perspectives, with -1 where a slot has no row. Returns int32 rows [2, n, S], the bool [2, n, S] of
    the slots that have one, and the rows a bag [2n]: S = 256 slots 8 * piece + target; with all pawn pairs (NNUE_REL_PPM=1) 256 more,
    16 * pawn i + pawn j, after them, which is the order the step-by-step version's stable sort gives (each bag's
    relation rows, then its pairs).

    Written to be compiled: torch.compile fuses it into a few kernels, so the [n, 32, 8] intermediates are never
    written out as the step-by-step version writes them (about 120 kernels and six host syncs, 6.6 ms at
    n = 131,072). The turned board's occupancy comes from a bit reversal instead of a flip and a sum; every other line
    is the step-by-step version's, on all slots instead of on the sliders or the attacks alone."""
    n = occ.shape[0]
    SQ64 = torch.arange(64, device=occ.device)
    bits = (occ.view(n, 1) >> SQ64) & 1                            # occupied squares, ascending; 64 = none
    slot = torch.where(bits.bool(), torch.cumsum(bits, 1) - 1, 32)
    sqs = torch.full((n, 33), 64, dtype=torch.long, device=occ.device).scatter(1, slot, SQ64.expand(n, 64))[:, :32]
    valid = sqs < 64
    nbl = nb.long()
    codes = torch.where(valid, torch.stack((nbl & 15, nbl >> 4), dim=2).view(n, 32), 12)
    pt, col = codes % 6, (codes // 6).clamp(max=1)
    wk = torch.gather(sqs, 1, (codes == 5).long().argmax(1, keepdim=True)).view(n)
    bk = torch.gather(sqs, 1, (codes == 11).long().argmax(1, keepdim=True)).view(n)
    mb = torch.full((n, 65), 12, dtype=torch.long, device=occ.device).scatter(1, sqs, codes)   # mailbox, 64 = none

    # every piece's first blockers in the order E N NE NW W S SW SE (64 = none), kept for the sliders' own lines
    fp = _first(occ.view(n, 1, 1) & PM[sqs], DB)
    fn = _first(_rev64(occ).view(n, 1, 1) & PM[torch.where(valid, 63 - sqs, 64)], DB)
    blk = torch.cat((fp, torch.where(fn < 64, 63 - fn, 64)), 2)
    slider = (valid & (pt >= B) & (pt <= Q)).unsqueeze(2)
    blk = torch.where(slider & DIR_OK[pt], blk, 64)

    pawn = (pt == P).unsqueeze(2)
    tgt = torch.where((pt == N).unsqueeze(2), KN[sqs].long(),
                      torch.where((pt == K).unsqueeze(2), KG[sqs].long(),
                                  torch.where(pawn, PA[col, sqs].long(), blk)))
    tgt = torch.where(valid.unsqueeze(2), tgt, 64)                 # [n, 32, 8]
    vcode = torch.gather(mb, 1, tgt.view(n, 256)).view(n, 32, 8)
    hit = vcode < 12
    pawn_on_pawn = pawn & ((vcode == 0) | (vcode == 6))
    if PPM:
        hit = hit & ~pawn_on_pawn                                  # every pawn pair comes from the pair slots
    else:
        hit = hit & ~(pawn_on_pawn & (col == 1).unsqueeze(2) & (vcode == 0))   # told by White's pawn
    if PPM:                                                        # the pawns' squares and colours, in slot order
        isp = valid & (pt == P)
        ps = torch.where(isp, torch.cumsum(isp.long(), 1) - 1, 16)
        psq = torch.full((n, 17), 64, dtype=torch.long, device=occ.device).scatter(1, ps, sqs)[:, :16]
        pcl = torch.zeros((n, 17), dtype=torch.long, device=occ.device).scatter(1, ps, col)[:, :16]
        pair_ok = (psq < 64).unsqueeze(2) & (psq < 64).unsqueeze(1) & TRI          # [n, 16, 16], i < j
        xi, xj, ci, cj = psq.unsqueeze(2), psq.unsqueeze(1), pcl.unsqueeze(2), pcl.unsqueeze(1)

    a_pt, a_col, a_sq = (codes % 6).unsqueeze(2), (codes // 6).unsqueeze(2), sqs.unsqueeze(2)
    v_pt, v_col = vcode % 6, vcode // 6
    rows = []
    for p, ksq in ((0, wk), (1, bk)):
        f = ((56 * p) ^ torch.where((ksq & 7) < 4, 7, 0)).view(n, 1, 1)
        xo, yo = (a_sq ^ f) & 63, (tgt ^ f) & 63
        sa, sv = (a_col != p).long(), (v_col != p).long()
        feat = FID[((((sa * 6 + a_pt) * 64 + xo) * 2 + sv) * 6 + v_pt) * 64 + yo]
        r = torch.where(hit & (feat >= 0), feat, -1).to(torch.int32).view(n, 256)
        if PPM:
            pxi, pxj = (xi ^ f) & 63, (xj ^ f) & 63
            si, sj = (ci != p).long(), (cj != p).long()
            pfeat = FID[((si * 384 + pxi) * 2 + sj) * 384 + pxj]
            r = torch.cat((r, torch.where(pair_ok & (pfeat >= 0), pfeat, -1).to(torch.int32).view(n, 256)), 1)
        rows.append(r)
    rows = torch.stack(rows)                                       # int32 throughout: no int64 copy of [2, n, S]
    ok = rows >= 0
    return rows, ok, ok.sum(2).view(-1)                            # the count inside the pass (a bool sum outside
                                                                   # it cost 0.95 ms a step)


_DENSE = {}


def _indices_fused(rec):
    tb = _tables(rec.device)
    n = rec.shape[0]
    fn = _DENSE.get("fn")
    if fn is None:
        fn = _DENSE["fn"] = torch.compile(_dense_rows, dynamic=False)
    occ = rec[:, 0:8].contiguous().view(torch.int64).view(n)
    rows, ok, cnt = fn(occ, rec[:, 8:24].contiguous(), tb["PM"], tb["DB"], tb["KN"], tb["KG"], tb["PA"],
                       tb["DIR_OK"], tb["FID"], tb["TRI"])
    S = rows.shape[2]
    flat = ok.view(-1).nonzero(as_tuple=True)[0]                   # ascending: perspective, position, slot
    return flat // S, rows.view(-1)[flat].long(), cnt


def indices(rec):
    """uint8 [n, 32] LCR1 records -> (bags, cols, count). bags are rows of a [2n, T] sum: position i's White
    perspective is bag i, its Black perspective bag n + i. cols are the table rows active in each bag (every row
    once). count [2n] is the number of active rows per bag.

    Written for memory traffic, which is all this costs on a GPU: the first version worked on int64 tensors of
    [n, 32, 8] throughout (268 MB each at n = 131,072, about 33 GB moved per batch, 40 ms). Here the board-wide
    tensors are bytes, the sliders' ray arithmetic runs on the sliders alone, and everything that depends on the
    perspective runs on the attacks that exist (about 20 a position) instead of on all 256 slots twice. The rows
    are the first version's, in the first version's order."""
    if FUSED and rec.is_cuda:
        return _indices_fused(rec)
    tb = _tables(rec.device)
    n, dv = rec.shape[0], rec.device
    SQ64 = tb["SQ64"]
    occ = rec[:, 0:8].contiguous().view(torch.int64).view(n)
    nb = rec[:, 8:24].to(torch.int64)
    codes = torch.stack((nb & 15, nb >> 4), dim=2).view(n, 32)
    bits = (occ.view(n, 1) >> SQ64) & 1
    slot = torch.where(bits.bool(), torch.cumsum(bits, 1) - 1, 32)
    sqs = torch.full((n, 33), 64, dtype=torch.int64, device=dv)
    sqs.scatter_(1, slot, SQ64.expand(n, 64))
    sqs = sqs[:, :32].contiguous()                                 # occupied squares in ascending order, 64 = none
    valid = SQ64[:32].view(1, 32) < bits.sum(1).view(n, 1)
    codes = torch.where(valid, codes, 12)                          # 0..5 white P..K, 6..11 black, 12 = none
    pt, col = codes % 6, (codes // 6).clamp(max=1)
    wk = torch.gather(sqs, 1, (codes == 5).long().argmax(1, keepdim=True)).view(n)
    bk = torch.gather(sqs, 1, (codes == 11).long().argmax(1, keepdim=True)).view(n)
    mb = torch.full((n, 65), 12, dtype=torch.uint8, device=dv)
    mb.scatter_(1, sqs, codes.to(torch.uint8))                     # mailbox; index 64 stays "none"

    # the sliders' first blockers, on the sliders alone: [S, 8] in the order E N NE NW W S SW SE
    def first(m):                                                  # the square of the lowest set bit, 64 if none
        low = m & -m
        return torch.where(m != 0, tb["DB"][((low * _DBC) >> 58) & 63], 64)

    sl = (valid & (pt >= B) & (pt <= Q)).view(-1).nonzero(as_tuple=True)[0]       # into [n * 32]
    spos, ssq = sl >> 5, sqs.view(-1)[sl]
    rocc = (bits.flip(1) << SQ64).sum(1)                           # the occupancy turned by 180 degrees
    fp = first(occ[spos].view(-1, 1) & tb["PM"][ssq])              # toward E, N, NE, NW
    fn = first(rocc[spos].view(-1, 1) & tb["PM"][63 - ssq])        # toward W, S, SW, SE, on the turned board
    sblk = torch.cat((fp, torch.where(fn < 64, 63 - fn, 64)), 1)
    sblk = torch.where(tb["DIR_OK"][pt.view(-1)[sl]], sblk, 64)
    blk = torch.full((n * 32, 8), 64, dtype=torch.uint8, device=dv)
    blk[sl] = sblk
    blk = blk.view(n, 32, 8)

    # what each piece attacks on the real board: up to 8 occupied squares, as bytes
    tgt = torch.where((pt == N).unsqueeze(2), tb["KN"][sqs],
                      torch.where((pt == K).unsqueeze(2), tb["KG"][sqs],
                                  torch.where((pt == P).unsqueeze(2), tb["PA"][col, sqs], blk)))
    tgt = torch.where(valid.unsqueeze(2), tgt, 64)                 # [n, 32, 8]
    vcode = torch.gather(mb, 1, tgt.view(n, 256).long()).view(n, 32, 8)
    hit = vcode < 12
    pawn_on_pawn = (pt == P).unsqueeze(2) & ((vcode == 0) | (vcode == 6))
    if PPM:
        hit = hit & ~pawn_on_pawn                                  # every pawn pair comes from the pair list below
    else:                                                          # two pawns that attack each other: told by White's
        hit = hit & ~(pawn_on_pawn & (col == 1).unsqueeze(2) & (vcode == 0))

    # the attacks that exist, once: position, attacker and victim
    flat = hit.view(-1).nonzero(as_tuple=True)[0]                  # into [n * 256], ascending
    pi, aj = flat >> 8, flat >> 3
    a_code, a_sq = codes.view(-1)[aj], sqs.view(-1)[aj]
    a_pt, a_col = a_code % 6, a_code // 6
    t_sq = tgt.view(-1)[flat].long()
    v_code = vcode.view(-1)[flat].long()
    v_pt, v_col = v_code % 6, v_code // 6
    if PPM:
        isp = valid & (pt == P)
        ps = torch.where(isp, torch.cumsum(isp.long(), 1) - 1, 16)
        psq = torch.full((n, 17), 64, dtype=torch.int64, device=dv)
        psq.scatter_(1, ps, sqs)
        pcl = torch.zeros((n, 17), dtype=torch.int64, device=dv)
        pcl.scatter_(1, ps, col)
        psq, pcl = psq[:, :16].contiguous(), pcl[:, :16].contiguous()
        pair_ok = (psq < 64).unsqueeze(2) & (psq < 64).unsqueeze(1) & tb["TRI"]
        pf = pair_ok.view(-1).nonzero(as_tuple=True)[0]            # into [n * 256]: position, pawn i, pawn j
        ppos, pii = pf >> 8, pf >> 4
        pjj = (ppos << 4) | (pf & 15)
        xi, xj = psq.view(-1)[pii], psq.view(-1)[pjj]
        ci, cj = pcl.view(-1)[pii], pcl.view(-1)[pjj]

    bags, cols = [], []
    for p, ksq in ((0, wk), (1, bk)):
        flip = (56 * p) ^ torch.where((ksq & 7) < 4, 7, 0)         # [n]: group 1's orientation
        f = flip[pi]
        xo, yo = (a_sq ^ f) & 63, (t_sq ^ f) & 63
        sa, sv = (a_col != p).long(), (v_col != p).long()
        feat = tb["FID"][((((sa * 6 + a_pt) * 64 + xo) * 2 + sv) * 6 + v_pt) * 64 + yo]
        ok = feat >= 0
        bags.append(pi[ok] + p * n)
        cols.append(feat[ok])
        if PPM:
            f = flip[ppos]
            pxi, pxj = (xi ^ f) & 63, (xj ^ f) & 63
            si, sj = (ci != p).long(), (cj != p).long()
            pfeat = tb["FID"][((si * 384 + pxi) * 2 + sj) * 384 + pxj]
            pok = pfeat >= 0
            bags.append(ppos[pok] + p * n)
            cols.append(pfeat[pok])
    bags, cols = torch.cat(bags), torch.cat(cols)
    if PPM:   # each perspective's pair rows follow its relation rows: into bag order (stable: relations, then pairs)
        order = torch.argsort(bags, stable=True)
        bags, cols = bags[order], cols[order]
    return bags, cols, counts(bags, 2 * n)


def counts(bags, nbags):
    """Active rows per bag, as torch.bincount(bags, minlength=nbags) gives them, without its host sync (bincount reads
    bags.max() back to size its output; every bag here is below nbags already)."""
    return torch.zeros(nbags, dtype=torch.int64, device=bags.device).index_add_(0, bags, torch.ones_like(bags))


# ---------------------------------------------------------------------------
# Rows -> lane sums
# ---------------------------------------------------------------------------
class _SpmmSum(torch.autograd.Function):
    """out = S W with S the 0/1 [bags, rows] incidence matrix; backward S^T G. Two sparse-dense products, nothing
    of size [active rows, T] in memory."""

    @staticmethod
    def forward(ctx, W, bags, cols, nbags):
        ones = torch.ones(len(cols), dtype=W.dtype, device=W.device)
        S = torch.sparse_coo_tensor(torch.stack((bags, cols)), ones, (nbags, W.shape[0]))
        ctx.save_for_backward(bags, cols)
        ctx.shape = (nbags, W.shape[0])
        return torch.sparse.mm(S, W)

    @staticmethod
    def backward(ctx, g):
        bags, cols = ctx.saved_tensors
        ones = torch.ones(len(cols), dtype=g.dtype, device=g.device)
        St = torch.sparse_coo_tensor(torch.stack((cols, bags)), ones, (ctx.shape[1], ctx.shape[0]))
        return torch.sparse.mm(St, g), None, None, None


def bag_sum(W, bags, cols, nbags, cnt=None):
    if IMPL == "spmm":
        return _SpmmSum.apply(W, bags, cols, nbags)
    if IMPL in ("triton", "trieb", "triseg", "trirows") and (W.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"):
        import rel_kernels   # next to this file; only this path needs triton
        if cnt is None:
            cnt = counts(bags, nbags)
        return rel_kernels.rel_sum(W, cols, cnt, nbags, bf16=TABLE16, eb_backward=IMPL == "trieb",
                                   seg_backward=IMPL in ("triseg", "trirows"))
    if IMPL in ("embag", "triton", "trieb", "triseg", "trirows"):
        # embedding_bag wants each bag's rows together: indices() returns them so, bag after bag (the step-by-step
        # version sorts the pair rows in with all pawn pairs; the test for unsorted rows was a host sync every step)
        if cnt is None:
            cnt = counts(bags, nbags)
        return torch.nn.functional.embedding_bag(cols, W, torch.cumsum(cnt, 0) - cnt, mode="sum")
    out = torch.zeros((nbags, W.shape[1]), dtype=W.dtype, device=W.device)
    for i in range(0, len(cols), CHUNK):
        out = out.index_add(0, bags[i:i + CHUNK], W.index_select(0, cols[i:i + CHUNK]))
    return out


# ---------------------------------------------------------------------------
# The trainer's hooks (patch_rel.py puts the calls in)
# ---------------------------------------------------------------------------
def check(qat, qat_after, ft_act, lanes=False):
    """lanes: the trainer's fused kernels take extra lane sums (ft_fused.py's lanes, the trainer's LANE_SUMS hook,
    which patch_rel.py sets to lane_sums()), so NNUE_FT_ACT=1 adds the relation rows inside them."""
    if not T:
        return
    if T % 64 or not 0 < T <= 768:
        raise SystemExit(f"FAIL: NNUE_REL_T={T}: a multiple of 64 up to 768")
    if qat_after:
        raise SystemExit("FAIL: NNUE_REL_T with NNUE_QAT_AFTER: run QAT as a phase of its own (NNUE_QAT=1) from the "
                         "float phase's checkpoint")
    _CTX["qat"] = bool(qat)
    if qat:
        print(f"[rel] QAT: the forward reads int8 rows (k / 128, |k| <= 127) and clamps each lane sum to "
              f"+-{LANE_L:.4f}", flush=True)
    if ft_act and not lanes:
        raise SystemExit("FAIL: NNUE_REL_T needs NNUE_FT_ACT=0 with this trainer: its fused activation kernels take no "
                         "second sum (run the control with NNUE_FT_ACT=0 too, so both arms take the same path)")
    print(f"[rel] relation inputs on: {N_ROWS:,} rows ({N_REL:,} relations + {N_PP:,} pawn pairs) into lanes 0..{T - 1}, "
          f"set {SET}, pawn pairs {'all' if PPM else 'in contact'}, lr x{LR_MULT:g}, sums by {IMPL}"
          f"{' (bf16 table)' if IMPL == 'triton' and TABLE16 else ''}, "
          f"{'inside the fused activation kernels' if ft_act else 'joined ahead of the activation'}", flush=True)


def _rows(model):
    """The relation table as the forward reads it: the float rows in the float phase; under QAT the engine's int8
    rows, k / 128 with |k| <= 127, the gradient passed straight through to the float rows."""
    w = model.rel
    if not _CTX.get("qat"):
        return w
    q = torch.round(w * 128.0).clamp(-127.0, 127.0) / 128.0
    return w + (q - w).detach()


def _clamp(s):
    """Under QAT each lane sum is clamped to +-511 steps (LANE_L), as the engine clamps it before the combination."""
    return s.clamp(-LANE_L, LANE_L) if _CTX.get("qat") else s


def attach(model, dev):
    """The relation table, zero: the net's output is the piece-square net's until the first step."""
    if T:
        model.rel = torch.nn.Parameter(torch.zeros((N_ROWS, T), device=dev))
        _CTX["model"] = model


def _rel_sums(model, rec):
    """The relation rows' sums [2n, T] fp32 for the records, with their autograd history to the table."""
    if IMPL == "trirows" and rec.is_cuda and not PPM:
        import rel_kernels   # next to this file, with rel_rows.py
        import rel_rows
        rows, cnt = rel_rows.rel_rows(rec)
        return rel_kernels.rel_sum_rows(_rows(model), rows, cnt, 2 * rec.shape[0], bf16=TABLE16, g16=G16)
    bags, cols, cnt = indices(rec)
    return bag_sum(_rows(model), bags, cols, 2 * rec.shape[0], cnt)


GS16 = os.environ.get("NNUE_REL_GS16", "0") == "1"   # see lane_sums (the float phase, trirows only)
GS16_PENDING = []   # (model, rows, cnt, nnz, ev): the batch's rows for FusedAct.backward's rows_grad16


@torch.compiler.disable
def lane_sums():
    """The relation rows' sums [2n, T] fp32 for the records expand() last saw, in the fused kernels' side-row order
    (bag i = White's view of position i, bag n + i Black's, as indices() numbers them), with their autograd history
    to the table: with NNUE_FT_ACT the kernels add them to lanes 0..T-1 after the bias and before SCReLU."""
    rec = _CTX["rec"]
    if rec is None:
        raise RuntimeError("rel_gate.lane_sums: a forward() without the expand() of its records")
    with torch.autocast(device_type=rec.device.type, enabled=False):
        model = _CTX["model"]
        if GS16 and not _CTX.get("qat") and IMPL == "trirows" and rec.is_cuda and not PPM:
            # NNUE_REL_GS16: the sums without autograd; FusedAct.backward takes the table's gradient from gs
            import rel_kernels
            import rel_rows
            rows, cnt = rel_rows.rel_rows(rec)
            with torch.no_grad():
                out, ctx = rel_kernels.rel_sum_rows_nograd(model.rel, rows, cnt, 2 * rec.shape[0], bf16=TABLE16)
            if torch.is_grad_enabled():
                GS16_PENDING.clear()
                GS16_PENDING.append((model,) + ctx)
            return out
        return _clamp(_rel_sums(model, rec))


def group(model, lr):
    return {"params": [model.rel], "weight_decay": 0.0, "lr": lr * LR_MULT}


@torch.compiler.disable
def note(rec):
    """expand() tells which records the next forward() belongs to."""
    if T:
        _CTX["rec"] = rec


class _LaneJoin(torch.autograd.Function):
    """acc_w[:, :T] += s[:n] and acc_b[:, :T] += s[n:], in place. Both sums are fresh tensors (the feature layer's
    output plus its bias) that no backward has saved, so writing into them is safe; autograd would refuse at the
    backward if one had been saved. The gradients pass through untouched and the relation sums' gradient is lanes
    0..T-1 of both, so forward and backward are the cat join's values bit for bit.

    The join this replaces, torch.cat((acc[:, :T] + s, acc[:, T:]), 1) on each side, wrote both [n, L1] sums again in
    the forward and, in the backward, a zero-padded full-size copy of the gradient for each of its six slices, then
    added them up: at n = 131,072 and L1 = 768, 4.7 ms a step in SliceBackward0 and about 3.4 ms more in the cats
    and the gradient sums, of A128's 36 ms."""

    @staticmethod
    def forward(ctx, acc_w, acc_b, s):
        n = acc_w.shape[0]
        acc_w[:, :T] += s[:n]
        acc_b[:, :T] += s[n:]
        ctx.mark_dirty(acc_w, acc_b)
        return acc_w, acc_b

    @staticmethod
    def backward(ctx, g_w, g_b):
        return g_w, g_b, torch.cat((g_w[:, :T], g_b[:, :T]), 0)


@torch.compiler.disable
def add(model, acc_w, acc_b):
    """Both sides' feature-layer sums (bias in, activation not yet applied) plus the relation rows' sums on lanes
    0..T-1, for the records expand() last saw."""
    rec = _CTX["rec"]
    n = acc_w.shape[0]
    if rec is None or rec.shape[0] != n:
        raise RuntimeError("rel_gate.add: forward() without the expand() of its records")
    # The trainer calls forward() under autocast(bf16) on CUDA. The sums are taken with it off: a sparse product
    # under autocast is cast to bf16 (measured: an error of 0.08 on sums of this size), and both ways of summing
    # must give the same fp32 sums.
    with torch.autocast(device_type=acc_w.device.type, enabled=False):
        s = _clamp(_rel_sums(model, rec))
    if JOIN == "inplace" and acc_w.dtype == s.dtype == acc_b.dtype:
        return _LaneJoin.apply(acc_w, acc_b, s)
    s = s.to(acc_w.dtype)
    acc_w = torch.cat((acc_w[:, :T] + s[:n], acc_w[:, T:]), 1)
    acc_b = torch.cat((acc_b[:, :T] + s[n:], acc_b[:, T:]), 1)
    return acc_w, acc_b


@torch.no_grad()
def clip(model):
    if T:
        model.rel.clamp_(-CLIP, CLIP)


OPT = os.environ.get("NNUE_REL_OPT", "torch")   # torch | fused | fused16: see step()
if OPT not in ("torch", "fused", "fused16"):
    raise SystemExit(f"FAIL: NNUE_REL_OPT={OPT}")


@torch.no_grad()
def step(model, opt, params, max_norm, ft=None, ft_rows=None):
    """clip_grad_norm_(params, max_norm), opt.step() and clip(model), with the relation table's share in one pass
    (NNUE_REL_OPT=fused, rel_kernels.rel_adamw): the total norm over every gradient, the other gradients scaled by
    torch as before, the table's gradient scaled, AdamW-updated and clamped by one kernel, the other groups'
    AdamW by torch (the table's gradient set aside, so torch's step leaves it alone). The same values as the three
    calls, bit for bit; it saves the table's separate scale, update and clamp passes.
    NNUE_REL_OPT=fused16 (rel_kernels.adamw16): the same for the relation table and, given ft (the feature layer's
    table), for that table too, each in one pass that also writes the bf16 copy of the table (ft: of its first
    ft_rows rows) that the next forward reads instead of casting it (rel_kernels.bf16_copy); the other groups'
    AdamW by torch on the gradients torch scaled. The same values bit for bit as the three calls and the casts."""
    if OPT == "fused16" and getattr(model, "rel", None) is not None and model.rel.is_cuda:
        return _step16(model, opt, params, max_norm, ft, ft_rows)
    w = getattr(model, "rel", None) if T and OPT == "fused" else None
    g = w.grad if w is not None else None
    if g is None or not w.is_cuda:
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm)
        opt.step()
        clip(model)
        return
    import rel_kernels
    if max_norm > 0:
        total = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None], 2.0, False, None)
        torch.nn.utils.clip_grads_with_norm_([p for p in params if p is not w], max_norm, total, None)
        coef = torch.clamp(float(max_norm) / (total + 1e-6), max=1.0)
    else:
        coef = torch.ones((), dtype=torch.float32, device=w.device)
    group = next(gr for gr in opt.param_groups if any(p is w for p in gr["params"]))
    if group["weight_decay"] != 0 or not group["amsgrad"] or group.get("maximize"):
        raise SystemExit("FAIL: NNUE_REL_OPT=fused: the table's group must be amsgrad without weight decay")
    w.grad = None
    opt.step()
    rel_kernels.rel_adamw(w, g, opt.state[w], coef, group["lr"], group["betas"], group["eps"], -CLIP, CLIP)


def _step16(model, opt, params, max_norm, ft, ft_rows):
    import rel_kernels
    tabs = [(model.rel, (-CLIP, CLIP), None)]   # (table, clamp, rows of its bf16 copy: None = all)
    if ft is not None and ft.grad is not None:
        tabs.append((ft, None, ft_rows))
    tabs = [t for t in tabs if t[0].grad is not None]
    if max_norm > 0:   # clip_grad_norm_'s total over every gradient, in its order; its coefficient, as it computes it
        total = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None], 2.0, False, None)
        torch.nn.utils.clip_grads_with_norm_([p for p in params if all(p is not t[0] for t in tabs)], max_norm,
                                             total, None)
        coef = torch.clamp(float(max_norm) / (total + 1e-6), max=1.0)
    else:
        coef = torch.ones((), dtype=torch.float32, device=model.rel.device)
    work = []
    for w, clamp, rows in tabs:
        group = next(gr for gr in opt.param_groups if any(p is w for p in gr["params"]))
        if group["weight_decay"] != 0 or not group["amsgrad"] or group.get("maximize") or not group.get("fused"):
            raise SystemExit("FAIL: NNUE_REL_OPT=fused16: the tables' groups must be fused amsgrad AdamW without "
                             "weight decay")
        work.append((w, w.grad, group, clamp, rows))
        w.grad = None   # torch's step leaves the table alone
    opt.step()
    for w, g, group, clamp, rows in work:
        rel_kernels.adamw16(w, g, opt.state[w], coef, group["lr"], group["betas"], group["eps"], clamp,
                            w.shape[0] if rows is None else rows)
    # the tables' gradients stay dropped, as step()'s fused path leaves the relation table's: nothing reads a
    # gradient after the step, and zero_grad(set_to_none=True) would drop them before the next backward


@torch.no_grad()
def health(model, gstep, rec=None):
    """A [REL] line: the table (size of its weights, how many sit at the clip, rows still zero) and, on `rec`
    (held-out records), the active rows a perspective and the lane sums."""
    if not T:
        return
    w = model.rel
    a = w.abs()
    line = (f"[REL step {gstep}] table |w| max {float(a.max()):.3f} mean {float(a.mean()):.5f}, "
            f"at the clip {float((a >= CLIP - 1e-6).float().mean()):.4%}, rows still zero "
            f"{float((a.amax(1) == 0).float().mean()):.1%}")
    if rec is not None and rec.shape[0] > 0:
        n = rec.shape[0]
        bags, cols, cnt = indices(rec)
        if not _CTX.get("checked") and rec.device.type != "cpu":
            # once per run: the device's rows against the CPU's for the same records (the CPU path is the one
            # check_rel_gate.py compares with the reference script)
            m = min(n, 8192)
            b1, c1, n1 = indices(rec[:m])
            b0, c0, n0 = indices(rec[:m].cpu())
            k1 = torch.sort(b1.cpu() * N_ROWS + c1.cpu()).values
            k0 = torch.sort(b0 * N_ROWS + c0).values
            same = bool(torch.equal(k1, k0)) and bool(torch.equal(n1.cpu(), n0))
            print(f"[rel] rows on {rec.device.type} against the CPU's, {m:,} positions x 2 perspectives: "
                  f"{'equal' if same else 'DIFFERENT'} ({len(k0):,} rows)", flush=True)
            if not same:
                raise SystemExit("FAIL: rel_gate: the device computes other rows than the CPU")
            _CTX["checked"] = True
        s = bag_sum(w, bags, cols, 2 * n).abs()
        line += (f" | active rows a perspective {float(cnt.float().mean()):.2f} (max {int(cnt.max())}) | "
                 f"lane sum |s| mean {float(s.mean()):.4f} max {float(s.max()):.3f}, beyond {LANE_L:.2f}: "
                 f"{float((s > LANE_L).float().mean()):.5%}")
    print(line, flush=True)
