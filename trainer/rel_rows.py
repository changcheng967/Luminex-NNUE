"""The V5 relation rows straight from the LCR1 records in one Triton kernel (main session, 2026-11-20), for
rel_gate.indices()'s pawn pairs in contact (NNUE_REL_PPM=0).

rel_gate._indices_fused runs a compiled dense pass ([2, n, 256] int32 rows with -1 for the empty slots, written and
read back: 1.2 ms at n = 131,072), then nonzero (a host sync) and gathers to compact them. Here one program takes
BLOCK positions and computes the same rows in registers: the pieces' squares from the occupancy (the k-th set bit),
their codes from the nibbles, every piece's 8 targets (the knight, king and pawn tables; a slider's first blocker
by the lowest set bit of occupancy & ray, the other four rays on the bit-reversed board, as rel_gate does), the
victim's code by its rank among the occupied squares, and each perspective's feature row from rel_gate's FID
table, read only for real attacks. Each bag's rows go out compacted in slot order (8 * piece + target), the order
rel_gate's flat nonzero gives them, into a [2n, 256] buffer: rows[b, :cnt[b]].

compact(rows, cnt, nnz) gives rel_gate's (bags, cols) from that, for any consumer that wants them flat."""
import os

import torch
import triton
import triton.language as tl

import rel_gate as RG

DBC = 0x03F79D71B4CB0A89   # rel_gate._DBC


@triton.jit
def _popc64(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((x * 0x0101010101010101) >> 56) & 0xFF


@triton.jit
def _rev64(x):
    x = ((x >> 1) & 0x5555555555555555) | ((x & 0x5555555555555555) << 1)
    x = ((x >> 2) & 0x3333333333333333) | ((x & 0x3333333333333333) << 2)
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0F) | ((x & 0x0F0F0F0F0F0F0F0F) << 4)
    x = ((x >> 8) & 0x00FF00FF00FF00FF) | ((x & 0x00FF00FF00FF00FF) << 8)
    x = ((x >> 16) & 0x0000FFFF0000FFFF) | ((x & 0x0000FFFF0000FFFF) << 16)
    return ((x >> 32) & 0x00000000FFFFFFFF) | (x << 32)


@triton.jit
def _rel_rows(rec, rec64, PM, DB, KN, KG, PA, DIROK, FID, rows, cnt, n, DBC_C: tl.constexpr,
              BLOCK: tl.constexpr):
    pos = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)          # [B]
    pm = pos < n
    occ = tl.load(rec64 + pos * 4, mask=pm, other=0)                                # int64, bit s = square s
    rocc = _rev64(occ)
    k = tl.arange(0, 32)
    # the k-th set bit of occ: the square of piece slot k (64 = no piece)
    sqs = tl.full((BLOCK, 32), 64, tl.int64)
    c = tl.zeros((BLOCK,), tl.int64)
    for s in tl.static_range(64):
        b = (occ >> s) & 1
        sqs = tl.where((b[:, None] == 1) & (k[None, :].to(tl.int64) == c[:, None]), s, sqs)
        c = c + b
    valid = k[None, :].to(tl.int64) < c[:, None]                                    # [B, 32]
    byte = tl.load(rec + pos[:, None] * 32 + 8 + (k // 2)[None, :], mask=pm[:, None], other=0).to(tl.int64)
    code = tl.where(valid, (byte >> (4 * (k & 1))[None, :].to(tl.int64)) & 15, 12)  # 0..11, 12 = none
    pt = code % 6
    col = tl.minimum(code // 6, 1)
    acol = code // 6
    # the kings' squares as argmax gives them: the first slot holding the king, slot 0 if none
    wi = tl.min(tl.where(code == 5, k[None, :], 32), axis=1)
    bi = tl.min(tl.where(code == 11, k[None, :], 32), axis=1)
    wi = tl.where(wi == 32, 0, wi)
    bi = tl.where(bi == 32, 0, bi)
    wk = tl.sum(tl.where(k[None, :] == wi[:, None], sqs, 0), axis=1)
    bk = tl.sum(tl.where(k[None, :] == bi[:, None], sqs, 0), axis=1)

    t = tl.arange(0, 8)
    sq3 = sqs[:, :, None]                                                         # [B, 32, 1]
    pt3 = pt[:, :, None]
    col3 = col[:, :, None]
    valid3 = valid[:, :, None]
    t3 = t[None, None, :]                                                         # [1, 1, 8]
    kn = tl.load(KN + sq3 * 8 + t3).to(tl.int64)                                   # [B, 32, 8]
    kg = tl.load(KG + sq3 * 8 + t3).to(tl.int64)
    pa = tl.load(PA + (col3 * 65 + sq3) * 8 + t3).to(tl.int64)
    # sliders: E N NE NW on the board, W S SW SE as the first four on the board turned by 180 degrees
    posd = t3 < 4
    psq = tl.where(posd, sq3, tl.where(sq3 < 64, 63 - sq3, 64))
    ray = tl.load(PM + psq * 4 + (t3 % 4))
    occ3 = tl.where(posd, occ[:, None, None], rocc[:, None, None])
    m = occ3 & ray
    low = m & (-m)
    idx = tl.load(DB + (((low * DBC_C) >> 58) & 63)).to(tl.int64)
    first = tl.where(m != 0, idx, 64)
    blk = tl.where(posd, first, tl.where(first < 64, 63 - first, 64))
    slider = valid3 & (pt3 >= 2) & (pt3 <= 4)
    dirok = tl.load(DIROK + pt3 * 8 + t3)
    blk = tl.where(slider & (dirok != 0), blk, 64)
    tgt = tl.where(pt3 == 1, kn, tl.where(pt3 == 5, kg, tl.where(pt3 == 0, pa, blk)))
    tgt = tl.where(valid3, tgt, 64)
    # the victim: its code by its rank among the occupied squares
    t63 = tgt & 63
    occd = (tgt < 64) & (((occ[:, None, None] >> t63) & 1) == 1)
    below = occ[:, None, None] & ((tl.full(tgt.shape, 1, tl.int64) << t63) - 1)
    vslot = _popc64(below)
    vbyte = tl.load(rec + pos[:, None, None] * 32 + 8 + vslot // 2, mask=occd & pm[:, None, None], other=0).to(tl.int64)
    vcode = tl.where(occd, (vbyte >> (4 * (vslot & 1))) & 15, 12)
    hit = vcode < 12
    pawn_on_pawn = (pt3 == 0) & ((vcode == 0) | (vcode == 6))
    hit = hit & ~(pawn_on_pawn & (col3 == 1) & (vcode == 0))                     # told by White's pawn
    a_pt3 = pt3
    a_col3 = acol[:, :, None]
    v_pt = vcode % 6
    v_col = vcode // 6
    for p in tl.static_range(2):
        if p == 0:
            ksq = wk
        else:
            ksq = bk
        f = (56 * p) ^ tl.where((ksq & 7) < 4, 7, 0)
        f3 = f[:, None, None]
        xo = (sq3 ^ f3) & 63
        yo = (tgt ^ f3) & 63
        sa = (a_col3 != p).to(tl.int64)
        sv = (v_col != p).to(tl.int64)
        fidx = ((((sa * 6 + a_pt3) * 64 + xo) * 2 + sv) * 6 + v_pt) * 64 + yo
        feat = tl.load(FID + fidx, mask=hit & pm[:, None, None], other=-1)
        r = tl.reshape(tl.where(hit & (feat >= 0), feat, -1), (BLOCK, 256))
        ok = r >= 0
        rank = tl.cumsum(ok.to(tl.int32), axis=1) - 1
        bag = p * n + pos
        tl.store(rows + bag[:, None] * 256 + rank, r.to(tl.int32), mask=ok & pm[:, None])
        tl.store(cnt + bag, tl.sum(ok.to(tl.int32), axis=1), mask=pm)


try:   # the CUDA libdevice functions (population count, bit reversal, find first set)
    from triton.language.extra import libdevice
except ImportError:   # older Triton
    from triton.language.extra.cuda import libdevice


@triton.jit
def _sel_step(x, r, base, W: tl.constexpr):
    # one halving of the search for the (r+1)-th set bit of x: above the low W bits when they hold r or fewer
    c = libdevice.popc(x & ((1 << W) - 1))
    up = r >= c
    return tl.where(up, x >> W, x), tl.where(up, r - c, r), tl.where(up, base + W, base)


@triton.jit
def _rel_rows2(rec, rec64, PM, KN, KG, PA, DIROK, FID32, rows, cnt, n, BLOCK: tl.constexpr):
    # _rel_rows's rows and counts, the same integers by other means: the pieces' squares by a binary search on
    # population counts (six halvings) instead of a 64-step scan, the hardware's population count, bit reversal and
    # find-first-set in place of the shift-and-mask versions and the de Bruijn lookup, int32 where a value fits, and
    # an int32 copy of the feature table
    pos = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)          # [B]
    pm = pos < n
    occ = tl.load(rec64 + pos * 4, mask=pm, other=0)                                # int64, bit s = square s
    rocc = libdevice.brev(occ)
    npc = libdevice.popc(occ)                                                       # int32
    k = tl.arange(0, 32)
    # the k-th set bit of occ: the square of piece slot k (64 = no piece)
    r = tl.zeros((BLOCK, 32), tl.int32) + k[None, :]
    x = tl.zeros((BLOCK, 32), tl.int64) + occ[:, None]
    base = tl.zeros((BLOCK, 32), tl.int32)
    x, r, base = _sel_step(x, r, base, 32)
    x, r, base = _sel_step(x, r, base, 16)
    x, r, base = _sel_step(x, r, base, 8)
    x, r, base = _sel_step(x, r, base, 4)
    x, r, base = _sel_step(x, r, base, 2)
    x, r, base = _sel_step(x, r, base, 1)
    valid = k[None, :] < npc[:, None]                                               # [B, 32]
    sqs = tl.where(valid, base, 64)
    byte = tl.load(rec + pos[:, None] * 32 + 8 + (k // 2)[None, :], mask=pm[:, None], other=0).to(tl.int32)
    code = tl.where(valid, (byte >> (4 * (k & 1))[None, :]) & 15, 12)              # 0..11, 12 = none
    pt = code % 6
    col = tl.minimum(code // 6, 1)
    acol = code // 6
    # the kings' squares as argmax gives them: the first slot holding the king, slot 0 if none
    wi = tl.min(tl.where(code == 5, k[None, :], 32), axis=1)
    bi = tl.min(tl.where(code == 11, k[None, :], 32), axis=1)
    wi = tl.where(wi == 32, 0, wi)
    bi = tl.where(bi == 32, 0, bi)
    wk = tl.sum(tl.where(k[None, :] == wi[:, None], sqs, 0), axis=1)
    bk = tl.sum(tl.where(k[None, :] == bi[:, None], sqs, 0), axis=1)

    t = tl.arange(0, 8)
    sq3 = sqs[:, :, None]                                                         # [B, 32, 1]
    pt3 = pt[:, :, None]
    col3 = col[:, :, None]
    valid3 = valid[:, :, None]
    t3 = t[None, None, :]                                                         # [1, 1, 8]
    kn = tl.load(KN + sq3 * 8 + t3).to(tl.int32)                                   # [B, 32, 8]
    kg = tl.load(KG + sq3 * 8 + t3).to(tl.int32)
    pa = tl.load(PA + (col3 * 65 + sq3) * 8 + t3).to(tl.int32)
    # sliders: E N NE NW on the board, W S SW SE as the first four on the board turned by 180 degrees
    posd = t3 < 4
    psq = tl.where(posd, sq3, tl.where(sq3 < 64, 63 - sq3, 64))
    ray = tl.load(PM + psq * 4 + (t3 % 4))
    occ3 = tl.where(posd, occ[:, None, None], rocc[:, None, None])
    m = occ3 & ray
    first = tl.where(m != 0, libdevice.ffs(m) - 1, 64)
    blk = tl.where(posd, first, tl.where(first < 64, 63 - first, 64))
    slider = valid3 & (pt3 >= 2) & (pt3 <= 4)
    dirok = tl.load(DIROK + pt3 * 8 + t3)
    blk = tl.where(slider & (dirok != 0), blk, 64)
    tgt = tl.where(pt3 == 1, kn, tl.where(pt3 == 5, kg, tl.where(pt3 == 0, pa, blk)))
    tgt = tl.where(valid3, tgt, 64)
    # the victim: its code by its rank among the occupied squares
    t63 = (tgt & 63).to(tl.int64)
    occd = (tgt < 64) & (((occ[:, None, None] >> t63) & 1) == 1)
    below = occ[:, None, None] & ((tl.full(t63.shape, 1, tl.int64) << t63) - 1)
    vslot = libdevice.popc(below)
    vbyte = tl.load(rec + pos[:, None, None] * 32 + 8 + vslot // 2, mask=occd & pm[:, None, None],
                    other=0).to(tl.int32)
    vcode = tl.where(occd, (vbyte >> (4 * (vslot & 1))) & 15, 12)
    hit = vcode < 12
    pawn_on_pawn = (pt3 == 0) & ((vcode == 0) | (vcode == 6))
    hit = hit & ~(pawn_on_pawn & (col3 == 1) & (vcode == 0))                     # told by White's pawn
    a_col3 = acol[:, :, None]
    v_pt = vcode % 6
    v_col = vcode // 6
    for p in tl.static_range(2):
        if p == 0:
            ksq = wk
        else:
            ksq = bk
        f = (56 * p) ^ tl.where((ksq & 7) < 4, 7, 0)
        f3 = f[:, None, None]
        xo = (sq3 ^ f3) & 63
        yo = (tgt ^ f3) & 63
        sa = (a_col3 != p).to(tl.int32)
        sv = (v_col != p).to(tl.int32)
        fidx = ((((sa * 6 + pt3) * 64 + xo) * 2 + sv) * 6 + v_pt) * 64 + yo
        feat = tl.load(FID32 + fidx, mask=hit & pm[:, None, None], other=-1)
        rr = tl.reshape(tl.where(hit & (feat >= 0), feat, -1), (BLOCK, 256))
        ok = rr >= 0
        rank = tl.cumsum(ok.to(tl.int32), axis=1) - 1
        bag = p * n + pos
        tl.store(rows + bag[:, None] * 256 + rank, rr, mask=ok & pm[:, None])
        tl.store(cnt + bag, tl.sum(ok.to(tl.int32), axis=1), mask=pm)


_TB = {}


def tables(dev):
    t = _TB.get(str(dev))
    if t is None:
        g = RG._tables(dev)
        t = dict(PM=g["PM"].contiguous(), DB=g["DB"].contiguous(), KN=g["KN"].contiguous(), KG=g["KG"].contiguous(),
                 PA=g["PA"].contiguous(), DIROK=g["DIR_OK"].to(torch.uint8).contiguous(), FID=g["FID"].contiguous())
        if int(t["FID"].abs().max()) >= 2 ** 31:
            raise SystemExit("FAIL: rel_rows: feature rows beyond int32")
        t["FID32"] = t["FID"].to(torch.int32).contiguous()
        _TB[str(dev)] = t
    return t


ROWS2 = os.environ.get("NNUE_REL_ROWS2", "0") == "1"   # 1: _rel_rows2, the same integers
ROWS2_CFG = (8, 4)   # _rel_rows2's BLOCK x num_warps: 0.296 ms at n = 131,072 against _rel_rows's 0.434 (box 8)


def rel_rows(rec, BLOCK=8, num_warps=4, v2=None):   # BLOCK 8 x 4 warps: 0.401 ms at n = 131,072 (4 x 4: 0.444)
    """uint8 [n, 32] LCR1 records -> rows [2n, 256] int32 (bag b's rows in rows[b, :cnt[b]], in slot order) and
    cnt [2n] int32. Pawn pairs in contact only (NNUE_REL_PPM=0). v2 (None: NNUE_REL_ROWS2; True; or (BLOCK,
    num_warps)): _rel_rows2, the same rows and counts with fewer instructions."""
    if RG.PPM:
        raise ValueError("rel_rows: NNUE_REL_PPM=0 only")
    n = rec.shape[0]
    tb = tables(rec.device)
    rec = rec.contiguous()
    rows = torch.empty((2 * n, 256), dtype=torch.int32, device=rec.device)
    cnt = torch.empty(2 * n, dtype=torch.int32, device=rec.device)
    if ROWS2 if v2 is None else v2 is not False:
        B2, W2 = ROWS2_CFG if v2 is None or v2 is True else v2
        _rel_rows2[(triton.cdiv(n, B2),)](rec, rec.view(torch.int64), tb["PM"], tb["KN"], tb["KG"], tb["PA"],
                                          tb["DIROK"], tb["FID32"], rows, cnt, n, BLOCK=B2, num_warps=W2)
        return rows, cnt
    _rel_rows[(triton.cdiv(n, BLOCK),)](rec, rec.view(torch.int64), tb["PM"], tb["DB"], tb["KN"], tb["KG"], tb["PA"],
                                        tb["DIROK"], tb["FID"], rows, cnt, n, DBC_C=DBC, BLOCK=BLOCK,
                                        num_warps=num_warps)
    return rows, cnt


def compact(rows, cnt):
    """rel_gate.indices()'s (bags, cols) from rel_rows()'s output: bag-major, slot order (a host sync: for checks)."""
    keep = torch.arange(rows.shape[1], device=rows.device)[None, :] < cnt[:, None]
    cols = rows[keep].long()
    bags = torch.arange(rows.shape[0], device=rows.device).repeat_interleave(cnt.long())
    return bags, cols
