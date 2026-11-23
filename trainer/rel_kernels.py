"""Triton kernels for the V5 relation rows' sums (rel_gate.py, NNUE_REL_IMPL=triton): the same [2n, T] fp32 sums as
embedding_bag(mode="sum") over the bags indices() returns, and their table gradient.

    forward:  out[b, :] = sum over k < cnt[b] of table[cols[off[b] + k], :]     (k in order, fp32)
    backward: dW[cols[off[b] + k], :] += g[b, :]                                 (fp32 atomics into a zeroed table)

The bags come sorted and each bag's rows together (indices() guarantees it), so the bags are a CSR: off = the running
count. A program takes BB bags x BT lanes: the forward gathers each bag's rows from the table (55,960 rows: 57 MB at
T = 256, held in the RTX PRO 6000's L2) and adds them in k order, the order embedding_bag adds them, so an fp32 table
gives its sums bit for bit; the backward loads each bag's gradient row once and adds it into every row the bag used.

The forward can read a bf16 copy of the table instead (rel_sum(bf16=True), NNUE_REL_TABLE16=1: half the bytes). That
is exact under QAT, where every row is k / 128 with |k| <= 127, and rounds each row to 8 significant bits otherwise,
as the feature layer's bf16 forward rounds its table. The gradient is always fp32, into the fp32 table.

The backward's atomics are the one thing a CPU check cannot price: rows that many bags share (a pawn defending a pawn)
take many adds at the same addresses. The embedding_bag backward (sort, then segment sums) is the alternative to keep
until box 1 has timed both (lanes_gpu_box1.py).

Box 1 timed them (2026-11-18, 131,072 positions, T = 256): the gather forward 0.60 ms against embedding_bag's 3.10,
the atomic backward 5.63 ms against embedding_bag's 3.97 (1.28G fp32 atomics: their throughput, not one hot address,
is the limit), and the atomics add in a varying order (2.1e-6 relative on a random gradient, run to run). So
rel_sum(eb_backward=True), NNUE_REL_IMPL=trieb: the gather forward with embedding_bag's own backward
(aten._embedding_bag_dense_backward, fed the bag of every index as embedding_bag's forward would have computed it):
both directions bit for bit what F.embedding_bag gives, deterministic, and the forward's 2.5 ms saved.

rel_sum(seg_backward=True), NNUE_REL_IMPL=triseg: the gather forward with a sorted backward of its own, in a fixed
order and without atomics. The entries are sorted by table row (one stable sort); each row's entries are cut into
pieces of at most CS; a program sums one piece's gradient rows over BTB lanes (tl.sum: a fixed tree) and stores the
sum where the row has one piece, or as a partial that a second kernel adds up, piece after piece, for the rows with
more. The lane block is the grid's slow axis, so the gradient's columns of one lane block (262,144 bags x 64 lanes x
4 bytes = 67 MB at the gate's batch) stay in the L2 while every piece gathers from them, as the forward gathers from
the table. Deterministic; not embedding_bag's order of additions, so equal to it bit for bit only where the sums are
exact (a gradient of whole multiples of 1/1024), and within rounding elsewhere."""
import os

import torch
import triton
import triton.language as tl

SEG_LIST = os.environ.get("NNUE_REL_SEGLIST", "0") == "1"   # seg_grad's join over the multi-piece rows only
EVICT = os.environ.get("NNUE_EVICT", "0") == "1"   # L2 hints in _rel_fwd and _seg_piece: the same values
LANE16 = os.environ.get("NNUE_REL_LANE16", "0") == "1"   # rel_sum_rows_nograd's sums in bf16 (with NNUE_REL_GS16)
JNP = 1024   # _seg_join_list's programs a lane block (2,048 to 32,768 timed no faster on a t91 batch)

BB = 32    # bags a program
BT = 128   # lanes a program
CS = 64    # sorted entries a piece (seg backward)
BTB = 64   # lanes a program (seg backward)


@triton.jit
def _rel_fwd(table, cols, off, cnt, out, nbags, T: tl.constexpr, BB: tl.constexpr, BT: tl.constexpr,
             EVR: tl.constexpr = "", EVS: tl.constexpr = ""):
    pb = tl.program_id(0)
    pt = tl.program_id(1)
    b = (pb * BB + tl.arange(0, BB)).to(tl.int64)
    bm = b < nbags
    o = tl.load(off + b, mask=bm, other=0)
    c = tl.load(cnt + b, mask=bm, other=0)
    lanes = pt * BT + tl.arange(0, BT)
    lm = lanes < T
    acc = tl.zeros((BB, BT), dtype=tl.float32)
    for k in range(0, tl.max(c, axis=0)):   # the block's longest bag; shorter bags are masked off
        km = bm & (k < c)
        col = tl.load(cols + o + k, mask=km, other=0).to(tl.int64)
        v = tl.load(table + col[:, None] * T + lanes[None, :], mask=km[:, None] & lm[None, :], other=0.0,
                    eviction_policy=EVR)
        acc += v.to(tl.float32)
    tl.store(out + b[:, None] * T + lanes[None, :], acc.to(out.dtype.element_ty), mask=bm[:, None] & lm[None, :],
             eviction_policy=EVS)


@triton.jit
def _rel_bwd(g, cols, off, cnt, dw, nbags, T: tl.constexpr, BB: tl.constexpr, BT: tl.constexpr):
    pb = tl.program_id(0)
    pt = tl.program_id(1)
    b = (pb * BB + tl.arange(0, BB)).to(tl.int64)
    bm = b < nbags
    o = tl.load(off + b, mask=bm, other=0)
    c = tl.load(cnt + b, mask=bm, other=0)
    lanes = pt * BT + tl.arange(0, BT)
    lm = lanes < T
    gv = tl.load(g + b[:, None] * T + lanes[None, :], mask=bm[:, None] & lm[None, :], other=0.0)
    for k in range(0, tl.max(c, axis=0)):
        km = bm & (k < c)
        col = tl.load(cols + o + k, mask=km, other=0).to(tl.int64)
        tl.atomic_add(dw + col[:, None] * T + lanes[None, :], gv, mask=km[:, None] & lm[None, :], sem="relaxed")


class RelSum(torch.autograd.Function):
    """[nbags, T] fp32 sums of w's rows over the CSR bags (cols int32, off and cnt int64); backward: w's fp32 gradient.
    table is what the forward reads: w itself (detached), or its bf16 copy."""

    @staticmethod
    def forward(ctx, w, table, cols, off, cnt, nbags):
        T = w.shape[1]
        out = torch.empty((nbags, T), dtype=torch.float32, device=w.device)
        _rel_fwd[(triton.cdiv(nbags, BB), triton.cdiv(T, BT))](table, cols, off, cnt, out, nbags, T=T, BB=BB, BT=BT)
        ctx.save_for_backward(cols, off, cnt)
        ctx.nbags, ctx.wshape = nbags, tuple(w.shape)
        return out

    @staticmethod
    def backward(ctx, g):
        cols, off, cnt = ctx.saved_tensors
        T = ctx.wshape[1]
        dw = torch.zeros(ctx.wshape, dtype=torch.float32, device=g.device)
        _rel_bwd[(triton.cdiv(ctx.nbags, BB), triton.cdiv(T, BT))](g.float().contiguous(), cols, off, cnt, dw,
                                                                   ctx.nbags, T=T, BB=BB, BT=BT)
        return dw, None, None, None, None, None


class RelSumEB(torch.autograd.Function):
    """RelSum's forward with embedding_bag's own backward: the table's gradient from
    aten._embedding_bag_dense_backward (sort by row, then segment sums in a fixed order), on the int64 indices and the
    bag of every index, which is what F.embedding_bag's forward hands its backward (it promotes int32 indices to its
    int64 offsets' type). Bit for bit F.embedding_bag's gradient, and deterministic."""

    @staticmethod
    def forward(ctx, w, table, cols, off, cnt, nbags):
        T = w.shape[1]
        out = torch.empty((nbags, T), dtype=torch.float32, device=w.device)
        _rel_fwd[(triton.cdiv(nbags, BB), triton.cdiv(T, BT))](table, cols, off, cnt, out, nbags, T=T, BB=BB, BT=BT)
        ctx.save_for_backward(cols, cnt)
        ctx.nbags, ctx.wshape = nbags, tuple(w.shape)
        return out

    @staticmethod
    def backward(ctx, g):
        cols, cnt = ctx.saved_tensors
        idx = cols.long()
        bag = torch.repeat_interleave(torch.arange(ctx.nbags, device=g.device), cnt, output_size=idx.shape[0])
        dw = torch.ops.aten._embedding_bag_dense_backward(g.float().contiguous(), idx, bag, cnt, idx.new_empty(0),
                                                          ctx.wshape[0], False, 0, None, -1)
        return dw, None, None, None, None, None


@triton.jit
def _seg_piece(g, sbag, rstart, rcnt, npiece, poff, prow, ptot, dw, part,
               T: tl.constexpr, CS: tl.constexpr, BT: tl.constexpr, EVR: tl.constexpr = "",
               EVS: tl.constexpr = "", inv=None, GSTRIDE: tl.constexpr = 0):
    # program (p, pt): piece p's at most CS sorted entries, lanes pt*BT .. pt*BT + BT - 1
    p = tl.program_id(0)
    pt = tl.program_id(1)
    ok = p < tl.load(ptot)                       # the grid is a bound on the pieces; programs past them do nothing
    row = tl.load(prow + p).to(tl.int64)
    k = (p - tl.load(poff + row)).to(tl.int64)   # this piece's place among its row's pieces
    s = tl.load(rstart + row) + k * CS
    n = tl.load(rcnt + row) - k * CS
    i = tl.arange(0, CS)
    im = (i < n) & ok
    b = tl.load(sbag + s + i, mask=im, other=0).to(tl.int64)
    lanes = pt * BT + tl.arange(0, BT)
    lm = lanes < T
    if GSTRIDE > 0:   # seg_grad16: bag b's gradient is row inv[b] of gs (bf16, GSTRIDE lanes a row), lanes 0..T-1
        r = tl.load(inv + b, mask=im, other=0).to(tl.int64)
        v = tl.load(g + r[:, None] * GSTRIDE + lanes[None, :], mask=im[:, None] & lm[None, :], other=0.0,
                    eviction_policy=EVR)
    else:
        v = tl.load(g + b[:, None] * T + lanes[None, :], mask=im[:, None] & lm[None, :], other=0.0,
                    eviction_policy=EVR)
    acc = tl.sum(v.to(tl.float32), axis=0)
    one = tl.load(npiece + row) == 1
    tl.store(dw + row * T + lanes, acc, mask=lm & ok & one, eviction_policy=EVS)
    tl.store(part + p.to(tl.int64) * T + lanes, acc, mask=lm & ok & (one == 0), eviction_policy=EVS)


@triton.jit
def _seg_join(part, npiece, poff, dw, T: tl.constexpr, BT: tl.constexpr):
    # program (row, pt): a row of two or more pieces adds its partials up in piece order
    row = tl.program_id(0).to(tl.int64)
    pt = tl.program_id(1)
    m = tl.load(npiece + row)
    lanes = pt * BT + tl.arange(0, BT)
    lm = lanes < T
    if m > 1:
        p0 = tl.load(poff + row).to(tl.int64)
        acc = tl.zeros((BT,), dtype=tl.float32)
        for j in range(0, m):
            acc += tl.load(part + (p0 + j) * T + lanes, mask=lm, other=0.0)
        tl.store(dw + row * T + lanes, acc, mask=lm)


@triton.jit
def _seg_join_list(part, npiece, poff, dw, mrows, nm, T: tl.constexpr, BT: tl.constexpr, NP: tl.constexpr):
    # program (q, pt): the rows mrows[q], mrows[q + NP], ... below nm[0] (the rows of two or more pieces, in row
    # order), each adding its partials up in piece order from 0.0, as _seg_join does
    q = tl.program_id(0)
    pt = tl.program_id(1)
    lanes = pt * BT + tl.arange(0, BT)
    lm = lanes < T
    for i in range(q, tl.load(nm), NP):
        row = tl.load(mrows + i).to(tl.int64)
        m = tl.load(npiece + row)
        p0 = tl.load(poff + row).to(tl.int64)
        acc = tl.zeros((BT,), dtype=tl.float32)
        for j in range(0, m):
            acc += tl.load(part + (p0 + j) * T + lanes, mask=lm, other=0.0)
        tl.store(dw + row * T + lanes, acc, mask=lm)


def seg_grad(g, cols, cnt, nbags, nrows, inv=None, T=None):
    """The table's gradient [nrows, T] fp32 from the sums' gradient g [nbags, T]: what embedding_bag's backward
    computes, in a fixed order of its own (see the module's text). No host synchronisation."""
    T = g.shape[1] if T is None else T   # inv given (seg_grad16): g is gs [nbags, L1] bf16 in sorted order
    nnz = cols.shape[0]
    dev = g.device
    if nrows <= 65536:   # rows - 32768 fit int16 in the same order: a stable sort gives the same permutation
        s16, perm = torch.sort((cols - 32768).to(torch.int16), stable=True)   # in 2 radix passes instead of 4
        srow = s16.to(cols.dtype) + 32768
    else:
        srow, perm = torch.sort(cols, stable=True)
    bag = torch.repeat_interleave(torch.arange(nbags, device=dev, dtype=torch.int32), cnt, output_size=nnz)
    sbag = bag[perm].contiguous()
    r = torch.arange(nrows, device=dev, dtype=srow.dtype)
    rstart = torch.searchsorted(srow, r)
    rcnt = torch.searchsorted(srow, r, right=True) - rstart
    npiece = (rcnt + CS - 1) // CS
    pend = torch.cumsum(npiece, 0)
    poff = (pend - npiece).contiguous()
    ptot = pend[-1:].contiguous()
    pmax = (nnz + CS - 1) // CS + nrows          # at least the number of pieces: no host read of ptot
    prow = torch.searchsorted(pend, torch.arange(pmax, device=dev, dtype=pend.dtype), right=True).clamp_(max=nrows - 1)
    dw = torch.zeros((nrows, T), dtype=torch.float32, device=dev)
    part = torch.empty((pmax, T), dtype=torch.float32, device=dev)
    gg = g.contiguous() if g.dtype == torch.bfloat16 else g.float().contiguous()   # bf16: rel_sum_rows(g16=True)
    _seg_piece[(pmax, triton.cdiv(T, BTB))](gg, sbag, rstart, rcnt, npiece, poff, prow.contiguous(), ptot, dw, part,
                                            T=T, CS=CS, BT=BTB, EVR="evict_last" if EVICT else "",
                                            EVS="evict_first" if EVICT else "",
                                            inv=inv if inv is not None else sbag,
                                            GSTRIDE=gg.shape[1] if inv is not None else 0)
    if SEG_LIST:   # the same sums over the rows that have partials only, found without a host sync
        multi = npiece > 1
        mrows = torch.argsort((~multi).to(torch.uint8), stable=True).to(torch.int32)   # those rows first, in order
        nm = multi.sum(dtype=torch.int32).view(1)
        _seg_join_list[(JNP, triton.cdiv(T, BTB))](part, npiece, poff, dw, mrows, nm, T=T, BT=BTB, NP=JNP)
    else:
        _seg_join[(nrows, triton.cdiv(T, BTB))](part, npiece, poff, dw, T=T, BT=BTB)
    return dw


class RelSumSeg(torch.autograd.Function):
    """RelSum's forward with seg_grad's backward: deterministic, no atomics."""

    @staticmethod
    def forward(ctx, w, table, cols, off, cnt, nbags):
        T = w.shape[1]
        out = torch.empty((nbags, T), dtype=torch.float32, device=w.device)
        _rel_fwd[(triton.cdiv(nbags, BB), triton.cdiv(T, BT))](table, cols, off, cnt, out, nbags, T=T, BB=BB, BT=BT)
        ctx.save_for_backward(cols, cnt)
        ctx.nbags, ctx.wshape = nbags, tuple(w.shape)
        return out

    @staticmethod
    def backward(ctx, g):
        cols, cnt = ctx.saved_tensors
        return seg_grad(g, cols, cnt, ctx.nbags, ctx.wshape[0]), None, None, None, None, None


def rel_sum(w, cols, cnt, nbags, bf16=False, eb_backward=False, seg_backward=False):
    """The relation rows' sums, as embedding_bag(cols, w, offsets=cumsum(cnt) - cnt, mode="sum") gives them: bags
    sorted, each bag's cols together (indices()'s order); cnt [nbags] rows a bag. bf16: the forward reads w in bf16.
    eb_backward: embedding_bag's own backward instead of the atomic one (RelSumEB)."""
    if w.dtype != torch.float32 or w.dim() != 2:
        raise ValueError(f"rel_sum: the table must be a 2-D fp32 tensor, not {w.dtype} {tuple(w.shape)}")
    if w.shape[0] >= 2 ** 31:
        raise ValueError("rel_sum: the table's rows must fit int32")
    cols = cols.to(torch.int32).contiguous()
    cnt = cnt.to(torch.int64).contiguous()
    if cnt.shape[0] != nbags:
        raise ValueError(f"rel_sum: {cnt.shape[0]} counts for {nbags} bags")
    off = (torch.cumsum(cnt, 0) - cnt).contiguous()
    table = w.detach().to(torch.bfloat16).contiguous() if bf16 else w.detach().contiguous()
    return (RelSumSeg if seg_backward else RelSumEB if eb_backward else RelSum).apply(w, table, cols, off, cnt, nbags)


@triton.jit
def _compact(rows, cnt, off, cols, nbags, S: tl.constexpr, BB: tl.constexpr):
    # program: BB bags' rows[b, :cnt[b]] to cols[off[b]:], in order
    b = (tl.program_id(0) * BB + tl.arange(0, BB)).to(tl.int64)
    bm = b < nbags
    c = tl.load(cnt + b, mask=bm, other=0)
    o = tl.load(off + b, mask=bm, other=0)
    k = tl.arange(0, S)
    m = bm[:, None] & (k[None, :] < c[:, None])
    tl.store(cols + o[:, None] + k[None, :], tl.load(rows + b[:, None] * S + k[None, :], mask=m, other=0), mask=m)


def rows_grad16(gs, inv, rows, cnt, nnz, ev, nrows, T):
    """NNUE_REL_GS16: the table's gradient from gs [nbags, L1] bf16 in sorted order (act_bwd's) through inv (side-row
    -> sorted slot), for the rows rel_rows() gave the forward: RelSumRows.backward's compact and seg_grad on bf16 rows."""
    ev.synchronize()
    n = int(nnz[0])
    nbags = rows.shape[0]
    c64 = cnt.long()
    off = (torch.cumsum(c64, 0) - c64).contiguous()
    flat = torch.empty(n, dtype=torch.int32, device=gs.device)
    _compact[(triton.cdiv(nbags, 16),)](rows, cnt, off, flat, nbags, S=rows.shape[1], BB=16)
    return seg_grad(gs, flat, c64, nbags, nrows, inv=inv, T=T)


def rel_sum_rows_nograd(w, rows, cnt, nbags, bf16=False):
    """RelSumRows's forward without autograd (NNUE_REL_GS16): the sums, and (rows, cnt, nnz, ev) for rows_grad16."""
    T, S = w.shape[1], rows.shape[1]
    table = bf16_copy(w) if bf16 else w.detach().contiguous()
    off = torch.arange(nbags, device=w.device, dtype=torch.int64) * S
    # NNUE_REL_LANE16: the sums rounded to bf16 as they are stored (the fp32 sum of each bag, then one rounding)
    out = torch.empty((nbags, T), dtype=torch.bfloat16 if LANE16 else torch.float32, device=w.device)
    _rel_fwd[(triton.cdiv(nbags, BB), triton.cdiv(T, BT))](table, rows, off, cnt, out, nbags, T=T, BB=BB, BT=BT,
                                                           EVR="evict_last" if EVICT else "",
                                                           EVS="evict_first" if EVICT else "")
    nnz = torch.empty(1, dtype=torch.int64, pin_memory=True)
    nnz.copy_(cnt.sum(dtype=torch.int64).view(1), non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    return out, (rows, cnt, nnz, ev)


class RelSumRows(torch.autograd.Function):
    """RelSumSeg on rel_rows.rel_rows()'s rows: bag b's rows in rows[b, :cnt[b]] (stride S = rows.shape[1]). The
    forward gathers them where they are (off = b * S): nothing is compacted and nothing waits for the host. It also
    copies the number of rows to the host without waiting; the backward, once that copy has landed (the GPU passed
    it long before: the rest of the forward and the backward up to here run in between), compacts the rows to the
    flat list RelSumSeg's backward sorts, the same rows in the same order. So both directions are RelSumSeg's bit
    for bit."""

    @staticmethod
    def forward(ctx, w, table, rows, cnt, nbags, g16=False):
        T, S = w.shape[1], rows.shape[1]
        off = torch.arange(nbags, device=w.device, dtype=torch.int64) * S
        out = torch.empty((nbags, T), dtype=torch.float32, device=w.device)
        _rel_fwd[(triton.cdiv(nbags, BB), triton.cdiv(T, BT))](table, rows, off, cnt, out, nbags, T=T, BB=BB, BT=BT,
                                                               EVR="evict_last" if EVICT else "",
                                                               EVS="evict_first" if EVICT else "")
        nnz = torch.empty(1, dtype=torch.int64, pin_memory=True)
        nnz.copy_(cnt.sum(dtype=torch.int64).view(1), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        ctx.save_for_backward(rows, cnt)
        ctx.nbags, ctx.wshape, ctx.nnz, ctx.ev, ctx.g16 = nbags, tuple(w.shape), nnz, ev, g16
        return out

    @staticmethod
    def backward(ctx, g):
        rows, cnt = ctx.saved_tensors
        ctx.ev.synchronize()
        nnz = int(ctx.nnz[0])
        c64 = cnt.long()
        off = (torch.cumsum(c64, 0) - c64).contiguous()
        cols = torch.empty(nnz, dtype=torch.int32, device=g.device)
        _compact[(triton.cdiv(ctx.nbags, 16),)](rows, cnt, off, cols, ctx.nbags, S=rows.shape[1], BB=16)
        if ctx.g16:   # the sums' gradient read in bf16 by the gather backward: half the bytes (not exact)
            g = g.to(torch.bfloat16)
        return seg_grad(g, cols, c64, ctx.nbags, ctx.wshape[0]), None, None, None, None, None


def rel_sum_rows(w, rows, cnt, nbags, bf16=False, g16=False):
    """rel_sum(seg_backward=True) on rel_rows.rel_rows()'s per-bag rows (see RelSumRows)."""
    if w.dtype != torch.float32 or w.dim() != 2:
        raise ValueError(f"rel_sum_rows: the table must be a 2-D fp32 tensor, not {w.dtype} {tuple(w.shape)}")
    if rows.dtype != torch.int32 or rows.shape[0] != nbags or cnt.shape[0] != nbags:
        raise ValueError(f"rel_sum_rows: rows {rows.dtype} {tuple(rows.shape)} and {cnt.shape[0]} counts for {nbags} bags")
    table = bf16_copy(w) if bf16 else w.detach().contiguous()   # bf16_copy: adamw16's copy, else the cast
    return RelSumRows.apply(w, table, rows, cnt, nbags, g16)


@triton.jit
def _rel_adamw(p_ptr, g_ptr, m_ptr, v_ptr, vm_ptr, coef_ptr, bc_ptr, n, lr, beta1, beta2, eps, lo, hi,
               BLOCK: tl.constexpr):
    # torch 2.11's fused AdamW (aten/src/ATen/native/cuda/fused_adam_utils.cuh, adam_math with amsgrad and no
    # weight decay), on the gradient times clip_grad_norm_'s coefficient (its _foreach_mul_), then clamp_(lo, hi):
    # the same fp32 operations in the same order, with IEEE division and square root as nvcc compiles them
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    p = tl.load(p_ptr + off, mask=mask, other=0.0)
    g = tl.load(g_ptr + off, mask=mask, other=0.0) * tl.load(coef_ptr)
    m = tl.load(m_ptr + off, mask=mask, other=0.0)
    v = tl.load(v_ptr + off, mask=mask, other=0.0)
    vm = tl.load(vm_ptr + off, mask=mask, other=0.0)
    bc1 = tl.load(bc_ptr)
    bc2s = tl.load(bc_ptr + 1)
    m = tl.fma(beta1, m, tl.fma(-beta1, g, g))
    gg = g * g
    v = tl.fma(beta2, v, tl.fma(-beta2, gg, gg))
    step_size = tl.div_rn(lr, bc1)
    vm = tl.maximum(vm, v)
    denom = tl.div_rn(tl.sqrt_rn(vm), bc2s) + eps
    p = p - tl.div_rn(step_size * m, denom)
    p = tl.minimum(tl.maximum(p, lo), hi)
    tl.store(p_ptr + off, p, mask=mask)
    tl.store(m_ptr + off, m, mask=mask)
    tl.store(v_ptr + off, v, mask=mask)
    tl.store(vm_ptr + off, vm, mask=mask)


def rel_adamw(w, g, state, coef, lr, betas, eps, lo, hi, BLOCK=2048):
    """One AdamW step (amsgrad, no weight decay) for the relation table in one pass, as torch's fused AdamW and
    clip_grad_norm_ compute it: g * coef, the update, then clamp_(lo, hi). state is the optimizer's state for w
    (torch's layout: step, exp_avg, exp_avg_sq, max_exp_avg_sq); step advances here, as torch advances it. The
    bias corrections come from torch's own pow on the device, as the fused kernel's pow_ (::pow) computes them."""
    if not state:
        state["step"] = torch.zeros((), dtype=torch.float32, device=w.device)
        state["exp_avg"] = torch.zeros_like(w, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(w, memory_format=torch.preserve_format)
        state["max_exp_avg_sq"] = torch.zeros_like(w, memory_format=torch.preserve_format)
    step = state["step"]
    step.add_(1)
    b1 = torch.tensor(float(betas[0]), dtype=torch.float32, device=w.device)
    b2 = torch.tensor(float(betas[1]), dtype=torch.float32, device=w.device)
    bc = torch.stack((1 - torch.pow(b1, step), torch.sqrt(1 - torch.pow(b2, step)))).contiguous()
    n = w.numel()
    _rel_adamw[(triton.cdiv(n, BLOCK),)](w, g, state["exp_avg"], state["exp_avg_sq"], state["max_exp_avg_sq"],
                                         coef, bc, n, float(lr), float(betas[0]), float(betas[1]), float(eps),
                                         float(lo), float(hi), BLOCK=BLOCK)


# ---- NNUE_REL_OPT=fused16: the two tables' AdamW step in one pass each, with their bf16 forward copies ------------
# rel_adamw's step for any table (the feature layer's or the relation rows'), plus, in the same pass, the bf16 copy
# of its first n16 values that the next forward reads in place of a cast (bf16_copy). The cast it replaces,
# w.to(torch.bfloat16), rounds to nearest even as this kernel's .to(tl.bfloat16) does (cvt.rn.bf16.f32), so the
# copy is the cast's bit for bit; the kernel writes it from the value it stores, after the clamp.
@triton.jit
def _adamw16(p_ptr, g_ptr, m_ptr, v_ptr, vm_ptr, o_ptr, coef_ptr, bc_ptr, n, n16, lr, beta1, beta2, eps, lo, hi,
             CLAMP: tl.constexpr, OUT16: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)   # n < 2^31
    mask = off < n
    p = tl.load(p_ptr + off, mask=mask, other=0.0)
    g = tl.load(g_ptr + off, mask=mask, other=0.0) * tl.load(coef_ptr)
    m = tl.load(m_ptr + off, mask=mask, other=0.0)
    v = tl.load(v_ptr + off, mask=mask, other=0.0)
    vm = tl.load(vm_ptr + off, mask=mask, other=0.0)
    bc1 = tl.load(bc_ptr)
    bc2s = tl.load(bc_ptr + 1)
    m = tl.fma(beta1, m, tl.fma(-beta1, g, g))
    gg = g * g
    v = tl.fma(beta2, v, tl.fma(-beta2, gg, gg))
    step_size = tl.div_rn(lr, bc1)
    vm = tl.maximum(vm, v)
    denom = tl.div_rn(tl.sqrt_rn(vm), bc2s) + eps
    p = p - tl.div_rn(step_size * m, denom)
    if CLAMP:
        p = tl.minimum(tl.maximum(p, lo), hi)
    tl.store(p_ptr + off, p, mask=mask)
    tl.store(m_ptr + off, m, mask=mask)
    tl.store(v_ptr + off, v, mask=mask)
    tl.store(vm_ptr + off, vm, mask=mask)
    if OUT16:
        tl.store(o_ptr + off, p.to(tl.bfloat16), mask=off < n16)


BF16 = {}   # id(table) -> (its bf16 copy of the first rows, the table's _version when the copy was written)
_BETAS = {}  # (device, beta1, beta2) -> both as fp32 tensors on the device, made once: a tensor made from a Python
             # float on a CUDA device is copied from pageable host memory, which waits for the GPU (a host sync)


def bf16_copy(w, rows=None):
    """w[:rows] (all of w without rows) in bf16 for a forward: the copy adamw16 wrote with the table's last step, if
    nothing has written the table since (a torch in-place op bumps its _version; adamw16's kernel does not), else
    the cast w[:rows].to(torch.bfloat16) of before. Both are the same bits."""
    e = BF16.get(id(w))
    nr = w.shape[0] if rows is None else rows
    if e is not None and e[1] == w._version and e[0].shape[0] == nr:
        return e[0]
    return w.detach()[:nr].to(torch.bfloat16).contiguous()


def adamw16(w, g, state, coef, lr, betas, eps, clamp=None, rows16=None, BLOCK=512, num_warps=4):
    """rel_adamw for any fp32 table w (amsgrad, no weight decay): g * coef, the update, clamp_(*clamp) when given,
    and with rows16 the bf16 copy of w[:rows16] for the next forward (bf16_copy finds it). state is the optimizer's
    state for w, created here as torch's AdamW creates it; its step advances here, as torch advances it."""
    if not state:
        state["step"] = torch.zeros((), dtype=torch.float32, device=w.device)
        state["exp_avg"] = torch.zeros_like(w, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(w, memory_format=torch.preserve_format)
        state["max_exp_avg_sq"] = torch.zeros_like(w, memory_format=torch.preserve_format)
    if not (w.is_contiguous() and g.is_contiguous() and g.shape == w.shape and w.dtype == g.dtype == torch.float32
            and all(state[k].is_contiguous() for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"))):
        raise ValueError("adamw16: contiguous fp32 table, gradient and state")
    step = state["step"]
    step.add_(1)
    key = (str(w.device), float(betas[0]), float(betas[1]))
    if key not in _BETAS:
        _BETAS[key] = tuple(torch.tensor(b, dtype=torch.float32, device=w.device) for b in key[1:])
    b1, b2 = _BETAS[key]
    bc = torch.stack((1 - torch.pow(b1, step), torch.sqrt(1 - torch.pow(b2, step)))).contiguous()
    n = w.numel()
    out = None
    if rows16 is not None:
        e = BF16.get(id(w))
        out = e[0] if e is not None and e[0].shape == (rows16,) + tuple(w.shape[1:]) else \
            torch.empty((rows16,) + tuple(w.shape[1:]), dtype=torch.bfloat16, device=w.device)
    lo, hi = clamp if clamp is not None else (0.0, 0.0)
    _adamw16[(triton.cdiv(n, BLOCK),)](w, g, state["exp_avg"], state["exp_avg_sq"], state["max_exp_avg_sq"],
                                       out if out is not None else w, coef, bc, n,
                                       out.numel() if out is not None else 0, float(lr), float(betas[0]),
                                       float(betas[1]), float(eps), float(lo), float(hi), CLAMP=clamp is not None,
                                       OUT16=out is not None, BLOCK=BLOCK, num_warps=num_warps)
    if out is not None:
        BF16[id(w)] = (out, w._version)
