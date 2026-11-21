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
import torch
import triton
import triton.language as tl

BB = 32    # bags a program
BT = 128   # lanes a program
CS = 64    # sorted entries a piece (seg backward)
BTB = 64   # lanes a program (seg backward)


@triton.jit
def _rel_fwd(table, cols, off, cnt, out, nbags, T: tl.constexpr, BB: tl.constexpr, BT: tl.constexpr):
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
        v = tl.load(table + col[:, None] * T + lanes[None, :], mask=km[:, None] & lm[None, :], other=0.0)
        acc += v.to(tl.float32)
    tl.store(out + b[:, None] * T + lanes[None, :], acc, mask=bm[:, None] & lm[None, :])


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
               T: tl.constexpr, CS: tl.constexpr, BT: tl.constexpr):
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
    v = tl.load(g + b[:, None] * T + lanes[None, :], mask=im[:, None] & lm[None, :], other=0.0)
    acc = tl.sum(v.to(tl.float32), axis=0)
    one = tl.load(npiece + row) == 1
    tl.store(dw + row * T + lanes, acc, mask=lm & ok & one)
    tl.store(part + p.to(tl.int64) * T + lanes, acc, mask=lm & ok & (one == 0))


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


def seg_grad(g, cols, cnt, nbags, nrows):
    """The table's gradient [nrows, T] fp32 from the sums' gradient g [nbags, T]: what embedding_bag's backward
    computes, in a fixed order of its own (see the module's text). No host synchronisation."""
    T = g.shape[1]
    nnz = cols.shape[0]
    dev = g.device
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
    gg = g.float().contiguous()
    _seg_piece[(pmax, triton.cdiv(T, BTB))](gg, sbag, rstart, rcnt, npiece, poff, prow.contiguous(), ptot, dw, part,
                                            T=T, CS=CS, BT=BTB)
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
