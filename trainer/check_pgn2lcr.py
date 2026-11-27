"""Check pgn2lcr.py's records against python-chess, with a decoder of its own (the V5 session, 2026-11-27).

For N random records of a converted member (MEMBER.lcr with MEMBER.aux.tsv.gz, or the .val pair), the game is
replayed from the PGN to that record's ply, and the record must match the board there:
  - placement and side to move, decoded as the trainer reads LCR1 (occupancy bits; 4-bit codes of the occupied
    squares in ascending order, low nibble first; PIECES = "PNBRQKpnbrqk"; flags bit 0);
  - the piece count;
  - in check / capture class, against the board and the move played;
  - the result bits, against the game's result for the side to move;
  - the score, against the move's comment;
  - the position hash, recomputed from the record's own bytes.
Over every record it also counts the classes, the result bits and the share of scores beyond the trainer's 1500
clip. Pure Python (Oracle has no numpy).
usage: check_pgn2lcr.py PGN(.bz2) LCR_DIR MEMBER [N] [--val]
"""
import bz2, gzip, random, re, struct, sys

import chess

PIECES = "PNBRQKpnbrqk"
MASK = 0xFFFFFFFFFFFFFFFF
COMMENT = re.compile(r"^\s*([+-]?)(M?)(\d+(?:\.\d+)?)/")
args = sys.argv[1:]
val = "--val" in args
args = [a for a in args if a != "--val"]
pgn, d, member = args[:3]
N = int(args[3]) if len(args) > 3 else 2000
suf = ".val" if val else ""
data = open(f"{d}/{member}{suf}.lcr", "rb").read()
assert len(data) % 32 == 0
recs = [data[i:i + 32] for i in range(0, len(data), 32)]
aux = [l.rstrip("\n").split("\t") for l in gzip.open(f"{d}/{member}{suf}.aux.tsv.gz", "rt")]
assert len(aux) == len(recs), f"{len(aux)} aux lines for {len(recs)} records"
raw = open(pgn, "rb").read()
text = (bz2.decompress(raw) if raw[:3] == b"BZh" else raw).decode("utf-8", "replace")
blocks = [b for b in re.split(r"\n(?=\[Event )", text) if b.strip()]


def decode(r):
    occ = int.from_bytes(r[0:8], "little")
    sq = [s for s in range(64) if occ >> s & 1]
    board = [None] * 64
    for k, s in enumerate(sq):
        board[s] = PIECES[r[8 + (k >> 1)] >> (4 * (k & 1)) & 15]
    rows = []
    for rank in range(7, -1, -1):
        row, e = "", 0
        for f in range(8):
            p = board[rank * 8 + f]
            if p is None:
                e += 1
            else:
                row += (str(e) if e else "") + p
                e = 0
        rows.append(row + (str(e) if e else ""))
    return "/".join(rows), bool(r[26] & 1), len(sq)


def mix(x):
    for shift_mul in (0xff51afd7ed558ccd, 0xc4ceb9fe1a85ec53):
        x = ((x ^ (x >> 33)) * shift_mul) & MASK
    return x ^ (x >> 33)


def moves_of(block):
    body = block.partition("\n\n")[2]
    out = []
    for com, tok in re.findall(r"\{([^}]*)\}|([^\s{}]+)", body):
        if tok:
            if re.match(r"^(\d+\.+|1-0|0-1|1/2-1/2|\*)$", tok):
                continue
            tok = re.sub(r"^\d+\.+", "", tok)
            if tok:
                out.append([tok, None])
        elif out and out[-1][1] is None:
            out[-1][1] = com
    return out


fails = {}
def bad(kind, i):
    fails.setdefault(kind, []).append(i)


random.seed(20261127)
picks = random.sample(range(len(recs)), min(N, len(recs)))
for i in picks:
    r = recs[i]
    ply, gid = int(aux[i][4]), aux[i][5]
    block = blocks[int(gid.rsplit(":", 1)[1]) - 1]
    fen = re.search(r'^\[FEN "([^"]+)"\]', block, re.M)
    result = re.search(r'^\[Result "([^"]+)"\]', block, re.M).group(1)
    board = chess.Board(fen.group(1) if fen else chess.STARTING_FEN)
    mv = moves_of(block)
    for san, _ in mv[:ply]:
        board.push_san(san)
    move = board.parse_san(mv[ply][0])
    placement, white, k = decode(r)
    if placement != board.board_fen(): bad("placement", i)
    if white != (board.turn == chess.WHITE): bad("side to move", i)
    if k != chess.popcount(board.occupied) or r[27] != k: bad("piece count", i)
    q = (r[26] >> 1) & 3
    if q != (2 if board.is_check() else (1 if board.is_capture(move) else 0)): bad("class", i)
    wr = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[result]
    if (r[26] >> 3) & 1 != 1 or (r[26] >> 4) & 3 != (wr if white else -wr) + 1: bad("result bits", i)
    m = COMMENT.match(mv[ply][1])
    s = (32000 - 2 * int(m.group(3))) if m.group(2) else round(float(m.group(3)) * 100)
    s = -s if m.group(1) == "-" else s
    if struct.unpack("<h", r[24:26])[0] != max(-31999, min(31999, s)): bad("score", i)
    occ = int.from_bytes(r[0:8], "little")
    lo, hi = int.from_bytes(r[8:16], "little"), int.from_bytes(r[16:24], "little")
    h = mix(occ ^ mix(lo ^ mix(hi ^ (1 if white else 0)))) >> 32
    if h != int.from_bytes(r[28:32], "little"): bad("hash", i)
cls = [0, 0, 0, 0]
res = {0: 0, 1: 0, 2: 0, "unknown": 0}
big = 0
for r in recs:
    cls[(r[26] >> 1) & 3] += 1
    res[(r[26] >> 4) & 3 if (r[26] >> 3) & 1 else "unknown"] += 1
    big += abs(struct.unpack("<h", r[24:26])[0]) > 1500
n = max(len(recs), 1)
print(f"{member}{suf}: {len(recs):,} records; {len(picks):,} checked against python-chess")
for kind in ("placement", "side to move", "piece count", "class", "result bits", "score", "hash"):
    print(f"  {kind:<13} {'ok' if kind not in fails else str(len(fails[kind])) + ' WRONG, e.g. record ' + str(fails[kind][0])}")
print(f"  classes: quiet {cls[0] / n:.3f}, capture {cls[1] / n:.3f}, check {cls[2] / n:.3f}; results for the side to "
      f"move: loss {res[0] / n:.3f}, draw {res[1] / n:.3f}, win {res[2] / n:.3f}, unknown {res['unknown'] / n:.3f}; "
      f"|score| > 1500: {big / n:.4f}")
print("CHECK_OK" if not fails else "CHECK_FAILED")
