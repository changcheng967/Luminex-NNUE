"""Own game data -> LCR1 records (the V5 session, 2026-11-27; the engine repo's nnue V6 DESIGN-NOTES 6.2).

Input: OpenBench datagen PGNs. These are fastchess self-play games starting from a FEN tag, every move commented by
the engine's DatagenInfo:
    {+1.04/13 0.119s, n=90538, sd=0, line=se=324 it=441,...,104 rn=d6d5:702,... b=e}

Output: each move with a usable comment becomes one LCR1 record of its position before the move. LCR1 is the 32-byte
record resident_train.py keeps in VRAM, laid out as binpack2lcr.cpp writes it:
  - occupancy u64;
  - the occupied squares' pieces in ascending square order, 4-bit codes (WHITE_PAWN 0 .. BLACK_KING 11), low nibble
    first;
  - i16 score for the side to move;
  - u8 flags: bit 0 white to move; bits 1-2: 0 quiet, 1 the move played captures, 2 in check; bit 3 the game result
    is known; bits 4-5 the side to move's result + 1 (0 loss, 1 draw, 2 win);
  - u8 piece count;
  - u32 position hash (binpack2lcr's mix64 hash, which the trainer's validation split reads).

The score is the engine's own search score for the side to move. The engine's UCI cp is its Value unchanged
(PAWN_VALUE 100), and Value is the net's output x 300, which is the trainer's target unit. So these labels sit on the
binpacks' scale. Mate in m is written as +-(32000 - 2m).

Splits and side files:
  - Each game goes whole to the training records or whole to the held-out ones. It is held out when the crc32 of its
    start FEN and movetext is 0 modulo HOLD, so equal games land together.
  - An .aux.tsv.gz beside each .lcr keeps the fields LCR1 has no room for, one line a record in the same order:
    se, it, rn, bound, ply, game.

Files per input member: MEMBER.lcr, MEMBER.val.lcr and their .aux.tsv.gz. manifest.tsv gets one line a member, and
--skip-done skips members it already lists, so a growing archive converts in passes.

usage: pgn2lcr.py OUT_DIR INPUT [INPUT ...] [--hold N] [--skip-done] [--part K/N]
  INPUT: an OpenBench archive (.tar of .pgn.bz2 members), a .pgn.bz2 or a .pgn
  --part K/N: this process takes every N-th member from the K-th (0-based), so N processes share an archive; they
              append to one manifest, one line a member.
"""
import bz2, gzip, hashlib, os, re, struct, sys, tarfile, time, zlib

import chess

M64 = (1 << 64) - 1
REC = struct.Struct("<Q16shBBI")
assert REC.size == 32
COMMENT = re.compile(r"^\s*([+-]?)(M?)(\d+(?:\.\d+)?)/(\d+) [\d.]+s, n=(\d+), sd=\d+, line=(.*?)\s*$", re.S)
TOKEN = re.compile(r"\{([^}]*)\}|([^\s{}]+)")
SKIP = re.compile(r"^(\d+\.+|1-0|0-1|1/2-1/2|\*)$")
MOVENO = re.compile(r"^\d+\.+")
WHITE_RESULT = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def mix64(x):
    x ^= x >> 33; x = (x * 0xff51afd7ed558ccd) & M64
    x ^= x >> 33; x = (x * 0xc4ceb9fe1a85ec53) & M64
    x ^= x >> 33
    return x


def record(board, move, score, res_white):
    occ = board.occupied
    white_occ = board.occupied_co[chess.WHITE]
    nib, k = 0, 0
    for sq in chess.scan_forward(occ):
        code = board.piece_type_at(sq) - 1 + (0 if (white_occ >> sq) & 1 else 6)
        nib |= code << (4 * k)
        k += 1
    pcs = nib.to_bytes(16, "little")
    white = board.turn == chess.WHITE
    q = 2 if board.is_check() else (1 if board.is_capture(move) else 0)
    res = (res_white if white else -res_white) + 1
    flags = (1 if white else 0) | (q << 1) | (1 << 3) | (res << 4)
    lo, hi = int.from_bytes(pcs[:8], "little"), int.from_bytes(pcs[8:], "little")
    h = mix64(occ ^ mix64(lo ^ mix64(hi ^ (1 if white else 0)))) >> 32
    return REC.pack(occ, pcs, max(-31999, min(31999, score)), flags, k, h), q


def parse_comment(c):
    m = COMMENT.match(c)
    if not m:
        return None
    sign, mate, val, depth, nodes, line = m.groups()
    kv = dict(x.split("=", 1) for x in line.split() if "=" in x)
    if mate:
        mv = int(val)
        score = 32000 - 2 * mv
    else:
        score = round(float(val) * 100)
    return (-score if sign == "-" else score), kv


def games(text):
    """(start fen, result, [(san, comment or None)]) for each game of a PGN text."""
    for block in re.split(r"\n(?=\[Event )", text):
        if not block.strip():
            continue
        head, _, body = block.partition("\n\n")
        fen = re.search(r'^\[FEN "([^"]+)"\]', head, re.M)
        result = re.search(r'^\[Result "([^"]+)"\]', head, re.M)
        moves = []
        for com, tok in TOKEN.findall(body):
            if tok:
                if SKIP.match(tok):
                    continue
                tok = MOVENO.sub("", tok)   # "12.e4" or "12...e5" written as one token
                if tok:
                    moves.append([tok, None])
            elif moves and moves[-1][1] is None:
                moves[-1][1] = com
        yield (fen.group(1) if fen else chess.STARTING_FEN), (result.group(1) if result else "*"), moves


def convert(name, text, out, hold, stats):
    paths = {s: f"{out}/{name}{s}" for s in (".lcr", ".val.lcr", ".aux.tsv.gz", ".val.aux.tsv.gz")}
    tmp = {s: p + ".tmp" for s, p in paths.items()}
    fl, fv = open(tmp[".lcr"], "wb"), open(tmp[".val.lcr"], "wb")
    al, av = gzip.open(tmp[".aux.tsv.gz"], "wt", compresslevel=6), gzip.open(tmp[".val.aux.tsv.gz"], "wt", compresslevel=6)
    n = dict(games=0, val_games=0, records=0, val_records=0, no_comment=0, no_result=0, bad=0, quiet=0, capture=0, check=0)
    gi = 0
    for fen, result, moves in games(text):
        gi += 1
        if result not in WHITE_RESULT:
            n["no_result"] += 1
            continue
        gid = f"{name}:{gi}"
        val = zlib.crc32((fen + " " + " ".join(m for m, _ in moves)).encode()) % hold == 0
        board = chess.Board(fen)
        recs, aux, qs, nc = [], [], [0, 0, 0], 0
        try:
            for ply, (san, com) in enumerate(moves):
                move = board.parse_san(san)
                parsed = parse_comment(com) if com is not None else None
                if parsed is None:
                    nc += 1
                else:
                    score, kv = parsed
                    rec, q = record(board, move, score, WHITE_RESULT[result])
                    recs.append(rec)
                    aux.append(f"{kv.get('se', '')}\t{kv.get('it', '')}\t{kv.get('rn', '')}\t{kv.get('b', '')}\t{ply}\t{gid}\n")
                    qs[q] += 1
                board.push(move)
        except ValueError:   # a move that does not parse: the whole game is left out
            n["bad"] += 1
            continue
        f, a = (fv, av) if val else (fl, al)
        f.write(b"".join(recs))
        a.write("".join(aux))
        n["val_games" if val else "games"] += 1
        n["val_records" if val else "records"] += len(recs)
        n["no_comment"] += nc
        n["quiet"] += qs[0]; n["capture"] += qs[1]; n["check"] += qs[2]
    for h in (fl, fv, al, av):
        h.close()
    for s in paths:
        os.replace(tmp[s], paths[s])
    sha = {s: hashlib.sha256(open(paths[s], "rb").read()).hexdigest()[:16] for s in (".lcr", ".val.lcr")}
    for k, v in n.items():
        stats[k] = stats.get(k, 0) + v
    return n, sha


def main():
    args = sys.argv[1:]
    hold, skip_done, part, parts = 20, False, 0, 1
    if "--hold" in args:
        i = args.index("--hold"); hold = int(args[i + 1]); del args[i:i + 2]
    if "--skip-done" in args:
        args.remove("--skip-done"); skip_done = True
    if "--part" in args:   # K/N: this process takes the members whose index is K modulo N
        i = args.index("--part"); part, parts = map(int, args[i + 1].split("/")); del args[i:i + 2]
    if len(args) < 2:
        sys.exit(__doc__)
    out, inputs = args[0], args[1:]
    os.makedirs(out, exist_ok=True)
    man = f"{out}/manifest.tsv"
    done = {l.split("\t")[0] for l in open(man)} if skip_done and os.path.exists(man) else set()
    stats, t0, seen = {}, time.time(), [0]

    def one(name, raw):
        seen[0] += 1
        if (seen[0] - 1) % parts != part or name in done:
            return
        text = (bz2.decompress(raw) if raw[:3] == b"BZh" else raw).decode("utf-8", "replace")
        n, sha = convert(name, text, out, hold, stats)
        with open(man, "a") as m:
            m.write(f"{name}\t{n['games']}\t{n['records']}\t{n['val_games']}\t{n['val_records']}\t{n['no_comment']}\t"
                    f"{n['no_result']}\t{n['bad']}\t{sha['.lcr']}\t{sha['.val.lcr']}\n")
        print(f"  {name}: {n['games'] + n['val_games']} games ({n['val_games']} held out), {n['records'] + n['val_records']:,} "
              f"records, {n['no_comment']} moves without a comment, {n['bad']} bad games, {time.time() - t0:.0f} s", flush=True)

    for path in inputs:
        if path.endswith(".tar"):
            with tarfile.open(path) as tf:
                for mem in tf:
                    if mem.isfile() and mem.name.endswith((".pgn.bz2", ".pgn")):
                        base = os.path.basename(mem.name)
                        one(base[:-len(".pgn.bz2")] if base.endswith(".bz2") else base[:-len(".pgn")], tf.extractfile(mem).read())
        else:
            base = os.path.basename(path)
            one(base[:-len(".pgn.bz2")] if base.endswith(".bz2") else base[:-len(".pgn")], open(path, "rb").read())
    r = stats.get("records", 0) + stats.get("val_records", 0)
    print(f"done: {stats.get('games', 0) + stats.get('val_games', 0):,} games ({stats.get('val_games', 0):,} held out), "
          f"{r:,} records (quiet {stats.get('quiet', 0):,}, capture {stats.get('capture', 0):,}, check {stats.get('check', 0):,}), "
          f"{stats.get('no_comment', 0)} moves without a comment, {stats.get('no_result', 0)} games without a result, "
          f"{stats.get('bad', 0)} bad games, {time.time() - t0:.0f} s")
    print("PGN2LCR_DONE")


if __name__ == "__main__":
    main()
