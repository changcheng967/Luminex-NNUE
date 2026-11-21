"""V5 value gate: puts rel_gate.py's relation inputs into resident_train.py. Eleven small, counted edits; every
anchor must be found exactly once or the patch stops. This version is for the consolidated trainer (branch
trainer-stack, whose fused kernels take extra lane sums: branch fused-rel-lanes): with NNUE_FT_ACT=1 the relation
rows go into the fused activation kernels through the trainer's LANE_SUMS hook (rel_gate.lane_sums()); with
NNUE_FT_ACT=0 they are joined ahead of the activation as before. It applies before or after the cells' draw-head
patch (PATCH_DRAW): none of its anchors is one of theirs.

With NNUE_REL_T unset or 0 the patched trainer computes what the unpatched one does, bit for bit: every hook
returns at once. So the control arm and the relation arms run the same file.

usage: patch_rel.py SRC DST      (rel_gate.py must sit next to DST)"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
rt = open(src, encoding="utf-8").read()
edits = 0


def patch(old, new, what):
    global rt, edits
    if rt.count(old) != 1:
        sys.exit(f"rel patch: {what} appears {rt.count(old)} times")
    rt = rt.replace(old, new)
    edits += 1


patch("REC = 32   # LCR1 record bytes (src/featurize.cpp CompactRec)\n",
      "import rel_gate as REL  # noqa: E402   V5 value gate: relation inputs from the raw boards (NNUE_REL_T)\n"
      "\n"
      "REC = 32   # LCR1 record bytes (src/featurize.cpp CompactRec)\n", "the record size line")

patch('dev = env("NNUE_DEVICE", "cuda")   # cpu: a slow dry run of every code path on a machine without a GPU\n',
      'dev = env("NNUE_DEVICE", "cuda")   # cpu: a slow dry run of every code path on a machine without a GPU\n'
      'REL.check(QAT, globals().get("QAT_AFTER", 0), globals().get("FT_ACT", False), lanes=True)\n', "the device line")

patch("    return wf, bf, (flags & 1).float(), target, (flags >> 1) & 3, cnt\n",
      "    REL.note(rec)   # the next forward() adds these records' relation rows\n"
      "    return wf, bf, (flags & 1).float(), target, (flags >> 1) & 3, cnt\n", "expand()'s return")

patch("        acc_b = model.ft(bf) + model.ft_bias\n",
      "        acc_b = model.ft(bf) + model.ft_bias\n"
      "    if REL.T and hv is None:   # the relation rows' sums on lanes 0..T-1, ahead of the activation\n"
      "        acc_w, acc_b = REL.add(model, acc_w, acc_b)\n", "forward()'s feature-layer sums")

patch("LANE_SUMS = None\n",
      "LANE_SUMS = REL.lane_sums if REL.T and FT_ACT else None   # V5 gate: the relation rows inside the fused kernels\n",
      "the fused kernels' lane-sums hook")

patch("    model = LNNUE(L1=L1).to(dev)\n",
      "    model = LNNUE(L1=L1).to(dev)\n"
      "    REL.attach(model, dev)   # model.rel, zero: a no-op without NNUE_REL_T\n", "model construction")

patch("    opt = torch.optim.AdamW(groups, lr=LR, amsgrad=True, fused=True)\n",
      "    if REL.T:   # the relation table: its own group, no weight decay, NNUE_REL_LR_MULT times the rate\n"
      "        groups.append(REL.group(model, LR))\n"
      "    opt = torch.optim.AdamW(groups, lr=LR, amsgrad=True, fused=True)\n", "the optimizer line")

m = re.findall(r"(?m)^( +)opt\.step\(\)\n", rt)
if len(m) != 1:
    sys.exit(f"rel patch: opt.step() appears {len(m)} times")
rt = re.sub(r"(?m)^( +)opt\.step\(\)\n", lambda k: f"{k.group(1)}opt.step()\n{k.group(1)}REL.clip(model)   # rows inside the int8 range\n", rt)
edits += 1

patch('          f"tail dead/stuck L2 {d2}/{s2} of {n2}, L3 {d3}/{s3} of {n3} | material (cp) {material_probe(model)}", flush=True)\n',
      '          f"tail dead/stuck L2 {d2}/{s2} of {n2}, L3 {d3}/{s3} of {n3} | material (cp) {material_probe(model)}", flush=True)\n'
      '    REL.health(model, gstep, val[:k])\n', "the [HEALTH] line")

# Two more lines of output, relations or not: the held-out losses to nine places. The trainer prints six, which is
# 0.08% of a January-tail loss, and the gate compares arms that differ by a few tenths of a percent.
patch('    REL.health(model, gstep, val[:k])\n',
      '    REL.health(model, gstep, val[:k])\n'
      '    print(f"[GATE HEALTH step {gstep}] loss={loss:.9f}", flush=True)\n', "the [HEALTH] hook")
patch('          f"{len(dval):,} positions", flush=True)\n',
      '          f"{len(dval):,} positions", flush=True)\n'
      '    _gate_tag = locals().get("tag", "DISJOINT")\n'
      '    print(f"[GATE {_gate_tag} step {gstep}] loss={loss:.9f}", flush=True)\n', "the [DISJOINT] line")

open(dst, "w", encoding="utf-8").write(rt)
print(f"  relation inputs: patched trainer {dst} ({edits} edits)")
