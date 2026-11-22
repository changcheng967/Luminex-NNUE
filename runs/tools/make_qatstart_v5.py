"""The V5 arm's QAT start from its float phase's checkpoint, as strip_heads.py makes the control's: the draw head out
of the model and out of the optimizer, the rest of the optimizer state kept. The V5 trainer's groups are
[feature layer, tail weights, tail biases] + the draw head (patch_draw) + the relation table (patch_rel); the QAT
trainer has no draw head, so its groups are the first three and the relation table, in that order. The draw head's
group is found by its parameters' shapes, the relation group by the table's, and the relation group's learning rate
is set to QAT's (NNUE_LR x NNUE_REL_LR_MULT): the trainer resets only the first three groups' rates after it loads
the state. The relation table must be there and trained, because the trainer loads with strict=False and would
leave a missing table at zero without a word.
usage: make_qatstart_v5.py SRC.pt DST.pt QAT_LR REL_LR_MULT"""
import sys

import torch

src, dst, qat_lr, mult = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["model"]
rel = sd.get("rel")
if rel is None or rel.dim() != 2 or float(rel.abs().max()) == 0.0:
    sys.exit(f"FAIL: {src} holds no trained relation table")
heads = [k for k in sd if k.startswith("dhead")]
model = {k: v for k, v in sd.items() if not k.startswith("dhead")}
opt = ck.get("opt")
if opt is None:
    sys.exit(f"FAIL: {src} holds no optimizer state")
shape = {i: tuple(s["exp_avg"].shape) for i, s in opt["state"].items() if "exp_avg" in s}
head_shapes = sorted(tuple(sd[k].shape) for k in heads)
groups, dropped, rel_groups = [], 0, 0
for g in opt["param_groups"]:
    gs = sorted(shape.get(i, ()) for i in g["params"])
    if heads and gs == head_shapes:
        dropped += 1
        continue
    if gs == [tuple(rel.shape)]:
        g = dict(g, lr=qat_lr * mult, initial_lr=qat_lr * mult)
        rel_groups += 1
    groups.append(g)
if dropped != (1 if heads else 0) or rel_groups != 1 or len(groups) != 4 or [tuple(shape[i]) for i in groups[3]["params"]] != [tuple(rel.shape)]:
    sys.exit(f"FAIL: optimizer groups {[sorted(shape.get(i, ()) for i in g['params']) for g in opt['param_groups']]}: "
             f"not the float phase's [feature layer, tail weights, tail biases, draw head, relation table]")
kept = {i for g in groups for i in g["params"]}
opt = {"state": {i: s for i, s in opt["state"].items() if i in kept}, "param_groups": groups}
torch.save({"model": model, "gstep": ck.get("gstep", 0), "opt": opt, "swa": None}, dst)
k = torch.round(rel * 128.0)
print(f"  V5 QAT start {dst}: {len(heads)} draw-head tensors and their optimizer group left out; relation table "
      f"{tuple(rel.shape)} |w| max {float(rel.abs().max()):.4f}, {float((k.abs() > 127).float().mean()):.4%} of it "
      f"beyond the int8 range; its group's learning rate {qat_lr * mult:g}; step {ck.get('gstep', 0)}")
