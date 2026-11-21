"""The grouped models must compute exactly what separate official MS-TCN models compute.

Three official models with different feature sizes are copied into one GroupedModel. Without
dropout (eval mode) the outputs, the losses, the gradients and the weights after one Adam step must
match the separate models to float precision. The batch order for a seed must match the official
batch generator's, and the inputs built on the card must equal the CPU loader's.

Usage: python -I tests/test_grouped.py
"""
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import optim  # noqa: E402

from mstcn.batch_gen import BatchGenerator  # noqa: E402
from mstcn.grouped import (FeatureSource, GroupedBatchGenerator, GroupedModel, GroupedTrainer,  # noqa: E402
                           group_losses)
from mstcn.model import MultiStageModel  # noqa: E402
from train_mstcn import make_loader  # noqa: E402

C, T = 48, 700
DIMS = [384, 2048, 768]


def official_loss(p_all, target, mask):
    ce, mse = torch.nn.CrossEntropyLoss(ignore_index=-100), torch.nn.MSELoss(reduction="none")
    loss = 0
    for p in p_all:
        loss += ce(p.transpose(2, 1).contiguous().view(-1, C), target.view(-1))
        loss += 0.15 * torch.mean(torch.clamp(mse(F.log_softmax(p[:, :, 1:], dim=1),
                                                  F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                              min=0, max=16) * mask[:, :, 1:])
    return loss


def close(a, b, what, atol=1e-5, rtol=1e-4):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{what}: max difference {(a - b).abs().max().item():.3g}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    # Full fp32 for the comparison. With the default TF32 convolutions on this card, the grouped and
    # separate kernels round differently (outputs differ by about 1e-3), as any two algorithms would.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    officials = [MultiStageModel(4, 10, 64, d, C).to(device).eval() for d in DIMS]
    grouped = GroupedModel(4, 10, 64, DIMS, C).to(device).eval()
    for g, m in enumerate(officials):
        grouped.load_official(g, m.state_dict())
        back = grouped.official_state(g)
        assert back.keys() == m.state_dict().keys()
        for k, v in m.state_dict().items():
            assert torch.equal(back[k], v), k

    xs = [torch.randn(1, d, T, device=device) for d in DIMS]
    target = torch.randint(0, C, (1, T), device=device)
    mask = torch.ones(1, C, T, device=device)
    out_g = grouped(xs, mask).view(4, 1, len(DIMS), C, T)
    losses = group_losses(grouped(xs, mask), target, mask, C)
    losses.sum().backward()
    opt_g = optim.Adam(grouped.parameters(), lr=0.0005)
    opt_g.step()
    for g, m in enumerate(officials):
        out = m(xs[g], mask)
        close(out_g[:, :, g], out, f"model {g} outputs")
        loss = official_loss(out, target, mask)
        close(losses[g], loss, f"model {g} loss")
        loss.backward()
        grads = {k: p.grad for k, p in m.named_parameters()}
        for p, name, rows in grouped._pairs(g):
            gp = p.grad if rows is None else p.grad[rows[0]:rows[1]]
            close(gp, grads[name], f"model {g} gradient {name}", atol=1e-6, rtol=1e-3)
        optim.Adam(m.parameters(), lr=0.0005).step()
        after = grouped.official_state(g)
        for k, v in m.state_dict().items():
            close(after[k], v, f"model {g} weight after Adam {k}", atol=1e-6, rtol=1e-4)
    print(f"outputs, losses, gradients and one Adam step match for {len(DIMS)} models on {device}")

    # initial weights: model g of a GroupedTrainer equals the official model drawn after manual_seed(seed)
    tr = GroupedTrainer(DIMS, C, seed=1538574472)
    for g, d in enumerate(DIMS):
        torch.manual_seed(1538574472)
        ref = MultiStageModel(4, 10, 64, d, C).state_dict()
        got = tr.model.official_state(g)
        assert all(torch.equal(got[k], ref[k]) for k in ref), f"initial weights of model {g}"
    print("initial weights match the official ones for the seed")

    # batch order: the same shuffles as the official batch generator for the same seed
    vids = [f"v{i}.txt" for i in range(50)]
    random.seed(7)
    a = BatchGenerator(C, {}, "", None, 1)
    a.read_data(vids)
    order_a = [list(a.list_of_examples)]
    a.reset()
    order_a.append(list(a.list_of_examples))
    random.seed(7)
    b = GroupedBatchGenerator({}, "", None)
    b.read_data(vids)
    order_b = [list(b.list_of_examples)]
    b.reset()
    order_b.append(list(b.list_of_examples))
    assert order_a == order_b
    print("batch order matches the official batch generator")

    # inputs built on the card equal train_mstcn.make_loader's, motion cue included
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        np.save(folder / "v.npy", (np.random.default_rng(0).standard_normal((384, 333)) * 3).astype(np.float16))
        np.save(folder / "w.npy", np.random.default_rng(1).standard_normal((2048, 333)).astype(np.float32))
        settings = [(folder, 0), (folder, 4), (folder, 8)]
        source = FeatureSource(settings)
        raw = source.load("v.txt")
        xs = source.inputs(raw, source.frames(raw), device)
        for (f, k), x in zip(settings, xs):
            ref = make_loader(f, k)("v.txt")
            assert x.dtype == torch.float32 and np.array_equal(x[0].cpu().numpy(), ref), f"inputs for k={k}"
        source = FeatureSource([(folder, 0)])
        raw = {folder: torch.from_numpy(np.load(folder / "w.npy")).pin_memory()}
        assert np.array_equal(source.inputs(raw, 333, device)[0][0].cpu().numpy(), make_loader(folder, 0)("w.txt"))
    print("inputs built on the card equal the CPU loader's, exactly")


if __name__ == "__main__":
    main()
