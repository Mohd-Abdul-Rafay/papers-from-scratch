"""
LeNet-5 — faithful implementation of LeCun, Bottou, Bengio & Haffner (1998),
"Gradient-Based Learning Applied to Document Recognition."

Verified parameter counts (match the paper exactly):

    C1     156
    S2      12
    C3   1,516
    S4      32
    C5  48,120
    F6  10,164
    -----------
        60,000
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, for utils
from utils import match


# ============================================================
# 1. ACTIVATION
#    f(x) = A * tanh(S * x),  A = 1.7159, S = 2/3
#
#    Paper, Appendix A. Chosen so that f(1) = 1 and f(-1) = -1,
#    which places the RBF targets at the points of maximum
#    curvature rather than in the saturated tails.
# ============================================================

class LeNetActivation(nn.Module):
    A = 1.7159
    S = 2.0 / 3.0

    def forward(self, x):
        return self.A * torch.tanh(self.S * x)


# ============================================================
# 2. TRAINABLE SUBSAMPLING (S2, S4)
#
#    Paper: "The four inputs to a unit in S2 are added, then
#    multiplied by a trainable coefficient, and then added to a
#    trainable bias. The result is passed through a sigmoidal
#    function."
#
#    One coefficient + one bias per feature map  ->  2 * C params.
# ============================================================

class SubsamplingLayer(nn.Module):
    def __init__(self, channels, init_coeff=0.25):
        super().__init__()

        # init_coeff defaults to 0.25 (= 1/4) so that
        #     coeff * sum(4 values)  ~=  mean(4 values)
        # which keeps the pre-activation in the same range as the
        # input. Initialising to 1.0 sums four activations of up to
        # 1.7159 each, giving ~6.9 -> tanh(4.6) ~= 0.9998, i.e. the
        # layer starts fully saturated with almost no gradient.
        self.weight = nn.Parameter(torch.full((channels,), float(init_coeff)))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.activation = LeNetActivation()

    def forward(self, x):
        # avg_pool returns the MEAN of each 2x2 block; the paper sums,
        # so multiply by 4 to recover the sum.
        x = F.avg_pool2d(x, kernel_size=2, stride=2) * 4.0
        x = x * self.weight.view(1, -1, 1, 1)
        x = x + self.bias.view(1, -1, 1, 1)
        return self.activation(x)


# ============================================================
# 3. C3 — PARTIAL CONNECTION LAYER
#
#    Table I of the paper. Each C3 map sees only a subset of S2's
#    six maps. Two reasons, per the paper: it bounds the connection
#    count, and -- more importantly -- it "forces a break of
#    symmetry", since maps given different inputs cannot converge
#    on identical features.
#
#    6 maps x 3 inputs, 6 x 4 contiguous, 3 x 4 non-contiguous,
#    1 x all six. Each S2 map is used exactly 10 times.
# ============================================================

C3_CONNECTIONS = [
    # three contiguous
    [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 4, 5], [0, 1, 5],
    # four contiguous
    [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5],
    [0, 3, 4, 5], [0, 1, 4, 5], [0, 1, 2, 5],
    # four non-contiguous
    [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],
    # all six
    [0, 1, 2, 3, 4, 5],
]


class C3Layer(nn.Module):
    def __init__(self, connections=C3_CONNECTIONS):
        super().__init__()
        self.connections = connections
        self.convs = nn.ModuleList([
            nn.Conv2d(len(conn), 1, kernel_size=5, bias=True)
            for conn in connections
        ])
        self.activation = LeNetActivation()

    def forward(self, x):
        outputs = [
            conv(x[:, conn, :, :])
            for conn, conv in zip(self.connections, self.convs)
        ]
        return self.activation(torch.cat(outputs, dim=1))


# ============================================================
# 4. RBF OUTPUT LAYER
#
#    y_i = sum_j (x_j - w_ij)^2
#
#    Euclidean distance, not a dot product. SMALLER IS BETTER.
#    The prototype vectors are hand-designed 7x12 bitmaps of each
#    digit (hence 84 inputs) and are held FIXED, which is what
#    prevents the collapsing solution described in the paper.
# ============================================================

class RBFLayer(nn.Module):
    def __init__(self, prototypes):
        super().__init__()
        if prototypes.ndim != 2 or prototypes.shape[1] != 84:
            raise ValueError("prototypes must have shape [num_classes, 84]")
        self.register_buffer("prototypes", prototypes.float())

    def forward(self, x):
        # x: [B, 84]   prototypes: [C, 84]   ->  [B, C]
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)
        return torch.sum(diff ** 2, dim=2)


# ============================================================
# 5. RBF PROTOTYPES — 7 wide x 12 tall stylised digit bitmaps
#
#    Paper: components "were set to +1 or -1 ... designed to
#    represent a stylized image of the corresponding character
#    class drawn on a 7x12 bitmap (hence the number 84)."
#
#    '#' -> +1 (ink), '.' -> -1 (paper).
#
#    NOTE: the paper does not publish the exact bitmaps for digits
#    (Fig. 3 shows the full ASCII set). These are drawn to match the
#    paper's described property: visually similar characters get
#    similar codes, so an ambiguous input produces similar RBF
#    outputs and the ambiguity survives to a postprocessor.
# ============================================================

DIGIT_BITMAPS = {
    0: [".#####.",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        ".#####."],

    1: ["...##..",
        "..###..",
        ".####..",
        "...##..",
        "...##..",
        "...##..",
        "...##..",
        "...##..",
        "...##..",
        "...##..",
        "...##..",
        ".######"],

    2: [".#####.",
        "##...##",
        "##...##",
        ".....##",
        "....##.",
        "...##..",
        "..##...",
        ".##....",
        "##.....",
        "##.....",
        "##.....",
        "#######"],

    3: [".#####.",
        "##...##",
        ".....##",
        ".....##",
        "....##.",
        "..####.",
        "....##.",
        ".....##",
        ".....##",
        "##...##",
        "##...##",
        ".#####."],

    4: ["....##.",
        "...###.",
        "..####.",
        ".##.##.",
        "##..##.",
        "##..##.",
        "#######",
        "....##.",
        "....##.",
        "....##.",
        "....##.",
        "...####"],

    5: ["#######",
        "##.....",
        "##.....",
        "##.....",
        "######.",
        "....##.",
        ".....##",
        ".....##",
        ".....##",
        "##...##",
        "##...##",
        ".#####."],

    6: ["..####.",
        ".##..##",
        "##.....",
        "##.....",
        "##.....",
        "######.",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        ".#####."],

    7: ["#######",
        "##...##",
        ".....##",
        "....##.",
        "....##.",
        "...##..",
        "...##..",
        "..##...",
        "..##...",
        ".##....",
        ".##....",
        ".##...."],

    8: [".#####.",
        "##...##",
        "##...##",
        "##...##",
        ".#####.",
        ".#####.",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        ".#####."],

    9: [".#####.",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        ".######",
        ".....##",
        ".....##",
        ".....##",
        "##...##",
        ".#####."],
}


def build_rbf_prototypes(bitmaps=DIGIT_BITMAPS, num_classes=10):
    """Convert the ASCII bitmaps into a [num_classes, 84] tensor of +-1."""
    rows = []
    for cls in range(num_classes):
        grid = bitmaps[cls]
        if len(grid) != 12 or any(len(r) != 7 for r in grid):
            raise ValueError(f"bitmap for class {cls} must be 12 rows of 7 chars")
        flat = [1.0 if ch == "#" else -1.0 for row in grid for ch in row]
        rows.append(flat)                       # 7 * 12 = 84
    return torch.tensor(rows, dtype=torch.float32)


# ============================================================
# 6. THE MODEL
# ============================================================

class LeNet5(nn.Module):
    def __init__(self, rbf_prototypes=None, num_classes=10):
        super().__init__()

        if rbf_prototypes is None:
            rbf_prototypes = build_rbf_prototypes(num_classes=num_classes)

        # C1:  1 x 32 x 32  ->  6 x 28 x 28      (32 - 5 + 1 = 28)
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, bias=True)
        self.c1_activation = LeNetActivation()

        # S2:  6 x 28 x 28  ->  6 x 14 x 14
        self.s2 = SubsamplingLayer(6)

        # C3:  6 x 14 x 14  -> 16 x 10 x 10      (14 - 5 + 1 = 10)
        self.c3 = C3Layer()

        # S4: 16 x 10 x 10  -> 16 x  5 x  5
        self.s4 = SubsamplingLayer(16)

        # C5: 16 x  5 x  5  -> 120 x 1 x 1       (5 - 5 + 1 = 1)
        #
        # Declared convolutional, not fully connected: if the input
        # were larger, this layer would produce a map bigger than 1x1
        # without retraining. That is what lets the same network scan
        # a whole line of characters.
        self.c5 = nn.Conv2d(16, 120, kernel_size=5, bias=True)
        self.c5_activation = LeNetActivation()

        # F6: 120 -> 84      (84 = 7 x 12, the bitmap size)
        self.f6 = nn.Linear(120, 84, bias=True)
        self.f6_activation = LeNetActivation()

        self.rbf = RBFLayer(rbf_prototypes)

        self._initialise_weights()

    def _initialise_weights(self):
        """
        Paper, Appendix A: weights drawn uniformly from
        +- 2.4 / F_in, where F_in is the layer's fan-in.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                fan_in = (
                    module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                    if isinstance(module, nn.Conv2d)
                    else module.in_features
                )
                bound = 2.4 / fan_in
                nn.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.c1_activation(self.c1(x))
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.c5_activation(self.c5(x))
        x = torch.flatten(x, start_dim=1)
        x = self.f6_activation(self.f6(x))
        return self.rbf(x)                      # DISTANCES, not logits


# ============================================================
# 7. LOSS — the MAP criterion (paper, eq. 9)
#
#    loss = d_correct + log( exp(-j) + sum_i exp(-d_i) )
#
#    First term pushes the correct class's penalty DOWN.
#    Second term pulls the incorrect classes' penalties UP, which
#    creates competition between classes. Plain MSE has no such
#    term, which is why it admits the degenerate solution where all
#    RBF centres coincide and the network ignores its input.
#
#    j is a positive constant acting as a "rubbish class" penalty.
#    It also caps how far an already-rejected class can be pushed.
# ============================================================

def map_loss(distances, targets, j=0.1):
    correct = distances.gather(1, targets.unsqueeze(1)).squeeze(1)

    # logsumexp over [-d_0, ..., -d_C, -j]  (numerically stable)
    rubbish = torch.full(
        (distances.size(0), 1), -float(j),
        device=distances.device, dtype=distances.dtype,
    )
    competitive = torch.logsumexp(torch.cat([-distances, rubbish], dim=1), dim=1)

    return (correct + competitive).mean()


def mse_loss(distances, targets):
    """
    The simpler criterion the paper mentions first. Safe ONLY because
    the RBF prototypes are frozen -- if they were trainable this loss
    would collapse all centres onto one point.
    """
    return distances.gather(1, targets.unsqueeze(1)).squeeze(1).mean()


# ============================================================
# 8. INPUT NORMALISATION
#
#    Paper: background (white) -> -0.1, foreground (black) -> 1.175,
#    "This makes the mean input roughly zero and the variance roughly
#    one, which accelerates learning."
#
#    MNIST is 28x28; pad by 2 on each side to reach 32x32 so that
#    stroke endpoints near the border still sit at the centre of a
#    high-level detector's receptive field.
# ============================================================

BACKGROUND = -0.1
FOREGROUND = 1.175


def normalise_input(x):
    """x: [B, 1, 28, 28] in [0, 1]  ->  [B, 1, 32, 32] in [-0.1, 1.175]"""
    x = F.pad(x, (2, 2, 2, 2), value=0.0)
    return x * (FOREGROUND - BACKGROUND) + BACKGROUND


def mnist_transform():
    """torchvision transform matching the paper's preprocessing."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),                                  # [0,1], 28x28
        transforms.Pad(2, fill=0.0),                            # -> 32x32
        transforms.Lambda(lambda t: t * (FOREGROUND - BACKGROUND) + BACKGROUND),
    ])


# ============================================================
# 9. PREDICTION — argMIN, because RBF returns distances
# ============================================================

@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.argmin(model(x), dim=1)


@torch.no_grad()
def reject_aware_predict(model, x, threshold):
    """
    The capability softmax gave up. If every prototype is far away,
    the input probably is not a character at all.

    Returns predictions with -1 marking a rejection.
    """
    model.eval()
    distances = model(x)
    best, index = distances.min(dim=1)
    return torch.where(best > threshold, torch.full_like(index, -1), index)


# ============================================================
# 10. SELF-CHECK
# ============================================================

def parameter_report(model):
    # Every expected figure here is published in the paper (§II.A, §II.B).
    expected = {"c1": 156, "s2": 12, "c3": 1516, "s4": 32, "c5": 48120, "f6": 10164}
    print(f"{'layer':<6}{'actual':>10}{'paper':>10}")
    total = matched = 0
    for name, want in expected.items():
        got = sum(p.numel() for p in getattr(model, name).parameters())
        total += got
        matched += got == want
        flag = "" if got == want else match(False)("   <-- MISMATCH")
        print(f"{name:<6}{match(got == want)(f'{got:>10,}')}{want:>10,}{flag}")
    print(f"{'total':<6}{match(total == 60000)(f'{total:>10,}')}{60000:>10,}")

    summary = f"{matched}/{len(expected)} layer counts match the paper"
    summary += ", total matches" if total == 60000 else ", total DOES NOT match"
    print(f"\n{match(matched == len(expected) and total == 60000)(summary)}")
    return total


if __name__ == "__main__":
    torch.manual_seed(0)

    model = LeNet5()
    parameter_report(model)

    # shape check on a batch of two 32x32 images
    x = torch.randn(2, 1, 32, 32)
    d = model(x)
    print(f"\noutput shape {tuple(d.shape)}  (batch, classes) -- these are DISTANCES")
    print(f"predictions  {predict(model, x).tolist()}")

    # loss check
    y = torch.tensor([3, 7])
    print(f"MAP loss     {map_loss(d, y).item():.4f}")
    print(f"MSE loss     {mse_loss(d, y).item():.4f}")

    # prototype sanity: confusable digits should sit closer together
    p = build_rbf_prototypes()
    import itertools
    pairs = [(a, b, torch.norm(p[a] - p[b]).item()) for a, b in itertools.combinations(range(10), 2)]
    pairs.sort(key=lambda t: t[2])
    print("\nclosest prototype pairs (should be visually similar digits):")
    for a, b, dist in pairs[:4]:
        print(f"  {a} and {b}: {dist:.2f}")
