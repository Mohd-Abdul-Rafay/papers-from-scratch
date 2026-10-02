"""
AlexNet — faithful implementation of Krizhevsky, Sutskever & Hinton (2012),
"ImageNet Classification with Deep Convolutional Neural Networks."

Verified: 60,965,224 parameters, matching the paper's "60 million parameters".

Key fidelity points, each a deliberate choice rather than an accident:

  * LRN comes BEFORE max-pooling, not after. Paper, §3.5: "Response-normalization
    layers follow the first and second convolutional layers. Max-pooling layers
    ... follow both response-normalization layers as well as the fifth
    convolutional layer." Order is Conv -> ReLU -> LRN -> MaxPool.

  * groups=2 on conv2, conv4, conv5. This is the two-GPU split. It was forced by
    the GTX 580's 3 GB of memory, not chosen for modelling reasons -- but the
    paper reports it IMPROVED accuracy, and observed the two halves specialising
    (one learned largely colour-agnostic features, the other colour-selective).
    Ungrouped, those three layers would cost 2,827,136 params instead of
    1,414,016.

  * Overlapping pooling: 3x3 windows at stride 2, so z=3 > s=2. Paper reports
    this cut top-1 error by 0.4% and top-5 by 0.3%, and that such models were
    "slightly more difficult to overfit".

  * Dropout on the OUTPUTS of the first two FC layers (fc1, fc2) -- not on the
    conv features feeding fc1, and not on fc3. Paper, §4.2: dropout sets "to zero
    the output of each hidden neuron", used "in the first two fully-connected
    layers". Those two layers hold 89.4% of all parameters; the entire
    convolutional stack holds 3.8%. You regularise where the parameters are.

  * LRN alpha is passed as 5e-4, not 1e-4. PyTorch's LocalResponseNorm divides
    alpha by size; the paper's formula does not. 5e-4 / 5 = 1e-4.

  * Bias initialised to 1 on conv2, conv4, conv5 and the fully-connected hidden
    layers (fc1, fc2); 0 elsewhere, including fc3.
    Paper §5: this "accelerates the early stages of learning by providing the
    ReLUs with positive inputs" -- i.e. it avoids starting in the dead zone.

Parameter / compute split (computed, not quoted):

    CONV    2,334,080 params ( 3.8%)   665,784,864 mult-adds (91.9%)
    FC     58,631,144 params (96.2%)    58,621,952 mult-adds ( 8.1%)

That inversion is the single most useful fact in the architecture: the conv
stack does nearly all the work and holds almost none of the weights.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()

        # LRN: the paper uses k=2, n=5, alpha=1e-4, beta=0.75 (§3.3), with
        #     b = a / (k + alpha * sum(a^2))^beta
        # PyTorch's LocalResponseNorm computes alpha / size * sum(a^2), so
        # alpha=5e-4 here gives the paper's 5e-4 / 5 = 1e-4.
        lrn = dict(size=5, alpha=5e-4, beta=0.75, k=2.0)

        self.features = nn.Sequential(OrderedDict([
            # ----------------------------------------------------------
            # conv1:  3 x 224 x 224  ->  96 x 55 x 55
            #
            # 11x11 at stride 4. The large kernel and aggressive stride
            # were a compute concession, not a design preference: 224x224
            # at stride 1 was not affordable in 2012. Note the network
            # discards three of every four input positions here, before
            # any features have been built.
            # ----------------------------------------------------------
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)),
            ("relu1", nn.ReLU(inplace=True)),
            ("lrn1",  nn.LocalResponseNorm(**lrn)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2)), # -> 96 x 27 x 27

            # ----------------------------------------------------------
            # conv2:  96 x 27 x 27  ->  256 x 27 x 27
            # groups=2: each of the 256 filters sees only 48 of the 96
            # input maps. This is the GPU split.
            # ----------------------------------------------------------
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("lrn2",  nn.LocalResponseNorm(**lrn)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2)), # -> 256 x 13 x 13

            # ----------------------------------------------------------
            # conv3:  256 x 13 x 13  ->  384 x 13 x 13
            # The ONLY layer where the two GPUs communicate -- hence no
            # grouping here. Every filter sees all 256 input maps.
            # ----------------------------------------------------------
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),

            # conv4:  384 x 13 x 13  ->  384 x 13 x 13   (split again)
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),

            # conv5:  384 x 13 x 13  ->  256 x 13 x 13   (split)
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2)), # -> 256 x 6 x 6
        ]))

        # Dropout zeroes the OUTPUTS of fc1 and fc2 (paper §4.2), so it sits
        # after each hidden layer's ReLU -- not on the 9216 conv features.
        self.classifier = nn.Sequential(OrderedDict([
            # 256 * 6 * 6 = 9216
            ("fc1",   nn.Linear(9216, 4096)),               # 37,752,832 params
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout(p=dropout)),

            ("fc2",   nn.Linear(4096, 4096)),               # 16,781,312 params
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout(p=dropout)),

            ("fc3",   nn.Linear(4096, num_classes)),        #  4,097,000 params
        ]))

        self._initialise_weights()

    def _initialise_weights(self):
        """
        Paper §5: weights from a zero-mean Gaussian with std 0.01.
        Biases set to 1 in conv2, conv4, conv5 and the fully-connected
        hidden layers fc1, fc2 (to give ReLUs positive input early in
        training), 0 elsewhere -- fc3 is the output layer, not hidden.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

        # bias = 1 on the layers the paper names
        for layer in (
            self.features.conv2, self.features.conv4, self.features.conv5,
            self.classifier.fc1, self.classifier.fc2,
        ):
            nn.init.ones_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)                           # logits


# ==============================================================
# TRAINING CONFIGURATION — paper §5
# ==============================================================

def build_optimiser(model, lr=0.01, momentum=0.9, weight_decay=5e-4):
    """
    SGD, batch 128, momentum 0.9, weight decay 0.0005.

    The paper is explicit that the weight decay is not merely a
    regulariser here: "weight decay reduced the model's training error",
    i.e. it mattered for optimisation, not just generalisation.

    Learning rate started at 0.01 and was divided by 10 whenever
    validation error stopped improving -- three times over ~90 epochs.
    """
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


# Loss is plain cross-entropy over the 1000 logits.
criterion = nn.CrossEntropyLoss()


# ==============================================================
# DATA AUGMENTATION — paper §4.1
# ==============================================================

def train_transform():
    """
    Two augmentations, both described as essential to controlling
    overfitting at 60M parameters:

      1. Random 224x224 crops from the 256x256 image, plus horizontal
         flips. The paper notes this enlarges the training set by a
         factor of 2048.
      2. PCA colour augmentation ("fancy PCA") -- add multiples of the
         principal components of the RGB covariance across ImageNet,
         scaled by a random Gaussian. Reported to cut top-1 error by
         over 1%.
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # PCALighting(alphastd=0.1) would go here — see below.
    ])


def test_transform():
    """
    At test time the paper averages predictions over ten patches:
    four corner crops, the centre crop, and the horizontal mirror of
    each. TenCrop returns all ten; average the logits yourself.
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(
                transforms.functional.to_tensor(c)
            ) for c in crops
        ])),
    ])


class PCALighting:
    """
    The paper's PCA colour augmentation. Eigenvalues and eigenvectors
    are those of the ImageNet RGB covariance matrix, as published.

    To each RGB pixel, add  [p1 p2 p3] @ [a1*l1, a2*l2, a3*l3]^T
    where ai ~ N(0, 0.1), drawn once per image (not per pixel).
    """
    EIGVAL = torch.tensor([0.2175, 0.0188, 0.0045])
    EIGVEC = torch.tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])

    def __init__(self, alphastd: float = 0.1):
        self.alphastd = alphastd

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.alphastd == 0:
            return img
        alpha = torch.randn(3) * self.alphastd
        rgb = (self.EIGVEC * alpha.view(1, 3) * self.EIGVAL.view(1, 3)).sum(dim=1)
        return img + rgb.view(3, 1, 1)


@torch.no_grad()
def ten_crop_predict(model, crops: torch.Tensor) -> torch.Tensor:
    """crops: [B, 10, 3, 224, 224] -> averaged logits [B, num_classes]"""
    model.eval()
    b, n = crops.shape[:2]
    logits = model(crops.view(-1, *crops.shape[2:]))
    return logits.view(b, n, -1).mean(dim=1)


# ==============================================================
# SELF-CHECK
# ==============================================================

def parameter_report(model):
    expected = [
        ("conv1", 34_944), ("conv2", 307_456), ("conv3", 885_120),
        ("conv4", 663_936), ("conv5", 442_624),
        ("fc1", 37_752_832), ("fc2", 16_781_312), ("fc3", 4_097_000),
    ]
    layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    print(f"{'layer':<8}{'actual':>13}{'paper':>13}{'share':>9}")
    total = sum(p.numel() for p in model.parameters())
    conv_total = fc_total = 0

    for (name, want), layer in zip(expected, layers):
        got = sum(p.numel() for p in layer.parameters())
        flag = "" if got == want else "  <-- MISMATCH"
        print(f"{name:<8}{got:>13,}{want:>13,}{100*got/total:>8.1f}%{flag}")
        if isinstance(layer, nn.Conv2d):
            conv_total += got
        else:
            fc_total += got

    print(f"{'TOTAL':<8}{total:>13,}{60_965_224:>13,}")
    print(f"\nCONV  {conv_total:>12,}  ({100*conv_total/total:4.1f}% of parameters)")
    print(f"FC    {fc_total:>12,}  ({100*fc_total/total:4.1f}% of parameters)")
    return total


def shape_trace(model):
    """Print the tensor shape after every layer in features."""
    x = torch.zeros(1, 3, 224, 224)
    print(f"{'input':<22}{tuple(x.shape)}")
    for layer in model.features:
        x = layer(x)
        name = layer.__class__.__name__
        if name in ("Conv2d", "MaxPool2d", "LocalResponseNorm"):
            print(f"{name:<22}{tuple(x.shape)}")
    x = torch.flatten(x, 1)
    print(f"{'flatten':<22}{tuple(x.shape)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    model = AlexNet()

    parameter_report(model)
    print()
    shape_trace(model)

    out = model(torch.randn(2, 3, 224, 224))
    print(f"\noutput shape {tuple(out.shape)}  (batch, 1000 logits)")

    # confirm LRN sits before pooling, as the paper specifies
    order = [m.__class__.__name__ for m in model.features]
    lrn = [i for i, n in enumerate(order) if n == "LocalResponseNorm"]
    pool = [i for i, n in enumerate(order) if n == "MaxPool2d"]
    ok = all(any(p > l for p in pool) for l in lrn) and lrn[0] < pool[0]
    print(f"LRN before pooling: {ok}")
