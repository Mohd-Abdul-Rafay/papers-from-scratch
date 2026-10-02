# papers-from-scratch

Faithful from-scratch PyTorch reimplementations of foundational computer vision papers, with parameter counts verified against the published figures and notes on the details most implementations get wrong.

## Implemented papers

| Model | Paper | Authors | Year | Code |
|---|---|---|---|---|
| LeNet-5 | [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791) | LeCun, Bottou, Bengio, Haffner | 1998 | [`lenet5/`](lenet5/) |
| AlexNet | [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | Krizhevsky, Sutskever, Hinton | 2012 | [`alexnet/`](alexnet/) |

Neither paper is on arXiv. The links go to the official publication pages.
Paper PDFs are not committed to this repo.

## Verification

Parameter counts and mult-adds are taken from `torchinfo` output, not hand-computed.

| Model | Params (this repo) | Params (paper) | Match |
|---|---:|---:|:---:|
| LeNet-5 | 60,000 | 60,000 (§II.A, and per layer in §II.B) | ✅ exact, all 6 layers |
| AlexNet | 60,965,224 | "60 million" (abstract) | ✅ |

Each model's README has the per-layer breakdown, and the LeNet-5 README
reconciles the paper's stated 340,908 connections against its per-layer figures.

## Scale: fourteen years apart

| | Params | Mult-adds |
|---|---:|---:|
| LeNet-5 (1998) | 60,000 | 0.33 M |
| AlexNet (2012) | 60,965,224 | 725.07 M |
| **Ratio** | **1,016×** | **2,183×** |

The compute ratio uses exact totals: 725,066,088 / 332,188. Dividing the
rounded figures (725.07 / 0.33) gives 2,197×, which is wrong.

LeNet-5's mult-adds are `torchinfo`'s count, which omits the custom S2, S4 and
RBF layers. Including them (the paper's 340,908 connections) gives a ratio of
2,127×.

## Setup

```bash
git clone https://github.com/Mohd-Abdul-Rafay/papers-from-scratch.git
cd papers-from-scratch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python lenet5/lenet5.py
python alexnet/alexnet.py
```

Each script prints a layer-by-layer parameter report against the paper and runs
a forward-pass shape check. Tested with Python 3.14, torch 2.14, torchvision 0.29,
torchinfo 1.8. Datasets go under `data/`, which is git-ignored.

## License

MIT. See [LICENSE](LICENSE).
