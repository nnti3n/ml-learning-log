# ML learning log

A personal log of learning machine learning from the fundamentals up, with the
goal of working on ML infrastructure: training runs, serving, and efficiency.

The path: work through _Understanding Deep Learning_ by Simon Prince (the book
and its notebooks), then build a small GPT from scratch.

## Layout

One numbered folder per entry. Each entry has a `README.md` (notes in my own
words) and the code that goes with it.

```
00-setup/              environment check (PyTorch on the Apple GPU)
01-gradient-descent/   fit a line to three points by hand-written gradient descent
```

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python 00-setup/mps_check.py
```

The check prints the versions, trains one linear layer on the `mps` device,
and exits non-zero if the Apple GPU is not available. Expected output is in
`00-setup/mps_check.out.txt`.
