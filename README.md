# GraphVQ Official Implementation

## Overall Architecture

```
code_refactored/
├── train.py                  # Unified training entrypoint
├── evaluate.py               # Unified evaluation entrypoint
├── requirements.txt
├── model/                    # Core model: VQ graph tokenizer
│   ├── encoder.py            #   Graph convolutional encoder
│   ├── vector_quantizer.py   #   Vector quantization layer (discrete codebook)
│   ├── decoder.py            #   Graph convolutional decoder
│   └── tokenizer.py          #   GraphVQTokenizer = encode → quantize → reconstruct
├── analysis/                 # Graph generation pipelines
│   ├── dataset.py            #   Dataset loading (TUDataset / QM9 / synthetic datasets)
│   ├── synthetic.py          #   Controlled synthetic graph distributions
│   ├── metrics.py            #   Generation metrics (validity / novelty / uniqueness, etc.)
│   ├── prior.py              #   Autoregressive prior (GRU / Transformer) training and sampling
│   ├── generation.py         #   One-stage generation: BFS ordering + explicit edge modeling
│   └── generation_twostage.py#   Two-stage generation: node sequence + edge completion
├── benchmarks/               # Graph classification benchmark and transfer evaluation
│   ├── data.py               #   TUDataset download and parsing (pure numpy, auto-downloaded on first run)
│   ├── batching.py           #   Batch construction
│   ├── layers.py             #   GIN / GCN / SAGE / GAT layers
│   ├── models.py             #   VQGIN and baseline models
│   ├── run.py                #   Graph classification benchmark (incl. codebook-size sweep)
│   └── transfer.py           #   Cross-dataset transfer (pretrain → freeze → few-shot fine-tune)
└── configs/                  # YAML configs for each task
    ├── benchmark.yaml        #   Graph classification benchmark
    ├── transfer.yaml         #   Transfer evaluation
    ├── generation.yaml       #   One-stage generation
    └── twostage.yaml         #   Two-stage generation
```

## Getting Started

Install dependencies (Python 3.10+):

```bash
pip install -r requirements.txt
```

The training and evaluation entrypoints accept the same task arguments. Available tasks are `benchmark` / `transfer` / `generation` / `twostage`; all tasks run by default:

```bash
# Training (single or multiple tasks)
python train.py generation
python train.py benchmark transfer

# Evaluation
python evaluate.py twostage
python evaluate.py            # run all
```

Hyperparameters for each task can be modified in the corresponding YAML file under `configs/`. TUDataset (MUTAG / PROTEINS / ENZYMES / NCI1) and QM9 are downloaded automatically on first run; dataset caches, checkpoints, and result files are generated under `benchmarks/data/`, `checkpoints/`, and `results/` in the working directory, respectively.
