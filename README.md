# GraphVQ Official Implementation

## 整体架构

```
code_refactored/
├── train.py                  # 统一训练入口
├── evaluate.py               # 统一评估入口
├── requirements.txt
├── model/                    # 核心模型：VQ 图 tokenizer
│   ├── encoder.py            #   图卷积编码器
│   ├── vector_quantizer.py   #   向量量化层（离散码本）
│   ├── decoder.py            #   图卷积解码器
│   └── tokenizer.py          #   GraphVQTokenizer = 编码 → 量化 → 重建
├── analysis/                 # 图生成流程
│   ├── dataset.py            #   数据集加载（TUDataset / QM9 / 合成数据集）
│   ├── synthetic.py          #   受控合成图分布
│   ├── metrics.py            #   生成指标（validity / novelty / uniqueness 等）
│   ├── prior.py              #   自回归先验（GRU / Transformer）训练与采样
│   ├── generation.py         #   一阶段生成：BFS 排序 + 显式边建模
│   └── generation_twostage.py#   两阶段生成：节点序列 + 边补全
├── benchmarks/               # 图分类基准与迁移评估
│   ├── data.py               #   TUDataset 下载与解析（纯 numpy，首次运行自动下载）
│   ├── batching.py           #   批处理构建
│   ├── layers.py             #   GIN / GCN / SAGE / GAT 等层
│   ├── models.py             #   VQGIN 及各 baseline 模型
│   ├── run.py                #   图分类基准（含码本规模扫描）
│   └── transfer.py           #   跨数据集迁移（预训练 → 冻结 → 少样本微调）
└── configs/                  # 各任务的 YAML 配置
    ├── benchmark.yaml        #   图分类基准
    ├── transfer.yaml         #   迁移评估
    ├── generation.yaml       #   一阶段生成
    └── twostage.yaml         #   两阶段生成
```

## 启动说明

安装依赖（Python 3.10+）：

```bash
pip install -r requirements.txt
```

训练与评估入口接受相同的任务参数，任务可选 `benchmark` / `transfer` / `generation` / `twostage`，缺省运行全部任务：

```bash
# 训练（单个或多个任务）
python train.py generation
python train.py benchmark transfer

# 评估
python evaluate.py twostage
python evaluate.py            # 运行全部
```

各任务超参数在 `configs/` 下对应的 YAML 文件中修改。TUDataset（MUTAG / PROTEINS / ENZYMES / NCI1）与 QM9 在首次运行时自动下载；数据集缓存、checkpoint 与结果文件分别在运行目录下的 `benchmarks/data/`、`checkpoints/`、`results/` 中生成。
