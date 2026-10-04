# CouponEvo

[English](README.md) | [简体中文](README.zh-CN.md)

**让 Agent 在固定随机实验和券预算内，迭代 uplift 模型与发券策略。**

发券问题不只是预测谁会购买，而是判断**谁会因为收到优惠而改变行为**，以及这份增量是否值得付出券成本。CouponEvo 将它组织成可复现的离线实验：Agent 提出算法与选人策略的改动，独立评估器在相同数据和预算下检验，再用未参与搜索的测试集判断结果能否站得住。

## 设计思路：借鉴 RSI 的可执行反馈闭环

[OpenRSI](https://github.com/FrontisAI/OpenRSI)将可执行的程序搜索、经验积累和对“改进者”自身的训练连接起来，作为走向递归自我改进的路径。CouponEvo 取其中**可执行搜索**这一层，限定在优惠券营销任务：Agent 修改 uplift 模型或预算内发券策略，在固定随机实验上运行，读取独立验证反馈，再决定下一轮尝试。已完成搜索的经验只在同一数据集和任务下复用。

这里演化的是**候选算法**，不是 Agent 本身。当前版本不训练 Agent 权重或提案策略，候选分数变好也不等于实现完整 RSI。数据、目标、预算和评估器不受 Agent 修改，保证各轮方案可比；验证集允许反复探索，最终只把冻结的冠军送入独立测试集。下面的公开实验正说明了这条边界的必要性：验证集胜出，不代表留出集能证实提升。

## Agent 能做什么，实验固定什么

| Agent 可改 | 执行器固定 |
| --- | --- |
| 基于 EconML `TLearner` 的候选代码、PyTorch 结果模型，以及可选的 `choose_policy` 预算内选人函数 | 数据与 manifest 哈希、处理概率、训练/验证/测试切分、目标、预算和评估器 |
| 根据验证集反馈提出下一轮假设，自选 `draft`、`improve`、`debug` 或 `crossover` | 独立的策略价值估计、预算检查、逐用户配对比较与最终留出集 |
| 将缺少的干预前画像记入 `feature_gaps.md` | 当前数据集；缺失特征只供后续人工完善数据，不在本轮虚构 |

候选代码在隔离的 Docker 容器中执行。每轮假设、代码快照、指标及失败原因保存在搜索日志里，同一任务可以断点续跑。可选的历史经验只读取**已完成搜索的验证集记录**，且必须匹配数据文件内容、manifest、目标、预算、切分种子和评估器。数据一变，经验就进入新范围。

核心评估对象是**发券策略的增量价值**。评估器根据随机分组及处理概率，用逆概率加权（IPW）估计策略增量，在同一批用户上计算新旧策略的配对差值和区间；Qini/AUUC 仅用于排序诊断。搜索结束后，`finalize` 冻结验证集冠军，只在独立测试集与初始策略比较一次，并计算配对 bootstrap 区间。

配对估计对评估用户平均 `(新策略 − 初始策略) × [干预 × 结果 / p − 对照 × 结果 / (1 − p)]`，其中 `p` 是有来源说明的处理概率。

```text
固定随机实验 → Agent 提议候选代码 → 容器训练与验证
                  ↑                    ↓
                  └── 假设与指标反馈 ──┘
                            ↓
                    冻结候选 → 最终留出集
```

## 公开实验给出的证据

在 [84,534 行 Starbucks 随机促销数据](examples/README.md)上，DeepSeek Agent 做了两轮搜索；EconML/PyTorch 候选模型在 RTX 5090 上运行。目标是固定**假设发送成本**预算下的增量购买转化。

| 候选 | 验证集 IPW 增量转化／候选用户 |
| --- | ---: |
| 初始方案：T-learner + PyTorch 岭回归 | 0.003312 |
| Agent 提案：PyTorch MLP 结果模型 | **0.004140** |
| Agent 提案：MLP 集成 | 0.002957 |

验证集冠军在独立测试集相对初始方案的**逐用户配对差值**为 `+0.000118`，95% bootstrap 区间是 `[-0.001065, +0.001301]`。区间跨零，因此**这次实验没有证实最终提升**。项目把“验证集看起来更好”和“最终证据支持更好”明确区分。详见[完整实验记录](docs/research/agent-search-gpu-2026-10-03.md)。

Starbucks 不包含 App 促活标签、实际券核销成本或用户级毛利，不能拿这个结果声称促活或真实净收益提升。Criteo、X5 RetailHero 还用于 [GPU 模型运行检查](docs/research/gpu-validation-2026-10-03.md)；公开数据不足以确认原始处理概率时，不输出因果策略收益结论。

## Dataset manifest

输入是一份固定的随机对照数据，每个随机分配单位（通常为用户）占一行，至少要有干预/对照分组、有来源说明的分组概率、干预前特征和一个结果指标。有用户 ID 时可检查重复用户；没有时报告会提示这一限制。[Starbucks manifest](examples/starbucks.json) 给出了字段映射。评估器禁止把分组或结果字段作为特征；如果有逐行特征时点，还可检查特征是否早于分组。

人数预算不要求成本标签。成本预算可使用实际券成本，或明确标记为假设的固定发送成本；要评估真实净毛利，还需要相应的毛利与成本口径。接入业务实验时参照[数据约定](docs/business-rct-contract.md)，公开数据的来源与限制见[数据说明](examples/README.md)。原始数据和运行产物不提交到仓库。

## 跑一次 Agent 搜索

准备 Python 3.12+、[uv](https://docs.astral.sh/uv/)，并按[数据说明](examples/README.md)将 Starbucks CSV 放在 `data/starbucks-training.csv`。下面的 GPU 示例需要 Linux CUDA 主机、匹配的 PyTorch CUDA 版本和 Docker。先在环境变量中设置 `AGENT_API_KEY`；[Agent 配置示例](examples/agent.deepseek.json)只保存服务 URL 和模型名，不保存密钥。

```bash
uv sync --frozen --python 3.12
docker build -f Dockerfile.sandbox -t couponevo-sandbox:py312-cuda128 .
export SANDBOX_IMAGE="$(docker image inspect couponevo-sandbox:py312-cuda128 --format '{{.Id}}')"
uv run --frozen python -m couponevo.cli search examples/starbucks.json \
  --budget-kind cost --budget 0.03 --objective conversion --seed 42 \
  --max-steps 3 --search-id starbucks-01 --device cuda \
  --agent-config examples/agent.deepseek.json --sandbox-image "$SANDBOX_IMAGE"
uv run --frozen python -m couponevo.cli finalize examples/starbucks.json \
  --budget-kind cost --budget 0.03 --objective conversion --seed 42 \
  --search-id starbucks-01 --device cuda --bootstrap-reps 2000 \
  --sandbox-image "$SANDBOX_IMAGE"
```

Agent 可通过 provider URL、模型名和密钥环境变量接入兼容服务。示例为代码迭代和报告分析请求 `high`，反常结果触发 `max` 复核；服务端须支持相应参数。报告与候选快照写入不提交到 Git 的 `runs/`。单次基线评估、断点续跑等参数见 `python -m couponevo.cli --help`。

## 适用范围与参与

当前支持单一干预对照、一次选人决策、固定 CSV 与离线随机实验评估；不执行线上发券，也未验证真实 App 促活效果。

进一步阅读[架构说明](docs/architecture.md)、[公开数据调研](docs/research/open-uplift-datasets.md)和[贡献指南](CONTRIBUTING.md)。测试命令：`uv run --frozen python -m unittest discover -s tests -v`。代码采用 [Apache-2.0](LICENSE)；公开数据遵循各自的授权条款。
