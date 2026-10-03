# 公开数据 GPU 验证（2026-10-03）

目标机：`Linux GPU host`，NVIDIA RTX 5090。所有下面的模型拟合与预测都在目标机使用 PyTorch CUDA 执行；CPU 与 CUDA 模型路径均使用 PyTorch。原始数据和运行产物保存在目标机 `$PROJECT_ROOT/data/`、`runs/`，均不进入 Git。

## Starbucks：完整固定 RCT 评测和 Agent 迭代

- 来源：[Udacity 发布的 `training.csv`](https://github.com/udacity/DSND_Term2/blob/master/portfolio_exercises/Experiments/data/training.csv)，84,534 行，原始文件 SHA-256 `4d48190fd0d6a65d3874fa9a9ac79d89007579716366c2cfe140ae999088aa4f`。
- 映射：`examples/starbucks.json`；七个匿名特征、随机促销分组、购买结果。课程的 0.15 是**每位被选中用户的假设发送成本**。没有真实收入、毛利、核销成本或 App 活跃结果。
- 基线命令：`python -m coupon_lab.cli run examples/starbucks.json --budget-kind cost --budget 0.03 --device cuda --seed 42 --output runs/starbucks-torch`。`runs/starbucks-torch/4b38fbe494d84719/report.json` 记录 `model_device=cuda`。验证集 16,907 人；转化策略选中 3,381 人，IPW 转化增量点估计 0.003312/评估用户，95% 正态近似区间 `[0.001707, 0.004918]`。随机策略点估计 0.001301，区间 `[-0.000288, 0.002891]`。改成统一 PyTorch 路径前的 CUDA 基线点估计相同。
- [DeepSeek V4.1 Flash](https://api-docs.deepseek.com/updates/)（API 模型名 `deepseek-flash`）Agent 在同一固定数据和 GPU 上修改候选算法，得到 `runs/agent-torch/34bb395ed6c4c838/`。新预测 SHA-256 与基线不同，候选代码在 CPU/CUDA 均保留 PyTorch 求解；转化策略点估计 0.003667，区间 `[0.002012, 0.005322]`。`analysis.json` 记录代码迭代与初评为 `high`；因无法权衡真实成本与增量收益，触发 `cost_tradeoff_unclear` 的 `max` 复核。当前报告没有前后策略**配对差值**的区间，不能据点估计变化断言模型提升。

## Criteo：GPU 模型检查

- 来源：[Criteo 官方公开数据卡](https://huggingface.co/datasets/criteo/criteo-uplift)。从官方 dataset viewer 的公开行接口，按固定种子 `20261003` 抽取 50 个互不重叠的 100 行区块，共 5,000 行；`source_row_idx` 保留原始公开行号。目标机文件 `data/criteo-sample.csv`，SHA-256 `958cf5394a6954dec9d659b9df3337761c655cb9e6cece4a759a03f6e69ed807`。
- 用 `f0`—`f11` 训练 CUDA T-learner，固定种子 42 切分为 4,000 训练、1,000 预测；`visit` 与 `conversion` 两列预测均为有限数，`model_device=cuda`，GPU 峰值已分配 9,011,712 字节。目标机检查记录：`runs/criteo-gpu-smoke.json`。
- 样本中 treatment=1 有 4,200 行、对照 800 行，转化仅 17 行。公开数据经过非均匀抽样，不能由这份样本恢复原始随机试验的分组概率；本轮不输出 IPW 因果效果或真实成本收益结论。

## X5 RetailHero：GPU 模型检查

- 来源：[scikit-uplift 的 X5 数据说明与加载代码](https://www.uplift-modeling.com/en/latest/_modules/sklift/datasets/datasets.html)。目标机直接下载 `uplift_train.csv.gz` 与 `clients.csv.gz`；源站公布的 MD5 分别为 `2720bbb659daa9e0989b2777b6a42d19`、`b9cdeb2806b732771de03e819b3354c5`，实际校验一致。对应 SHA-256 分别为 `0a218e864a1ec1dfe26e980a082b81d5f46d199c4a6ae32abbaa95265af1c2c7`、`bfca1719b862735ae9816ed9bbba6924931c6a4dff761c481efe9addff458429`。
- 用客户 ID 合并 200,039 条分组记录与客户画像，只选 `age`、`gender` 做特征；不使用可能存在时间边界疑问的首次兑付日期。固定种子 42 切分为 160,000 训练、40,039 预测；CUDA T-learner 输出均为有限数，`model_device=cuda`，GPU 峰值已分配 15,250,432 字节。目标机检查记录：`runs/x5-gpu-smoke.json`。
- 公开说明给出干预后购买标签，却未提供足以核实随机分组机制与处理概率的信息。本轮只验证模型可运行，不计算 IPW 因果效果或成本收益。

数据授权与其他候选见[公开数据调研](open-uplift-datasets.md)。

## EconML T-learner、Agent 调优与配对区间

- 在目标机 GPU 环境安装 `econml==0.17.0`。默认模型使用 EconML `TLearner` 组织处理/对照两臂，底层 `TorchRidgeRegressor` 在 CUDA 上拟合与预测。DeepSeek `deepseek-flash` Agent 将其改为 PyTorch MLP 结果模型，仍调用 EconML `TLearner`。生成代码已保存为 [Agent 候选](../../examples/agent-mlp-candidate.py)；目标机快照在 `runs/econml-agent/01d1aa77a38fec5c/candidate.py`。
- Agent 首次运行暴露 MLP 初始化和批次顺序的随机性；执行器现按 `--seed` 固定 PyTorch RNG。相同 GPU test 命令重复执行后，`report.json` SHA-256 两次均为 `7f7abff47f60fc33ee8dcf7a2241d300aee0d008c63cf92647e2f3258e34d1ea`。
- 固定种子后，Starbucks 验证集 16,907 人：Agent MLP 与 EconML 岭回归基线的转化策略配对 IPW 差值为 `+0.000237/人`，95% 正态近似区间 `[-0.001435, +0.001909]`；文件为 `runs/econml-seeded-validation/f8ea8926029bd4fe/report.json`。
- 冻结两份代码后在独立 test 集 16,907 人比较一次：差值为 `-0.000946/人`，95% 配对区间 `[-0.002519, +0.000626]`；[最终报告 JSON](econml-agent-starbucks-final.json) 同步自目标机 `runs/econml-seeded-final/98cf3c8ac059e3ff/report.json`。两个区间均跨零，**没有 Agent 提升的证据**。结果衡量的是购买转化，不是 App 促活或真实净利润；$0.15 是课程假设成本。
- 对 test 集同一逐用户配对差值做 2,000 次有放回 bootstrap，百分位 95% 区间为 `[-0.002721, +0.000591]`，与正态近似结论一致；固定随机种子 `20261003`，详见 [bootstrap 校验 JSON](econml-agent-starbucks-bootstrap.json)。两种区间均跨零，不支持“新模型更好”。
- 目标机测试 `python -m unittest discover -s tests -q`：46 个测试通过，包括 EconML CPU/CUDA 路径、同策略零差值、逐用户配对计算、随机候选的可重复运行、仅有收入和实际成本时的净收入策略，以及 Agent 不得移除 EconML learner。
- X5 200,039 行上，以 `age`、`gender` 两个特征训练 160,031 行、预测 40,008 行；EconML + CUDA 输出全为有限数，GPU 峰值 13,007,360 字节，记录 `runs/x5-econml-smoke.json`。Criteo v2.1 公开 5,000 行样本上，以 12 个匿名特征训练 4,000 行、预测 1,000 行；CUDA 输出全为有限数，记录 `runs/criteo-econml-smoke.json`。这两次只做运行检查，不给因果效果或真实成本结论。
- 尝试从 [Criteo 官方 v2.1 下载地址](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) 向目标机取完整 296 MiB 文件；当时传输速度约 17 KiB/s，预估需约五小时，故停止并将 1.5 MiB 不完整文件标为 `data/criteo-uplift-v2.1.csv.gz.partial`。完整文件未参加本轮测试。

## 补齐预算策略与数据校验后的复跑

- 目标机按 `uv.lock` 创建 Python 3.12 环境，使用 PyTorch `2.14.1+cu130` 和 NVIDIA RTX 5090。`./.venv/bin/python -m unittest discover -s tests -q`：62 项通过，包含 CUDA 路径。旧目标机默认 `base` 环境缺依赖；本次复跑使用项目 `.venv`。
- Starbucks 原始文件 SHA-256 仍为 `4d48190fd0d6a65d3874fa9a9ac79d89007579716366c2cfe140ae999088aa4f`。新 manifest 记录了随机分组概率的官方 notebook 引用；七个匿名特征仍无逐行时间戳，报告标为 `declared_only`，不能使用 `--strict-data` 伪称已核验时点。
- EconML + PyTorch CUDA 基线验证集运行：`runs/completion-baseline/1b324659887ae3b3/report.json`，16,907 人，转化策略相对随机发促销的同用户配对 IPW 差值 `+0.002011/人`，95% 正态近似区间 `[+0.000113,+0.003909]`。这是验证集诊断，不作为最终提升证据。
- 冻结旧 Agent MLP 和默认岭回归代码后复跑独立 test 集：`runs/completion-final/689debc980935c51/report.json`，16,907 人，`model_device=cuda`。[最终 JSON](coupon-uplift-completion-final.json) 已同步到仓库。MLP 相对岭回归的转化策略差值 `-0.000946/人`，95% 正态近似区间 `[-0.002519,+0.000626]`；内置 2,000 次用户级配对 bootstrap 区间 `[-0.002484,+0.000591]`。两者跨零，仍无 Agent 改善转化的证据；没有 App 活跃或真实净收益标签。
- 使用同一模型但增加候选 `choose_policy` 的 GPU 运行检查：`runs/completion-policy-smoke/aceff58fc80df1cf/report.json`。预测 SHA-256 与默认基线一致、策略 SHA-256 不同，确认代码允许只改预算策略。该检查故意选择零个用户，只验证接口与评估流程，不是有效营销方案。
- DeepSeek `deepseek-flash` 在目标机对这个零发券候选实际完成一次 Agent 轮次：旧运行 `runs/completion-agent-policy/ecf75484b098298a/report.json`，新运行 `runs/completion-agent-policy/5ce392ae2fd9dfc7/report.json`；最终框架版本用这两份冻结候选代码再跑出 `runs/completion-agent-policy-rerun/1fce3abf6492799a/report.json`，策略选中人数和配对差值相同。[策略报告](coupon-uplift-agent-policy-report.json)和[high/max 分析](coupon-uplift-agent-policy-analysis.json)已同步。模型预测指纹相同、策略指纹不同，Agent 修改 `choose_policy` 后在预测成本预算内选中 3,381 人；验证集相对零发券策略的配对 IPW 转化差值 `+0.003312/人`、95%区间 `[+0.001707,+0.004918]`。这只是从刻意设置的空策略恢复发券的闭环验证，不能宣称优于正常业务策略，也未使用最终测试集证明提升。分析记录 `high` 初评及因特征时点仅声明、假设成本难以权衡而触发的 `max` 复核。
