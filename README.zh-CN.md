# PromoLift

[English](README.md) | 简体中文

PromoLift 面向优惠券等营销触达场景，是固定随机对照数据上的 Agent 主导算法实验项目。Agent 可改动[候选 uplift 模型](src/promolift/candidate.py)及候选预算策略；数据加载、预算约束和评测器独立于候选代码。

## 状态与边界

Agent 能在固定任务中提出假设、修改 EconML/PyTorch 候选代码、读取验证反馈、积累与数据集版本绑定的经验，并在冻结测试集上比较最终候选。当前未训练 Agent 自身，也未证明改进能力随经验递归增强，因此不宣称实现完整 RSI。公开数据仅供离线方法验证，不代表 App 促活或真实优惠券净收益已提升。

## 运行

需要 Python 3.12+、pandas、NumPy、PyTorch 和 EconML；GPU 实验需要支持 CUDA 的 PyTorch。依赖版本由 `uv.lock` 固定，可先运行 `uv sync --frozen --python 3.12`。按[公开数据说明](examples/README.md)将 Starbucks 促销实验 CSV 下载到 `data/starbucks-training.csv`，再在项目根目录执行：

```bash
uv run --frozen python -m promolift.cli run examples/starbucks.json --budget-kind cost --budget 0.03 --device cuda --seed 42
```

运行产物在 `runs/<run_id>/report.json`、`report.md`、`candidate.py` 和 `split_manifest.json`。切分文件以固定数据版本的行号记录三份样本；随机模型的 PyTorch 种子由执行器固定。默认只评估验证集；选定方案后可加 `--final` 使用测试集。

若本机已安装并登录 Codex CLI，可运行一次自动代码修改与复评。Agent 生成的候选代码必须在容器内执行，先在 Linux GPU 实验机准备与宿主机 Python 3.12 虚拟环境兼容的镜像：

```bash
docker build -f Dockerfile.sandbox -t promolift-sandbox:py312-cuda128 .
export SANDBOX_IMAGE="$(docker image inspect promolift-sandbox:py312-cuda128 --format '{{.Id}}')"
```

容器只挂载本轮训练样本、留出集特征、候选代码、只读运行时与框架源码；禁用网络、只读根文件系统并限制资源。宿主机虚拟环境默认取当前 Python 的 `sys.prefix`，也可用 `COUPON_LAB_SANDBOX_VENV` 指向 Linux Python 3.12 环境。镜像 ID 会写入实验身份和报告；修改镜像或依赖后应新开任务。

```bash
python -m promolift.cli agent examples/starbucks.json --budget-kind cost --budget 0.03 \
  --device cuda --seed 42 --sandbox-image "$SANDBOX_IMAGE"
```

Agent 使用本机 Codex CLI 的默认模型；若该模型在当前 CLI 登录账户中不可用，可加 `--agent-model MODEL` 指定可用模型。

也可以让 DeepSeek V4.1 Flash 充当修改候选算法的 Agent。先在当前终端设置 `DEEPSEEK_API_KEY`，再执行：

```bash
mkdir -p runs
cp src/promolift/candidate.py runs/my-candidate.py
python -m promolift.cli agent examples/starbucks.json --budget-kind cost --budget 0.03 --device cuda --seed 42 \
  --agent-provider deepseek --agent-model deepseek-flash --candidate runs/my-candidate.py \
  --sandbox-image "$SANDBOX_IMAGE"
```

Agent 会修改传入的候选文件。DeepSeek 的代码迭代和新报告解读使用 `high` 推理档位；若解读发现疑似特征泄漏，或评测器发现 uplift 低于随机策略、随机基线跨轮漂移、净收益区间跨零、促活与收益策略互有取舍等情况，会追加一次 `max` 复核。两轮结论与触发原因分别保存在 `runs/<run_id>/analysis.json` 和 `analysis.md`；评测指标仍以 `report.json` 为准，Agent 对代码或因果解释的判断需人工核查。报告记录 Agent 提供方、模型、前一轮运行 ID、预测和策略 SHA-256。候选若没有改变验证集预测和发券决策，本轮会报错并恢复候选文件。密钥从环境变量读取，不写入报告或候选代码。

可用[配置示例](examples/agent.deepseek.json)指定 OpenAI 兼容 Chat Completions 的 provider URL、模型、密钥环境变量及推理档位。设置 `AGENT_API_KEY` 后执行：

```bash
python -m promolift.cli agent examples/starbucks.json --budget-kind cost --budget 0.03 --device cuda \
  --candidate runs/my-candidate.py --agent-config examples/agent.deepseek.json \
  --sandbox-image "$SANDBOX_IMAGE"
```

也可直接传 `--agent-provider-url URL --agent-model MODEL`，并在 `AGENT_API_KEY` 中提供密钥；完整 `/chat/completions` URL 和 API base URL 均可。使用 `api.deepseek.com` 时默认显式发送 `thinking: enabled`；配置中的 `thinking: "enabled"` 可用于代理 DeepSeek 的其他 URL。其他兼容端点可设为 `"omit"`，仍会发送 `reasoning_effort`。端点需支持 Chat Completions、JSON 输出及所选推理档位；若请求被拒绝，程序会报错，不会主动降级到关闭思考模式。API key 不放入配置文件或命令行。

这个命令先生成报告，再让 Agent 读取候选代码和报告、修改 `candidate.py`，检查语法后重新实验。候选算法使用 EconML `TLearner`，底层结果模型以 PyTorch 在指定设备训练；Agent 可以调模型参数、使用干预前特征，或加入 `choose_policy(scores, costs, budget_kind, budget_value)` 返回逐用户布尔发券决策。评估器会拒绝超过人数或预测成本预算的策略。Codex 使用临时工作目录；DeepSeek API 只接收候选代码和报告。若发现缺少关键的干预前用户特征，Agent 可另写 `feature_gaps.md`，记录建议字段、来源、时点、证据、泄漏风险和下一版验证方法；该文件会进入新一轮报告，当前固定 dataset 不变。自动修改可能产生性能更差的候选；各轮报告和代码快照可供比较。

## Agent 主导的多轮搜索

`search` 让 API Agent 根据历史候选、评估指标和失败原因，自行选择 `draft`、`improve`、`debug` 或 `crossover`，提出假设并生成下一版完整候选代码。实验目标、数据、预算、评估器和最多迭代次数由命令固定。沿用上面的 `DEEPSEEK_API_KEY` 环境变量，在有 CUDA 的目标机执行：

```bash
python -m promolift.cli search examples/starbucks.json --budget-kind cost --budget 0.03 \
  --objective conversion --max-steps 3 --search-id starbucks-search-01 \
  --device cuda --agent-provider deepseek --agent-model deepseek-flash \
  --sandbox-image "$SANDBOX_IMAGE"
```

也可用 `--agent-config` 或 `--agent-provider-url`、`--agent-model` 和密钥环境变量配置其他兼容服务。`runs/<search-id>/journal.json` 保存基线、每轮假设、父候选、状态、验证集分数和分析；`steps/` 保存候选快照，`runs/` 保存评估报告。失败候选留在日志中供 Agent 修复；中断后用相同参数加 `--resume` 继续，`--max-steps` 可以增大。Agent 代码迭代和报告分析使用 `high`，已有反常 uplift、疑似泄漏或成本权衡不清时仍按上述规则进行 `max` 复核。

搜索分数只是反复使用验证反馈后的**探索性排序**。确定搜索结束后，使用相同任务参数冻结验证集冠军，并只在独立测试集比较一次：

```bash
python -m promolift.cli finalize examples/starbucks.json --budget-kind cost --budget 0.03 \
  --objective conversion --search-id starbucks-search-01 --device cuda --bootstrap-reps 2000 \
  --sandbox-image "$SANDBOX_IMAGE"
```

最终报告位于 `runs/<search-id>/final/<run_id>/`，含与初始候选的逐用户配对差值和 bootstrap 区间；完成最终测试后，该搜索不能继续迭代。`search` 和 `agent` 默认要求 `--sandbox-image`；只有显式传 `--unsafe-local-execution` 才允许直接运行生成代码。容器隔离需要可信的 Docker 守护进程、镜像及宿主机；接入业务数据前仍需对镜像、虚拟环境和宿主机权限做部署审查。公开 Starbucks 只有购买转化及假设发送成本，不能据此判定 App 促活或真实净收益提升。

## 固定任务与数据集绑定经验基准

`examples/tasks/` 将数据 SHA-256、初始候选、目标、预算、随机种子和 Agent 最大提案数固定下来。下载 [Starbucks](examples/README.md) 与 [Hillstrom](examples/README.md) 后，可用同一 Agent 配置与镜像运行两阶段基准：

```bash
python -m promolift.benchmark examples/tasks/starbucks-conversion.json \
  examples/tasks/hillstrom-revenue.json --agent-config examples/agent.deepseek.json \
  --sandbox-image "$SANDBOX_IMAGE" --device cuda --output runs/benchmark-01
```

第一阶段对每个任务做无历史经验搜索；第二阶段仅复用**同一数据集版本、同一 manifest、目标、预算、切分种子和评估器版本**的已完成搜索中的验证集假设与分数变化。CSV 内容或 manifest 变化后，旧经验不会被读取；同一搜索内部的步骤历史也始终供 Agent 参考。两个阶段使用相同初始候选、预算、种子和提案数。`summary.json` 记录验证集分数、失败次数、用时和独立测试集上“复用经验减无经验”的逐用户配对 bootstrap 区间。测试集结果不会进入经验。单次对照不足以证明经验本身造成提升，需多次重复并人工复核。这里没有训练 Agent 自身的权重；Agent 仍会编写和调优 uplift 候选模型。

## Dataset manifest

参照 [starbucks.json](examples/starbucks.json) 映射固定 CSV。`treatment` 选定一个对照臂和一个干预臂；其他臂被过滤。`outcomes` 至少有一个非成本结果，可包含 `active`、`visit`、`click`、`conversion`、`revenue`、`gross_margin`、`coupon_cost`。特征须在随机分组前可用，且不得包含分组、结果或成本字段。有用户 ID 时要求一人一行；若没有用户 ID，会按行切分并在报告中提示无法检查同一用户跨切分。

未来的 App 促活随机试验可按[业务数据接入约定](docs/business-rct-contract.md)提供固定数据和现行发券规则；目前没有真实业务数据，因此不会声称公开邮件或购买数据验证了实际促活与券成本收益。

`probability` 优先填实验方案的处理概率，同时填写 `probability_source`（`protocol`、`assignment_log` 或 `empirical`）及可审阅的 `probability_reference`；来源缺失时拒绝评估。只有确认简单随机分组且公开样本未按组差异抽样时，才可填 `"empirical"` 并声明这两个前提。若有逐行 `assignment_time_column` 和每个特征对应的 `feature_time_columns`，加载器会拒绝记录时间晚于或等于分组时间的特征；`--strict-data` 要求这组时间证据齐全。Starbucks 只有字段级的干预前声明，报告会标为 `declared_only`，不能据此证明没有特征泄漏。随机化来源引用也需要人工核查试验执行情况。

有 `gross_margin` 和实际或假设成本时，必须声明 `margin_includes_coupon_cost`，避免重复扣减。若没有实际成本，可选填 `fixed_send_cost` 作明确标注的假设成本；实际成本与假设成本不能同时填。否则只能用 `--budget-kind count`。`count` 的预算值是发券人数占该次评估人群的比例，`cost` 的预算值是每个候选用户允许的预测平均成本。

报告只使用 dataset 支持的结果名称。收入扣券成本记作增量净收入，只有毛利口径明确时才评估增量净收益。仅有 `fixed_send_cost` 时会标为假设净收益。成本预算根据预测成本选人，报告同时给出随机实验上的观测成本区间；实际成本可能超出预测预算。

## 验证与限制

```bash
uv run --frozen python -m unittest discover -s tests -v
```

离线评估使用冻结随机试验的处理概率和独立验证集，输出逆概率加权的策略增量及 95% 正态近似区间，并给出相对随机策略的配对差值区间。Qini/AUUC 只用于诊断 uplift 排序。Agent 每轮还会在同一验证集上计算新旧策略逐用户差值的配对区间。选定候选后，用冻结的两份代码在未参与调参的测试集上做最终配对比较；可加 `--bootstrap-reps 2000` 输出用户级配对 bootstrap 百分位区间：

```bash
python -m promolift.cli run examples/starbucks.json --budget-kind cost --budget 0.03 --device cuda --seed 42 \
  --candidate runs/new-candidate.py --compare-candidate runs/baseline-candidate.py --final --bootstrap-reps 2000
```

验证集配对区间不能作为多轮筛选后的最终提升证据；测试集也应只在候选与指标确定后使用一次。若现行业务规则可复现，可把它实现为另一份候选文件并用 `--compare-candidate` 比较；当前尚未取得该规则。当前只有一种干预相对不干预、一次决策、CSV 输入、数值和类别特征的 EconML T-learner，默认 PyTorch 岭回归结果模型。CPU 和 CUDA 路径都用 PyTorch 训练和预测；`--device cuda` 要求候选算法确认设备，容器运行还要求观测到 CUDA 张量分配。特征读取与预处理仍由 CPU 完成。该原型不执行线上发券，不自动采集数据，也不将公开数据缺少的收入或券成本补造成真实标签。公开数据的授权和字段差异见[调研记录](docs/research/open-uplift-datasets.md)；三份公开数据在目标机的测试结果见[GPU 验证记录](docs/research/gpu-validation-2026-10-03.md)，本次 Agent 多轮搜索见[实验记录](docs/research/agent-search-gpu-2026-10-03.md)。先前的跨数据集基准见[历史记录](docs/research/agent-benchmark-gpu-2026-10-03.md)，不代表当前经验机制。

## 参与与许可

参见 [架构说明](docs/architecture.md)与 [CONTRIBUTING.md](CONTRIBUTING.md)。项目代码采用 [Apache-2.0](LICENSE)；公开数据集另有各自的许可，原始数据不会打包或提交。
