# Coupon Uplift Lab

固定随机对照数据上的营销算法实验原型。Agent 可改动[候选 uplift 模型](src/coupon_lab/candidate.py)及候选预算策略；数据加载、预算约束和评测器独立于候选代码。

## 运行

需要 Python 3.12+、pandas、NumPy、PyTorch 和 EconML；GPU 实验需要支持 CUDA 的 PyTorch。依赖版本由 `uv.lock` 固定，可先运行 `uv sync --frozen --python 3.12`。按[公开数据说明](examples/README.md)将 Starbucks 促销实验 CSV 下载到 `data/starbucks-training.csv`，再在项目根目录执行：

```bash
python -m pip install -e .
python -m coupon_lab.cli run examples/starbucks.json --budget-kind cost --budget 0.03 --device cuda --seed 42
```

运行产物在 `runs/<run_id>/report.json`、`report.md`、`candidate.py` 和 `split_manifest.json`。切分文件以固定数据版本的行号记录三份样本；随机模型的 PyTorch 种子由执行器固定。默认只评估验证集；选定方案后可加 `--final` 使用测试集。

若本机已安装并登录 Codex CLI，可运行一次自动代码修改与复评：

```bash
python -m coupon_lab.cli agent examples/starbucks.json --budget-kind cost --budget 0.03 --device cuda --seed 42
```

Agent 使用本机 Codex CLI 的默认模型；若该模型在当前 CLI 登录账户中不可用，可加 `--agent-model MODEL` 指定可用模型。

也可以让 DeepSeek V4.1 Flash 充当修改候选算法的 Agent。先在当前终端设置 `DEEPSEEK_API_KEY`，再执行：

```bash
mkdir -p runs
cp src/coupon_lab/candidate.py runs/my-candidate.py
python -m coupon_lab.cli agent examples/starbucks.json --budget-kind cost --budget 0.03 --device cuda --seed 42 \
  --agent-provider deepseek --agent-model deepseek-flash --candidate runs/my-candidate.py
```

Agent 会修改传入的候选文件。DeepSeek 的代码迭代和新报告解读使用 `high` 推理档位；若解读发现疑似特征泄漏，或评测器发现 uplift 低于随机策略、随机基线跨轮漂移、净收益区间跨零、促活与收益策略互有取舍等情况，会追加一次 `max` 复核。两轮结论与触发原因分别保存在 `runs/<run_id>/analysis.json` 和 `analysis.md`；评测指标仍以 `report.json` 为准，Agent 对代码或因果解释的判断需人工核查。报告记录 Agent 提供方、模型、前一轮运行 ID、预测和策略 SHA-256。候选若没有改变验证集预测和发券决策，本轮会报错并恢复候选文件。密钥从环境变量读取，不写入报告或候选代码。

可用[配置示例](examples/agent.deepseek.json)指定 OpenAI 兼容 Chat Completions 的 provider URL、模型、密钥环境变量及推理档位。设置 `AGENT_API_KEY` 后执行：

```bash
python -m coupon_lab.cli agent examples/starbucks.json --budget-kind cost --budget 0.03 --device cuda \
  --candidate runs/my-candidate.py --agent-config examples/agent.deepseek.json
```

也可直接传 `--agent-provider-url URL --agent-model MODEL`，并在 `AGENT_API_KEY` 中提供密钥；完整 `/chat/completions` URL 和 API base URL 均可。使用 `api.deepseek.com` 时默认显式发送 `thinking: enabled`；配置中的 `thinking: "enabled"` 可用于代理 DeepSeek 的其他 URL。其他兼容端点可设为 `"omit"`，仍会发送 `reasoning_effort`。端点需支持 Chat Completions、JSON 输出及所选推理档位；若请求被拒绝，程序会报错，不会主动降级到关闭思考模式。API key 不放入配置文件或命令行。

这个命令先生成报告，再让 Agent 读取候选代码和报告、修改 `candidate.py`，检查语法后重新实验。候选算法使用 EconML `TLearner`，底层结果模型以 PyTorch 在指定设备训练；Agent 可以调模型参数、使用干预前特征，或加入 `choose_policy(scores, costs, budget_kind, budget_value)` 返回逐用户布尔发券决策。评估器会拒绝超过人数或预测成本预算的策略。Codex 使用临时工作目录；DeepSeek API 只接收候选代码和报告。若发现缺少关键的干预前用户特征，Agent 可另写 `feature_gaps.md`，记录建议字段、来源、时点、证据、泄漏风险和下一版验证方法；该文件会进入新一轮报告，当前固定 dataset 不变。自动修改可能产生性能更差的候选；各轮报告和代码快照可供比较。

## Dataset manifest

参照 [starbucks.json](examples/starbucks.json) 映射固定 CSV。`treatment` 选定一个对照臂和一个干预臂；其他臂被过滤。`outcomes` 至少有一个非成本结果，可包含 `active`、`visit`、`click`、`conversion`、`revenue`、`gross_margin`、`coupon_cost`。特征须在随机分组前可用，且不得包含分组、结果或成本字段。有用户 ID 时要求一人一行；若没有用户 ID，会按行切分并在报告中提示无法检查同一用户跨切分。

`probability` 优先填实验方案的处理概率，同时填写 `probability_source`（`protocol`、`assignment_log` 或 `empirical`）及可审阅的 `probability_reference`；来源缺失时拒绝评估。只有确认简单随机分组且公开样本未按组差异抽样时，才可填 `"empirical"` 并声明这两个前提。若有逐行 `assignment_time_column` 和每个特征对应的 `feature_time_columns`，加载器会拒绝记录时间晚于或等于分组时间的特征；`--strict-data` 要求这组时间证据齐全。Starbucks 只有字段级的干预前声明，报告会标为 `declared_only`，不能据此证明没有特征泄漏。随机化来源引用也需要人工核查试验执行情况。

有 `gross_margin` 和实际或假设成本时，必须声明 `margin_includes_coupon_cost`，避免重复扣减。若没有实际成本，可选填 `fixed_send_cost` 作明确标注的假设成本；实际成本与假设成本不能同时填。否则只能用 `--budget-kind count`。`count` 的预算值是发券人数占该次评估人群的比例，`cost` 的预算值是每个候选用户允许的预测平均成本。

报告只使用 dataset 支持的结果名称。收入扣券成本记作增量净收入，只有毛利口径明确时才评估增量净收益。仅有 `fixed_send_cost` 时会标为假设净收益。成本预算根据预测成本选人，报告同时给出随机实验上的观测成本区间；实际成本可能超出预测预算。

## 验证与限制

```bash
uv run --frozen python -m unittest discover -s tests -v
```

离线评估使用冻结随机试验的处理概率和独立验证集，输出逆概率加权的策略增量及 95% 正态近似区间，并给出相对随机策略的配对差值区间。Qini/AUUC 只用于诊断 uplift 排序。Agent 每轮还会在同一验证集上计算新旧策略逐用户差值的配对区间。选定候选后，用冻结的两份代码在未参与调参的测试集上做最终配对比较；可加 `--bootstrap-reps 2000` 输出用户级配对 bootstrap 百分位区间：

```bash
python -m coupon_lab.cli run examples/starbucks.json --budget-kind cost --budget 0.03 --device cuda --seed 42 \
  --candidate runs/new-candidate.py --compare-candidate runs/baseline-candidate.py --final --bootstrap-reps 2000
```

验证集配对区间不能作为多轮筛选后的最终提升证据；测试集也应只在候选与指标确定后使用一次。若现行业务规则可复现，可把它实现为另一份候选文件并用 `--compare-candidate` 比较；当前尚未取得该规则。当前只有一种干预相对不干预、一次决策、CSV 输入、数值和类别特征的 EconML T-learner，默认 PyTorch 岭回归结果模型。CPU 和 CUDA 路径都用 PyTorch 训练和预测；`--device cuda` 要求候选算法在 GPU 上执行并确认设备。特征读取与预处理仍由 CPU 完成。该原型不执行线上发券，不自动采集数据，也不将公开数据缺少的收入或券成本补造成真实标签。公开数据的授权和字段差异见[调研记录](docs/research/open-uplift-datasets.md)；三份公开数据在目标机的测试结果见[GPU 验证记录](docs/research/gpu-validation-2026-10-03.md)。
