# Coupon Uplift Lab

固定随机对照数据上的营销算法实验原型。Agent 可改动 [候选 uplift 算法](src/coupon_lab/candidate.py)，固定的加载、切分、预算策略与评测器负责比较每一轮结果。

## 运行

需要 Python 3.12+、pandas、NumPy；在项目根目录执行：

```bash
python -m pip install -e .
python -m coupon_lab.cli run examples/demo.json --budget-kind cost --budget 0.25 --seed 42
```

运行产物在 `runs/<run_id>/report.json`、`report.md` 和 `candidate.py`。相同数据、manifest、候选代码、预算和种子产生相同 `run_id` 与指标。默认只评估验证集；选定方案后可加 `--final` 使用测试集。

若本机已安装并登录 Codex CLI，可运行一次自动代码修改与复评：

```bash
python -m coupon_lab.cli agent examples/demo.json --budget-kind cost --budget 0.25 --seed 42
```

Agent 使用本机 Codex CLI 的默认模型；若该模型在当前 CLI 登录账户中不可用，可加 `--agent-model MODEL` 指定可用模型。

这个命令先生成报告，再让 Codex 在临时工作目录中读取报告、修改一份 `candidate.py`，检查语法后复制回项目，并重新实验。若发现缺少关键的干预前用户特征，Agent 可另写 `feature_gaps.md`，记录建议字段、来源、时点、证据、泄漏风险和下一版验证方法；该文件会进入新一轮报告，当前固定 dataset 不变。数据和评测代码不进入 Agent 的临时工作目录。自动修改可能产生性能更差的候选；各轮报告和代码快照可供比较。

## Dataset manifest

参照 [demo.json](examples/demo.json) 映射固定 CSV。`treatment` 选定一个对照臂和一个干预臂；其他臂被过滤。`outcomes` 至少有一个非成本结果，可包含 `active`、`visit`、`click`、`conversion`、`revenue`、`gross_margin`、`coupon_cost`。特征须在随机分组前可用，且不得包含分组、结果或成本字段。有用户 ID 时要求一人一行；若没有用户 ID，会按行切分并在报告中提示无法检查同一用户跨切分。

`probability` 优先填实验方案的处理概率；只在确认简单随机分组且公开样本未按组差异抽样时，才可填 `"empirical"`。有 `gross_margin` 和实际或假设成本时，必须声明 `margin_includes_coupon_cost`，避免重复扣减。若没有实际成本，可选填 `fixed_send_cost` 作明确标注的假设成本；实际成本与假设成本不能同时填。否则只能用 `--budget-kind count`。`count` 的预算值是发券人数占该次评估人群的比例，`cost` 的预算值是每个候选用户允许的预测平均成本。

报告只使用 dataset 支持的结果名称。收入扣券成本记作增量净收入，只有毛利口径明确时才评估增量净收益。仅有 `fixed_send_cost` 时会标为假设净收益。成本预算根据预测成本选人，报告同时给出随机实验上的观测成本区间；实际成本可能超出预测预算。

## 验证与限制

```bash
python -m unittest discover -s tests -v
```

离线评估使用冻结随机试验的处理概率和独立验证集，输出逆概率加权的策略增量及 95% 正态近似区间。当前只有一种干预相对不干预、一次决策、CSV 输入、数值和类别特征的岭回归 T-learner。该原型不执行线上发券，不自动采集数据，也不将公开数据缺少的收入或券成本补造成真实标签。公开数据的授权和字段差异见[调研记录](docs/research/open-uplift-datasets.md)。
