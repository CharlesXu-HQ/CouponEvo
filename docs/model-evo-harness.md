# ModelEvoHarness 接入与 GPU A/B 验证

ModelEvoHarness 是独立的跨场景模型迭代项目。CouponEvo 通过 `--harness model-evo` 读取其研究 catalog、判断方向与当前任务的适用性，并校验 Agent 的实验设计。训练、候选代码隔离执行、固定目标、预算、验证集和最终留出集仍由 CouponEvo 控制。仓库内的 [JSON 研究插件](harness.md) 是另一种可选入口。

## 安装与任务边界

CouponEvo 用 `third_party/model-evo-harness` Git submodule 引入独立仓库。克隆后先初始化 submodule，并从本地检出目录安装到运行搜索的主机 Python 环境；仅克隆 CouponEvo 不会安装这个包。每次新建 `search --harness model-evo` 会在导入包前自动执行 `git submodule update --init --remote`，获取配置的最新 `main`。候选代码的 Docker 镜像不需要安装它。当前 `uv.lock` 不含这项可选依赖，因此先同步 CouponEvo 环境，再安装 submodule，并用 `.venv/bin/python` 运行；若以后重新同步环境，应从**同一检出目录**重新安装。

```bash
git submodule update --init third_party/model-evo-harness
git -C third_party/model-evo-harness rev-parse HEAD
git -C third_party/model-evo-harness status --short  # 应为空
uv sync --frozen --python 3.12
uv pip install --python .venv/bin/python -e ./third_party/model-evo-harness
.venv/bin/python -c 'import model_evo_harness; print(model_evo_harness.__file__)'
```

CouponEvo 的 Gitlink 记录 submodule 的默认提交；新搜索会把本地检出推进到最新 `main`。如需运行前查看远端版本，也可手动执行 `git submodule update --init --remote third_party/model-evo-harness`，但新搜索不依赖这一步。检查 submodule 工作区无未提交修改，确保记录的提交与运行代码一致。同一次搜索的 `search`、`--resume` 和 `finalize` 都必须带 `--harness model-evo`。`--resume` 和 `finalize` 不更新 submodule；其提交须与搜索记录相同。CouponEvo 将源码提交、包实现摘要和 catalog 摘要写入任务身份；版本变化会拒绝续跑，也不会复用旧版本经验。省略该参数就是原有搜索；`--harness harnesses/coupon-research.json` 仍使用仓库内 JSON 指南。

接入只向 Agent 提供训练分区的字段类型、基数和缺失比例，以及 manifest 已证明的数据契约。Starbucks 任务阶段是 `policy`，模型框架是 `pytorch`，原始输入只有 `V1` 至 `V7`；处理分配概率有来源说明。匿名静态字段不能被解释为行为序列、物品目录或场景上下文。已知研究家族需满足 catalog 的阶段和能力条件；Agent 可提出 catalog 以外的新方向，但必须用当前真实输入完成可证伪实验。规范化的研究设计和外部包身份保存在 CouponEvo journal 中。Agent 上下文包含适用模型的本地 `model_api` 构造和调用签名。ModelEvo 模式允许候选代码导入 catalog 声明的 PyTorch 参考模型模块及 `models.pytorch.training`。评估器仅将 submodule 中的 PyTorch 参考源码复制到临时目录，再以只读方式挂载到 Docker，并设置 `PYTHONPATH`；候选容器看不到 Harness 的 provider、engine、TensorFlow 代码或宿主机 API 密钥。其他 Harness 模块的导入会在候选校验时被拒绝。

ModelEvo 模式把可用字段上的网络结构、损失、优化和采样实验放在缺失特征请求之前。实验依据型请求必须引用至少两个已评估且机制不同的试验 ID。若业务专家已能直接确认必要输入缺失，可在 manifest 中显式声明领域要求，允许 Agent 立即请求相应字段：

```json
{
  "domain_requirements": [
    {
      "id": "coupon-history",
      "source": "coupon operations owner",
      "fields": ["prior_coupon_use"],
      "as_of": "before treatment assignment"
    }
  ]
}
```

`feature_request` 保留字段名称、定义、来源、时点、证据和新数据集验证计划，并须设置 `basis=experimental_evidence` 与 `trial_ids`，或 `basis=domain_requirement` 与 `requirement_id`。实验提议若附带 `feature_gaps_md`，也必须附带符合相同规则的结构化 `feature_request`；自由文本不能绕过证据要求。领域要求只从 manifest 进入任务快照；Agent 不能自行补写。每轮反思分别保存技术经验与业务经验；后者只能引用验证集报告中的策略级指标 ID，缺乏可观察证据时记录 `not_observable`。相同数据集和实验任务可复用这两类经验，最终留出集结果不进入 Agent 上下文。

## GPU A/B 复现实验

2026-10-05 的接入冒烟验证使用 CouponEvo `b2359aa`、ModelEvoHarness `544badf` 的源码归档，在 RTX 5090 上运行。输入为完整的 Starbucks CSV（84,534 行，不抽样）；验证集 16,907 行。候选代码在 Docker 沙箱中导入 Harness 的 PyTorch `FM` 和 `focal_loss`，执行 CUDA 前向和反向传播，再由原有 T-learner 产生策略预测。报告记录 `model_device=cuda` 和 `cuda_peak_bytes=35397120`。这只验证源码挂载、导入和 GPU 执行；FM 探针没有参与策略预测，不能据此推断算法指标提升或 Agent 已完成新一轮搜索。

种子 42 的全量 Starbucks GPU 对照已执行。两组 Agent 都提出新数据需求，因此按现有规则没有运行正式 `finalize`；冻结验证集候选后的单次诊断性配对测试，其 95% 区间包含零。实验结果、源提交及产物见[报告](research/model-evo-harness-starbucks-ab-2026-10-04.md)。下面保留实验方案和复现命令。

目标是比较**相同 CouponEvo 版本下**的基础 Agent 搜索与启用 ModelEvoHarness 的搜索。两组使用相同的 Starbucks CSV、manifest、初始候选、DeepSeek 配置、目标 `conversion`、每位候选用户 `0.03` 的假设成本预算、每轮最多 3 次实验、CUDA Docker 镜像。至少按种子 `42` 做一对完整数据搜索；资源允许时，可在看最终留出集结果**之前**预定增加 `43`、`44`。每对只改变 `--harness model-evo`。不传 `--experience-dir`，避免两组互相读到历史经验。Agent 可能提前停止或提交数据请求，这也是实验结果，不能补跑到预期步数后只保留有利样本。

先在 Linux CUDA 主机按 [数据说明](../examples/README.md)准备完整 `data/starbucks-training.csv`，配置 Docker GPU 和匹配的 PyTorch CUDA 版本，把 API 密钥放在 `AGENT_API_KEY` 环境变量中。下面的手动更新仅用于预检和记录计划使用的提交；每次新建 ModelEvoHarness 搜索仍会自动检查最新 `main`。脚本会核对实际提交，防止一组 A/B 搜索混用不同版本。续跑与最终评估不会自动更新。

```bash
set -euo pipefail
: "${AGENT_API_KEY:?set AGENT_API_KEY before the benchmark}"
BENCH_OUT=runs/model-evo-ab-001
BENCH_SEEDS=(42)  # 资源允许且尚未查看留出集时，可预先改为 (42 43 44)
mkdir -p "$BENCH_OUT"

git submodule update --init --remote third_party/model-evo-harness
test -z "$(git -C third_party/model-evo-harness status --porcelain)"
MODEL_EVO_COMMIT="$(git -C third_party/model-evo-harness rev-parse HEAD)"
uv sync --frozen --python 3.12
uv pip install --python .venv/bin/python -e ./third_party/model-evo-harness
docker build -f Dockerfile.sandbox -t couponevo-sandbox:py312-cuda128 .
SANDBOX_IMAGE="$(docker image inspect couponevo-sandbox:py312-cuda128 --format '{{.Id}}')"

git rev-parse HEAD > "$BENCH_OUT/couponevo-commit.txt"
printf '%s\n' "$MODEL_EVO_COMMIT" > "$BENCH_OUT/model-evo-commit.txt"
printf '%s\n' "$SANDBOX_IMAGE" > "$BENCH_OUT/sandbox-image.txt"
sha256sum examples/starbucks.json data/starbucks-training.csv \
  src/couponevo/candidate.py uv.lock > "$BENCH_OUT/input-sha256.txt"
uv pip freeze --python .venv/bin/python > "$BENCH_OUT/environment.txt"
printf '%s\n' "${BENCH_SEEDS[@]}" > "$BENCH_OUT/planned-seeds.txt"

for seed in "${BENCH_SEEDS[@]}"; do
  for arm in plain model-evo; do
    harness_args=()
    if [[ "$arm" == model-evo ]]; then harness_args=(--harness model-evo); fi
    .venv/bin/python -m couponevo.cli search examples/starbucks.json \
      --budget-kind cost --budget 0.03 --objective conversion --seed "$seed" \
      --max-steps 3 --search-id "${arm}-s${seed}" --output "$BENCH_OUT" \
      --device cuda --agent-config examples/agent.deepseek.json \
      --sandbox-image "$SANDBOX_IMAGE" "${harness_args[@]}"
    if [[ "$arm" == model-evo ]]; then
      test "$(git -C third_party/model-evo-harness rev-parse HEAD)" = "$MODEL_EVO_COMMIT" || {
        echo 'ModelEvoHarness main changed during benchmark; stop and restart the A/B pair' >&2
        exit 1
      }
    fi
  done
done
```

如发生可恢复的中断，先确认 `git -C third_party/model-evo-harness rev-parse HEAD` 仍等于 `model-evo-commit.txt` 中记录的提交；如需重建 Python 环境，只从该检出目录重装，不运行 `git submodule update --remote`。随后使用原搜索命令、相同参数加 `--resume`；不要更改数据、镜像、目标或 `--harness`。检查每对 `journal.json` 的候选状态、研究方向、失败和数据请求。只有搜索完整且可以冻结冠军时，才对两组执行以下预先确定的最终评估：

```bash
for seed in "${BENCH_SEEDS[@]}"; do
  for arm in plain model-evo; do
    harness_args=()
    if [[ "$arm" == model-evo ]]; then harness_args=(--harness model-evo); fi
    .venv/bin/python -m couponevo.cli finalize examples/starbucks.json \
      --budget-kind cost --budget 0.03 --objective conversion --seed "$seed" \
      --search-id "${arm}-s${seed}" --output "$BENCH_OUT" \
      --device cuda --bootstrap-reps 2000 --sandbox-image "$SANDBOX_IMAGE" \
      "${harness_args[@]}"
  done
done
```

`finalize` 分别报告两组冠军相对共同初始候选的留出集结果。若任一 Agent 提出新数据需求，现有 `finalize` 会拒绝运行；仍可在**不改动搜索日志和不再选择模型**的条件下，将两个验证集 `best_id` 冻结，做一次诊断性留出集配对比较。下面的脚本处理这两种情况，以 ModelEvoHarness 组为新候选、基础组为比较候选；它把是否正式完成 `finalize` 写入摘要。每个种子只运行一次直接配对评估，不能把诊断性结果称为正式搜索结论。

```bash
.venv/bin/python - "$BENCH_OUT" "$SANDBOX_IMAGE" "${BENCH_SEEDS[@]}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root, image = Path(sys.argv[1]), sys.argv[2]
seeds = [int(value) for value in sys.argv[3:]]
summary = []

def frozen_candidate(search_id):
    search_root = root / search_id
    journal = json.loads((search_root / "journal.json").read_text())
    if any(step["status"] == "pending" for step in journal["steps"]):
        raise ValueError(f"search has pending work: {search_id}")
    best = journal["best_id"]
    entry = (journal["baseline"] if best == "seed" else
             next(step for step in journal["steps"] if step["id"] == best))
    if entry["status"] != "evaluated":
        raise ValueError(f"best candidate was not evaluated: {search_id}")
    return search_root / entry["candidate"], journal

for seed in seeds:
    plain, plain_journal = frozen_candidate(f"plain-s{seed}")
    guided, guided_journal = frozen_candidate(f"model-evo-s{seed}")
    plain_task = {key: value for key, value in plain_journal["task"].items() if key != "harness"}
    guided_task = {key: value for key, value in guided_journal["task"].items() if key != "harness"}
    if plain_task != guided_task:
        raise ValueError(f"unmatched A/B task for seed {seed}")
    status = "finalized" if "final" in plain_journal and "final" in guided_journal else "diagnostic"
    result = subprocess.run([
        sys.executable, "-m", "couponevo.cli", "run", "examples/starbucks.json",
        "--budget-kind", "cost", "--budget", "0.03", "--seed", str(seed),
        "--output", str(root / "paired"), "--candidate", str(guided),
        "--compare-candidate", str(plain), "--device", "cuda", "--final",
        "--bootstrap-reps", "2000", "--sandbox-image", image,
    ], check=True, capture_output=True, text=True)
    report_path = Path(result.stdout.strip().splitlines()[-1]).with_name("report.json")
    report = json.loads(report_path.read_text())
    if report["holdout"] != "test" or report["model_device"] != "cuda":
        raise ValueError(f"unexpected evaluation mode for seed {seed}")
    if report["dataset_sha256"] != plain_journal["task"]["dataset"]:
        raise ValueError(f"dataset changed for seed {seed}")
    paired = report["paired_vs_baseline_bootstrap"]["conversion"]["conversion"]
    summary.append({"seed": seed, "status": status,
                    "data_requests": {"plain": "data_request" in plain_journal,
                                      "guided": "data_request" in guided_journal},
                    "guided_minus_plain": paired,
                    "plain_steps": len(plain_journal["steps"]),
                    "guided_steps": len(guided_journal["steps"]),
                    "report": str(report_path)})

(root / "paired-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
```

报告每个预定种子的验证集轨迹、候选是否真实使用 CUDA、直接配对差值及区间、提前停止和数据请求；完整展示全部预定种子，不据留出集结果挑选一个种子或继续改提案。DeepSeek 生成可能变化，单对或少量搜索都不能证明 harness 稳定提高业务收益。Starbucks 只有购买转化与假设发送成本，没有 App 促活标签、真实券成本或用户级毛利；结论限定为这份公开任务的离线购买转化策略。
