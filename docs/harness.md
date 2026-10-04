# Research harness plugin / 实验研究插件

CouponEvo 的 `search` 可以通过一个 JSON 文件接入研究指南。插件描述实验方向、适用条件和常见误区；执行器提供数据概况、校验提案，并把研究设计贯穿到评估、反思和经验中。替换插件不需要改 Agent provider 或训练执行代码。插件不执行 Python，也不替换固定的评估器、预算或数据契约。

## 接入

在已有搜索命令中增加一个参数：

```bash
uv run --frozen python -m couponevo.cli search examples/starbucks.json \
  --budget-kind cost --budget 0.03 --objective conversion --seed 42 \
  --max-steps 3 --search-id starbucks-research --device cuda \
  --agent-config examples/agent.deepseek.json --sandbox-image "$SANDBOX_IMAGE" \
  --harness harnesses/coupon-research.json
```

API 使用 `run_search(..., harness_path=Path("harnesses/coupon-research.json"))`。`finalize` 和 `--resume` 也必须传入同一份内容的插件。不传 `--harness` 时保持基础搜索；旧的单次 `agent` 命令不支持此插件，会明确拒绝该参数。

插件内容的哈希进入任务和经验身份。修改指南后需要新搜索；改文件路径或 JSON 缩进不会改变内容哈希。数据或 manifest 改变时，仍按原有规则开启新任务。插件快照与数据概况保存在搜索目录的 `journal.json` 中，候选代码和指南不依赖运行时访问外部网站。

## 分类如何参与实验

提供的 [coupon-research.json](../harnesses/coupon-research.json) 参考 FunRec 的知识组织，并补充优惠券实验需要的因果与预算约束。下列名称是可扩展的研究视角，存在交叉关系，不是校验器的模型枚举。

| 研究视角 | Agent 需要回答的问题 | 当前任务的关键边界 |
| --- | --- | --- |
| 处理效应估计与响应结构 | 共享响应、分组建模、残差或正交目标哪个误差来源更值得检验？ | 预测必须表示处理效应；保持随机化概率与 PyTorch 设备要求 |
| 表示与预处理 | 编码、缺失、稀疏性或尺度是否限制泛化？ | 转换只在训练数据拟合，匿名字段不能虚构语义 |
| 显式与学习型交叉 | 当前模型漏掉哪种依赖，交叉组件是否真的带来收益？ | FM、DCN、注意力等只是机制示例；用消融区分结构和容量 |
| 行为历史与序列 | 顺序、近因性、会话或目标相关兴趣是否有额外信息？ | DIN/DIEN 等需要实际序列及相应上下文，静态表不能还原历史 |
| 多目标与任务依赖 | 共享信息是否改善固定主目标，是否存在负迁移？ | 需要实测辅助标签；不自动假设点击到购买漏斗 |
| 多场景与条件参数 | 哪些有证据的场景需要共享或独立响应？ | 需要场景字段和分组支持；多券动作超出当前二元契约 |
| 训练与稳定性 | 瓶颈是优化、稀有结果、方差还是容量？ | 重采样不能改变评估人群，内部早停或交叉拟合只用训练分区 |
| 校准、组合与不确定性 | 排序和幅度哪个因素影响了选人？ | 个体模型方差不是因果置信区间；保持配对评估独立 |
| 成本预算与策略选择 | 相同预测能否通过更合适的可行分配提高价值？ | 实测券成本与固定发送成本假设区分，目标保持固定 |
| 专家统计特征与数据补充 | 哪个缺失测量阻止了可检验假设？ | 写清实体、窗口、交叉维度、缺失规则和分配前截止时点 |
| 召回、协同及预训练/生成式表示 | 额外交互结构能否形成更好的响应表示？ | 需要实际交互、物品或序列与可用执行支持；不能生成虚拟 RCT 证据 |

Agent 每轮先读训练数据概况和此前实际实验，再比较候选机制。它不必把每一类都跑一遍，可以直接提出分类之外的方向。备选方案表示“考虑过”，不计为已实验；一个配置失败也不等于整个类别无效。

Starbucks 的匿名静态字段支持编码和交叉等假设，但其字段名不提供消费画像语义；没有历史序列或真实核销成本时，相应机制应提交数据需求。缺少证据的分组误差不能写成已经观察到的事实。

## 实验提案约定

启用插件后，实验 JSON 除原有字段外需要 `research`：

```json
{
  "direction": "自由文本，可使用新方向",
  "mechanism": "具体计算机制及相对已有方案的差异",
  "why_now": "基于当前报告、历史实验与备选机制的选择依据",
  "input_features": ["V1", "V2"],
  "data_rationale": "这些原始字段如何支持本次变换，以及仍未知的语义",
  "comparison": "在同一预算与目标下的对照或组件消融",
  "falsification": "什么结果会推翻当前假设",
  "alternatives": [
    {"direction": "另一个方向", "mechanism": "备选机制", "reason": "本轮暂缓它的依据"}
  ]
}
```

执行器校验结构、非空证据说明和原始输入字段。它无法仅靠这些文字验证因果解释或确认代码真的实施了某个机制；这些仍需要 Agent 代码分析、独立评估和人的审查。无协变量模型或只处理已有预测的策略改动可使用空的 `input_features`，并在 `data_rationale` 中说明。派生特征列出其原始输入，不要求预先出现在 CSV 中；缺少原始输入则不能继续该实验，Agent 可改提 `request_data`。数据请求、诊断和停止动作不要求伪造一份模型实验设计。

研究设计随候选保存在 journal，下一轮 context 的 `research_history` 只汇总实际提交的实验与结果。反思对照 mechanism、comparison、falsification；同任务经验保存该设计但不读取最终测试指标。

## 数据边界与自定义

概况只遍历 manifest 特征，记录训练行数、缺失比例和基数；列类型来自 CSV 解析后的 schema。同时提供现有结果标签、成本是实测/假设/缺失、特征时点是否已验证。不会传送原始用户记录、验证/测试分区统计或最终测试指标。列类型与基数不代表业务含义，也不证明某列是行为序列。

复制插件 JSON，修改 `name`、`guidance` 和 `directions` 即可使用自己的指南。`schema_version` 当前为 `1`；每个 direction 包含自由文本的 `name`、`question`、`requires`、`experiment`、`pitfalls`。`sources` 提供可选来源记录。指南作为参考数据传给 Agent，不能覆盖执行器或 system 的约束。

这次接入实现了实验组织方式，未把所有列举算法安装成模型库，也不保证 Agent 的选择或最终业务指标提升。生成的模型仍须符合现有候选接口，在用户指定设备上执行并接受独立评估。

## FunRec 参考来源

- [项目目录与章节](https://github.com/datawhalechina/fun-rec)：从召回、排序、重排到生成式表示的研究组织。
- [特征结构与处理](https://github.com/datawhalechina/fun-rec/tree/master/src/funrec/features)：字段及表示的组织方式。
- [模型目录](https://github.com/datawhalechina/fun-rec/tree/master/src/funrec/models)：交叉、序列、多目标、多场景等不同机制。
- [FM](https://github.com/datawhalechina/fun-rec/blob/master/src/funrec/models/fm.py)、[DIN](https://github.com/datawhalechina/fun-rec/blob/master/src/funrec/models/din.py)、[MMOE](https://github.com/datawhalechina/fun-rec/blob/master/src/funrec/models/mmoe.py)：分别核对交互、序列输入和多任务组织，避免把推荐预测结构直接当作 uplift 模型。

以上是思想与组织参考；本项目未复制 FunRec 的实现或章节文字。插件指南为针对 CouponEvo 数据与评估契约独立撰写的内容。
