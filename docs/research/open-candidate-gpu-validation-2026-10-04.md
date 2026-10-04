# 开放候选算法的 GPU 验证（2026-10-04）

本次检查新的搜索约束是否允许非 T-Learner 候选进入相同的隔离评估流程。目标机为 RTX 5090，代码目录为 `/home/charles/couponevo-validation-92df1e3`，使用固定 Docker 沙箱镜像 `sha256:ec2754fb9f85d20060360fad9165fd272ea3d84c07e9301757c7bb42c62d6095`。本次提案由确定性验证脚本提交，**不是 DeepSeek 自主提出的实验**。

以仓库的 PyTorch 岭回归基线为起点，将 EconML `TLearner` 换成 `SLearner(overall_model=TorchRidgeRegressor(device=device))`，保持 `fit_predict` 接口、预算、目标和评估器不变。候选快照 SHA-256 为 `d1e78607c3fd747b5dff96845c46cde9d870b0fb92a25b89c29e17febb5f4ce4`。训练和预测中的 PyTorch 张量按 `device=cuda` 创建；运行报告记录 `model_device=cuda`、CUDA 峰值分配 `38,024,192` 字节。

使用完整的 [Starbucks 随机促销数据](../../examples/README.md)，原始 CSV SHA-256 为 `4d48190fd0d6a65d3874fa9a9ac79d89007579716366c2cfe140ae999088aa4f`。新运行的切分清单为训练 `50,720`、验证 `16,907`、测试 `16,907` 行；并集 `84,534` 行，切分间重叠 `0`。搜索日志位于 `runs/open-slearner-full-20261004/journal.json`。本次只运行验证集搜索，**没有执行 finalize 或使用最终测试集选模型**。

在固定 `cost=0.03` 预算下，S-Learner 的验证集购买转化策略值为 `0.001656`，低于同一运行的基线 `0.003312`。这证明非 T-Learner 路径能进入 GPU 沙箱并产生可比较的离线报告，不构成性能提升证据。简单线性 S-Learner 的异质性表达有限，本实验也不用于比较模型家族优劣。

本地和目标机完整回归均为 `113` 项，其中 `112` 项通过、`1` 项跳过。测试覆盖开放候选校验、Agent 提案字段、反思后的终局决策、结构化数据请求、请求后的恢复和最终评估限制。新的 Agent 提示词可以提出不同机制，但本次没有验证 DeepSeek 在真实 API 调用中会自主选择哪一种机制。
