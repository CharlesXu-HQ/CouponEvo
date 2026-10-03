# 公开数据

`starbucks.json` 映射 [Udacity 发布的 Starbucks 促销实验 training.csv](https://github.com/udacity/DSND_Term2/blob/master/portfolio_exercises/Experiments/data/training.csv)。将原始 CSV 保存为项目根目录 `data/starbucks-training.csv`。它有 84,534 行、随机促销与不促销两组、七个干预前匿名特征和购买结果；`fixed_send_cost: 0.15` 是课程题设的每次促销费用，不是用户级实际核销成本。没有 App 活跃、用户级收入或毛利，因而不能从该数据得出增量净利润。

`data/` 已加入 `.gitignore`，公开数据不会随代码提交。Criteo 广告增量数据只用于 GPU 模型训练与预测检查：其[公开数据卡](https://huggingface.co/datasets/criteo/criteo-uplift)说明发布版经非均匀抽样，原始实验的处理概率不可从公开文件可靠恢复，因此本项目不对它输出逆概率加权的因果收益报告。
