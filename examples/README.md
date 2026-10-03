# 公开数据

`starbucks.json` 映射 [Udacity 发布的 Starbucks 促销实验 training.csv](https://github.com/udacity/DSND_Term2/blob/master/portfolio_exercises/Experiments/data/training.csv)。将原始 CSV 保存为项目根目录 `data/starbucks-training.csv`。它有 84,534 行、随机促销与不促销两组、七个干预前匿名特征和购买结果；`fixed_send_cost: 0.15` 是课程题设的每次促销费用，不是用户级实际核销成本。没有 App 活跃、用户级收入或毛利，因而不能从该数据得出增量净利润。

`data/` 已加入 `.gitignore`，公开数据不会随代码提交。Criteo 广告增量数据只用于 GPU 模型训练与预测检查：其[公开数据卡](https://huggingface.co/datasets/criteo/criteo-uplift)说明发布版经非均匀抽样，原始实验的处理概率不可从公开文件可靠恢复，因此本项目不对它输出逆概率加权的因果收益报告。

[Hillstrom/MineThatData 邮件随机试验](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)包含 64,000 名客户；男装邮件、女装邮件和不发邮件各按约三分之一随机分配。原始 CSV 可从 `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv` 下载为 `data/hillstrom.csv`，预期 SHA-256 为 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`。`hillstrom.json` 只比较男装邮件与不发邮件，因此该二臂样本内的处理概率为 0.5。字段包含历史购买画像、两周内访问、转化和消费额，没有真实邮件/优惠券成本或毛利，预算只能按人数比例设置；收入指标不能称为净收益，也不能代表 App 促活。数据没有用户 ID 和逐行特征记录时间，报告会标注这两个限制。
