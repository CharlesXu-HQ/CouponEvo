# 营销 Uplift 公开数据集调研

核对日期：2026-10-03。为「App 站内优惠券促活」项目寻找可复现的随机对照数据；增量活跃和增量净收益是否都能比较，取决于具体数据的字段。这里的「公开」只表示可获取；授权与可商用性单独判断。

## 候选数据

| 数据集 | 随机对照与规模 | 可用结果/成本 | 与目标的差距 | 建议用途 |
| --- | --- | --- | --- | --- |
| [MT-LIFT](https://github.com/MTDJDSP/MT-LIFT) | 美团 App 外卖优惠券随机试验；5,541,842 条，5 个 treatment，99 个匿名特征 | click、conversion | 发布的字段表没有活跃、收入/毛利、核销或实际券成本；仓库未标明数据许可证 | 最贴近业务的发券 uplift 算法基准；不能用其验证真实净利润 |
| [Hillstrom 邮件实验](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) | 64,000 客户，随机分到男装邮件、女装邮件或不发邮件 | visit、conversion、spend | 邮件而非 App 优惠券；spend 是消费额而非毛利；没有真实营销成本；原作者未给出明确数据许可证 | 小规模端到端流程、活跃与消费双结果演示 |
| [Lenta](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html) | 零售营销处理/对照，687,029 条；本次未核实随机分组方案 | `response_att`：到店；有干预前购物和折扣特征 | 无结果期用户级金额与实际干预成本；公开文档没有确认分组概率 | 字段设计与特征工程参考；因果评估前先核实随机化 |
| [X5 RetailHero](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html) | 零售营销处理/对照，训练集约 200,039 名客户；本次未核实随机分组方案 | `target`：干预后是否购买；含干预前购买明细 | 购买明细不是干预后收入；无实际促销成本；公开文档没有确认分组概率 | 字段设计与特征工程参考；因果评估前先核实随机化 |
| [Criteo Uplift](https://huggingface.co/datasets/criteo/criteo-uplift) | 广告增量试验，公开文件约 13,979,592 行；处理/对照 | visit、conversion | 广告而非优惠券；没有收入、毛利或真实投放成本；公开版经过非均匀抽样；CC BY-NC-SA 4.0 | 大规模 uplift 排序指标压力测试；不直接还原原始人群绝对效果 |
| [Udacity Starbucks 促销实验](https://github.com/udacity/DSND_Term2/tree/master/portfolio_exercises/Experiments) | 约 120,000 条，随机发促销/不发促销；7 个匿名特征 | purchase；课程用每次促销 $0.15、每次购买 $10 构造净收入 | 没有逐用户实收毛利或真实核销成本，也没有 App 活跃标签；仓库 LICENSE 与 README 授权声明不一致，复用前要核对 | 最小化验证双目标预算策略的教学示例，金额须标为题设常数 |
| [CUVET-policy](https://huggingface.co/datasets/anonadata/CUVET-policy) | 在线广告平台 2 周 A/B，5 档随机出价策略，约 86.7M 条 | value、cost | 广告出价而非发券；cost 是广告成本，不是券成本；数据集卡网页本次未能打开，字段与授权需下载前再核实 | 预算约束多 treatment 策略优化参考 |

## 关键来源和核对

- [MT-LIFT 官方 README](https://github.com/MTDJDSP/MT-LIFT) 明确写了 Meituan App 优惠券、RCT、样本量及字段。仓库当前文件列表仅有 README；数据通过 Google/Baidu 网盘提供，数据许可未看到明确文本。
- [Hillstrom 原作者发布说明](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) 写明随机三组与 64,000 名客户；原始 [CSV 下载链接](https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv)。该网页可由搜索索引读取，但本次浏览工具打开网页时遇到重定向错误，因此没有验证 CSV 可直接下载。
- [Criteo 官方数据卡](https://huggingface.co/datasets/criteo/criteo-uplift) 显示当前公开文件约 1,398 万行、字段与 CC BY-NC-SA 4.0。其介绍段还提到 2,500 万条；报告使用文件统计中的行数。
- [scikit-uplift 的 Lenta 数据说明](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html) 将 `response_att` 定义为到店，字段表列出干预前购买、折扣和访问间隔特征；[X5 数据说明](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html) 将 `target` 定义为干预后购买，`purchases.csv` 是干预前购买记录。两者页面未说明随机分组机制、逐用户结果期成本及概率；仅有 treatment/control 列不足以证明随机试验。
- [Udacity 官方 notebook](https://github.com/udacity/DSND_Term2/blob/master/portfolio_exercises/Experiments/Starbucks.ipynb) 写明随机分组、字段、$10/$0.15 的评分规则；[仓库 README](https://github.com/udacity/DSND_Term2) 写 CC BY-NC-ND 4.0，但 GitHub 显示仓库有 MIT LICENSE，故不推断数据能商用或再分发。
- [CUVET 论文](https://openreview.net/pdf?id=ue4YbN3wgh) 说明 86.7M 个出价机会、随机 5 档处理、value/cost 与预算优化。[Hugging Face 索引](https://huggingface.co/datasets?license=license%3Acc-by-nc-sa-4.0&p=78&sort=trending) 显示该数据集在 CC BY-NC-SA 4.0 列表中；其卡片本次未能打开，进一步使用前应以数据卡原文为准。

## 对项目的选择

公开候选中，未找到已核实同时满足 **App 站内发券 RCT、活跃、用户级毛利和实际核销券成本** 的数据；这不是接入项目的门槛。首版用 MT-LIFT 验证 Agent 对发券 uplift 代码的迭代，用 Hillstrom 验证访问/消费等不同结果口径，用 Starbucks 或 CUVET-policy 验证带成本假设的预算分配。Lenta 和 X5 可用于字段与特征工程设计，但在随机化机制核实前不用于因果成绩。缺成本或金额时，报告实际可评估的目标；公开数据上的金额代理值只用于流程测试，不作为真实净利润结论。接入用户现有 RCT dataset 时，也按其实际字段启用对应目标。

若需发布开源项目，代码中提供数据下载或映射说明；不要直接再分发授权不明或限定非商业的数据。
