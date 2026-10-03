# 营销 Uplift 公开数据集调研

核对日期：2026-10-03。为「App 站内优惠券促活」项目寻找可复现的随机对照数据；增量活跃和增量净收益是否都能比较，取决于具体数据的字段。这里的「公开」只表示可获取；授权与可商用性单独判断。

## 候选数据

| 数据集 | 随机对照与规模 | 可用结果/成本 | 与目标的差距 | 建议用途 |
| --- | --- | --- | --- | --- |
| [MT-LIFT](https://github.com/MTDJDSP/MT-LIFT) | 美团 App 外卖优惠券随机试验；5,541,842 条，5 个 treatment，99 个匿名特征 | click、conversion | 发布的字段表没有活跃、收入/毛利、核销或实际券成本；仓库未标明数据许可证 | 最贴近业务的发券 uplift 算法基准；不能用其验证真实净利润 |
| [Lenta](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html) | 零售营销处理/对照，687,029 条；本次未核实随机分组方案 | `response_att`：到店；有干预前购物和折扣特征 | 无结果期用户级金额与实际干预成本；公开文档没有确认分组概率 | 字段设计与特征工程参考；因果评估前先核实随机化 |
| [X5 RetailHero](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data) | 零售短信营销处理/对照，训练集 200,039 名客户；官方赛题未给出随机分组方案 | `target`：干预后是否购买；客户画像与干预前购买明细 | 无结果期收入或实际促销成本；公开文档没有确认分组概率 | GPU 模型训练与预测检查；因果评估前先核实随机化 |
| [Criteo Uplift v2.1](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) | 广告增量随机试验，修正后的公开文件 13,979,592 行；处理/对照 | visit、conversion | 广告而非优惠券；没有收入、毛利或真实投放成本；公开版经过隐私抽样；CC BY-NC-SA 4.0 | GPU 模型训练与预测检查；不直接还原原始人群绝对效果 |
| [MegaFon Uplift](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_megafon.html) | 模拟生成的电信营销数据，训练集 600,000 行 | `conversion`；50 个匿名特征 | 非真实随机实验，也没有收入或实际成本 | 不用于本项目的真实数据验证 |
| [Udacity Starbucks 促销实验](https://github.com/udacity/DSND_Term2/tree/master/portfolio_exercises/Experiments) | 已发布的 `training.csv` 有 84,534 条，随机发促销/不发促销；7 个匿名特征 | purchase；课程设定每次促销 $0.15 | 没有逐用户实收毛利或真实核销成本，也没有 App 活跃标签；仓库 LICENSE 与 README 授权声明不一致，复用前要核对 | 固定 RCT 上的预算策略端到端评测；费用只作题设成本 |
| [CUVET-policy](https://huggingface.co/datasets/anonadata/CUVET-policy) | 在线广告平台 2 周 A/B，5 档随机出价策略，约 86.7M 条 | value、cost | 广告出价而非发券；cost 是广告成本，不是券成本；数据集卡网页本次未能打开，字段与授权需下载前再核实 | 预算约束多 treatment 策略优化参考 |

## 关键来源和核对

- [MT-LIFT 官方 README](https://github.com/MTDJDSP/MT-LIFT) 明确写了 Meituan App 优惠券、RCT、样本量及字段。仓库当前文件列表仅有 README；数据通过 Google/Baidu 网盘提供，数据许可未看到明确文本。
- [Criteo 官方数据页](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) 明确说明原始广告增量试验随机分配，并列出 v2.1 修正文件 13,979,592 行、visit/conversion/exposure 字段与隐私抽样；许可证为 CC BY-NC-SA 4.0。原版约 2,500 万行，有跨广告主特征泄漏，不能与 v2.1 混用。
- [scikit-uplift 的 Lenta 数据说明](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html) 将 `response_att` 定义为到店，字段表列出干预前购买、折扣和访问间隔特征；[X5 赛题原文](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data) 将 `target` 定义为短信后购买，`purchases.csv` 是干预前购买记录。两者页面未说明随机分组机制、逐用户结果期成本及概率；仅有 treatment/control 列不足以证明随机试验。[scikit-uplift 的 MegaFon 数据说明](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_megafon.html) 明确标注该数据为模拟生成，因此排除真实数据验证。
- [Udacity 官方 notebook](https://github.com/udacity/DSND_Term2/blob/master/portfolio_exercises/Experiments/Starbucks.ipynb) 写明随机分组、字段、$10/$0.15 的评分规则；本项目只用了题设的 $0.15 发送成本，没有把 $10 当成实际用户收入。[仓库 README](https://github.com/udacity/DSND_Term2) 写 CC BY-NC-ND 4.0，但 GitHub 显示仓库有 MIT LICENSE，故不推断数据能商用或再分发。
- [CUVET 论文](https://openreview.net/pdf?id=ue4YbN3wgh) 说明 86.7M 个出价机会、随机 5 档处理、value/cost 与预算优化。[Hugging Face 索引](https://huggingface.co/datasets?license=license%3Acc-by-nc-sa-4.0&p=78&sort=trending) 显示该数据集在 CC BY-NC-SA 4.0 列表中；其卡片本次未能打开，进一步使用前应以数据卡原文为准。

## 对项目的选择

公开候选中，未找到已核实同时满足 **App 站内发券 RCT、活跃、用户级毛利和实际核销券成本** 的数据；这不是接入项目的门槛。本轮按用户要求暂缓 MT-LIFT，使用 Starbucks 验证固定 RCT 的端到端预算策略，Criteo 和 X5 检查 GPU 模型训练与预测。Criteo 的公开抽样和 X5 未核实的随机分组机制，使它们目前不适合生成项目的逆概率加权因果报告。缺成本或金额时，只报告数据实际支持的目标，不把费用假设当作实际净利润。接入用户现有 RCT dataset 时，也按其实际字段启用对应目标。

若需发布开源项目，代码中提供数据下载或映射说明；不要直接再分发授权不明或限定非商业的数据。
