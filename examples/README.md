# 示例数据

`demo.csv` 是固定种子的**合成测试数据**，只用于验证项目能运行和复现；其结果不能作为实际营销效果。

`hillstrom.json` 将公开的 [Hillstrom 邮件随机实验](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) 映射为单一邮件与不发送的二元比较。先从原作者提供的 [CSV](https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv) 下载并保存为本目录的 `hillstrom.csv`。选择其中两个原始等概率组后，条件处理概率为 0.5。此数据没有用户 ID、毛利和邮件实际成本；报告中的 `visit` 是网站访问，`spend` 是销售额，均不等同于 App 活跃或利润。
