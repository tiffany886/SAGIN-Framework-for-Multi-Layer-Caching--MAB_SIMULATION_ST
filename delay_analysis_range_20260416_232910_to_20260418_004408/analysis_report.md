# Delay 汇总分析报告 (20260416_232910 ~ 20260418_004408)

## 数据覆盖
- 汇总行数: 288
- 覆盖运行批次: 32
- 算法数量: 8
- Alpha 数量: 4
- 内容类型数量: 3
- 分类数量: 4

## 关键结论
- 最优算法(加权平均 Mean_Delay 最低): MAB_Contextual (0.156796)
- 最慢算法(加权平均 Mean_Delay 最高): Popularity (0.388443)
- 最优相对最慢的延迟降幅: 59.63%

## 显著性检验
- Friedman 检验: statistic=92.0417, p=4.70897e-17, scenarios=36
- Holm 校正后显著的两两比较数: 9
- 最显著的若干算法对:
  - MAB_Contextual vs MAB_Original: holm_p=0.000510028, mean_diff=-0.117194
  - MAB_Contextual vs Popularity: holm_p=0.000510028, mean_diff=-0.148213
  - Federated_MAB_EnergyAware vs MAB_Original: holm_p=0.000510028, mean_diff=-0.109664
  - Federated_MAB_EnergyAware vs Popularity: holm_p=0.000510028, mean_diff=-0.140683
  - MAB_Contextual_EnergyAware vs MAB_Original: holm_p=0.000510028, mean_diff=-0.105710
  - MAB_Contextual_EnergyAware vs Popularity: holm_p=0.000510028, mean_diff=-0.136729
  - Enhanced_Federated_MAB_EnergyAware vs MAB_Original: holm_p=0.000510028, mean_diff=-0.102790
  - Enhanced_Federated_MAB_EnergyAware vs Popularity: holm_p=0.000510028, mean_diff=-0.133809

## 各 Alpha 最优算法
- Alpha=0.25: LRU (Weighted_Mean_Delay=0.140652)
- Alpha=0.5: LRU (Weighted_Mean_Delay=0.155503)
- Alpha=1.0: MAB_Contextual (Weighted_Mean_Delay=0.148970)
- Alpha=2.0: MAB_Contextual (Weighted_Mean_Delay=0.150623)

## 产物清单
- merged_delay_summary.csv
- stats_by_algorithm.csv
- stats_by_alpha_algorithm.csv
- stats_by_content_category_algorithm.csv
- fig1_boxplot_mean_delay_by_algorithm.png
- fig2_weighted_mean_delay_vs_alpha.png
- fig3_heatmap_content_category_algorithm.png
- fig4_average_rank_by_algorithm.png
- fig5_pairwise_significance_heatmap.png
- significance_friedman_test.csv
- significance_pairwise_wilcoxon.csv
- significance_pairwise_wilcoxon_significant.csv
- stats_by_algorithm_rank.csv

由于各算法的 Run_ID 不重合，我改为按相同场景键 Alpha + Content_Type + Category 的 36 个场景做配对检验，使用 Friedman 总体检验 + Wilcoxon 两两比较，并做了 Holm 校正。脚本在 analyze_delay_range.py 里。

关键结论也已写入 analysis_report.md。现在的核心结果是：Friedman 检验 statistic=92.04，p=4.71e-17，说明 8 个算法在 36 个场景上的差异整体显著；Holm 校正后仍有 9 组两两比较显著。按全局加权均值，MAB_Contextual 仍是最优，Popularity 最差，延迟降幅约 59.63%。

新生成的图表已经放在输出目录里，包含 fig4_average_rank_by_algorithm.png 和 fig5_pairwise_significance_heatmap.png，同时也生成了对应的检验 CSV。