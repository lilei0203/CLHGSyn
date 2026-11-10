"""Generate a human-readable report from the saved, measured metrics."""
import json
from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    metrics = ['AUC', 'AUPR', 'ACC', 'Precision', 'Recall', 'F1']
    lines = ['# CLHGSyn 实际运行结果', '',
             '本表由已保存的 `summary.json` 自动生成；数字为实际预测结果，不是论文表格。', '',
             '## 五折交叉验证', '',
             '在约 90% 开发集上执行五折，另外约 10% 为独立测试集。各折内部再留出验证集选模型。表中为均值 ± 样本标准差（ddof=1）。', '',
             '| 数据集 | ' + ' | '.join(metrics) + ' |',
             '|---|' + '---:|' * len(metrics)]
    datasets, rows = [], []
    for name in ['DrugCombDB', 'OncologyScreen']:
        directory = root / 'results' / name
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        data = json.loads((directory / 'data_report.json').read_text(encoding='utf-8'))
        datasets.append((name, summary, data))
        lines.append('| ' + name + ' | ' + ' | '.join(
            f'{summary["fold_mean"][k]:.4f} ± {summary["fold_std_ddof1"][k]:.4f}' for k in metrics) + ' |')
        rows.append({'dataset': name, **{k + '_mean': summary['fold_mean'][k] for k in metrics},
                     **{k + '_std': summary['fold_std_ddof1'][k] for k in metrics}})
    lines += ['', 'AUPR 为 PR 曲线梯形积分；AP 单独保存在 JSON/CSV 中。分类阈值固定为 0.5，没有根据测试集调整。', '',
              '## 每折结果', '']
    for name, summary, data in datasets:
        lines += [f'### {name}', '', '| 折 | 最优 epoch | 实际 epoch | AUC | AUPR | ACC | F1 |', '|---:|---:|---:|---:|---:|---:|---:|']
        folds = pd.read_csv(root / 'results' / name / 'fold_metrics.csv')
        for _, row in folds.iterrows():
            lines.append(f'| {int(row.fold)} | {int(row.best_epoch)} | {int(row.epochs_run)} | '
                         + ' | '.join(f'{row[k]:.4f}' for k in ['AUC', 'AUPR', 'ACC', 'F1']) + ' |')
        lines += ['', f'原始 {data["rows"]:,} 条，开发集 {summary["n_development"]:,} 条，独立集 {summary["n_independent"]:,} 条。'
                      f'该数据集运行总耗时约 {summary["seconds"] / 60:.1f} 分钟（含数据处理）。', '',
                  f'无序三元组 {data["unique_triplets"]:,} 个，额外重复行 {data["repeated_triplet_rows"]:,} 条；'
                  f'其中 {data["groups_with_conflicting_labels"]:,} 个组包含冲突标签，均保留且未跨集合拆分。', '']
    lines += ['## 独立测试集', '', '五个模型预测概率的平均值，仅在训练完成后评估。', '',
              '| 数据集 | AUC | AUPR | ACC | F1 |', '|---|---:|---:|---:|---:|']
    for name, summary, _ in datasets:
        lines.append('| ' + name + ' | ' + ' | '.join(f'{summary["independent_ensemble"][k]:.4f}' for k in ['AUC', 'AUPR', 'ACC', 'F1']) + ' |')
    lines += ['', '## 解释与边界', '',
              '- 保留双视图、HGNN-ECA、蛋白双聚合器、对比损失和双线性注意力；缺失的分子/表达/文本初始特征用可训练嵌入替代。',
              '- 本次使用学习率 0.001、上限 150 轮、内部验证 AUC 早停；不是严格的论文参数重现实验。详见 README。',
              '- OncologyScreen 部分折按 AUC 选择的模型在 0.5 阈值下偏向阳性，导致 ACC 波动；未用测试集校准阈值或选择新的运行结果。',
              '- DrugCombDB 的 764 药物、76 细胞系是背景网络实体数；实际组合表涉及 572 药物、73 细胞系。',
              '- 本次没有调参搜索、基线显著性检验、冷启动评估或消融实验，不能直接据此比较论文中的模型排名。',
              '- 核心公式与梯度的 6 个单元测试通过；结果独立核验记录见 `results/verification.json`。', '',
              '训练源码、完整日志、10 个 checkpoint、划分索引、逐样本概率均随项目交付。', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    pd.DataFrame(rows).to_csv(root / 'results' / 'comparison.csv', index=False)
    print('Wrote RESULTS.md and results/comparison.csv')


if __name__ == '__main__':
    main()
