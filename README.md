# CLHGSyn 模块化复现

根据用户提供的论文《Dual-Topology Contrastive Learning for Drug Synergy Prediction》和框架图实现。**这是基于所提供数据的适配复现，不是作者官方源码，也不声称严格重现原文数值。** 支持两个数据集的完整五折训练，并保存模型、划分与预测结果。

`CLHGSyn_code.zip` 包含源码、必要原始数据和运行说明，不包含训练 checkpoint 和结果；解压后进入 `CLHGSyn` 目录即可运行。工作目录中已经完成的训练结果保存在 `results` 下，某个数据集出现 `summary.json` 才代表其全部五折完成。

## 快速运行

建议 Python 3.9–3.12。依赖仅为 PyTorch、NumPy、pandas、SciPy、scikit-learn 和 openpyxl，不需要 PyG、DGL 或 RDKit。

```powershell
python -m pip install -r requirements.txt
python run_cv.py --data-zip "D:/drug/Drug combinations-22-7/GraphSynergy-master/data.zip" --dataset all --epochs 150 --lr 0.001 --patience 25 --min-epochs 30 --output-dir results_rerun
```

交付目录已包含从 zip 提取的必要原始数据，可省略 `--data-zip`。不会改动用户原始 zip 或 GraphSynergy 项目；不读取旧 pickle、不沿用其中不稳定的 ID 映射。

本机实际使用 `D:/Anaconda/python.exe`，其中 PyTorch 为 CPU 版，训练不需要安装新环境。可用以下 PowerShell 脚本按本次设置重跑，结果保存到带时间戳的新目录：

```powershell
.\run_reproduction.ps1 -Python "D:/Anaconda/python.exe"
```

仅跑一个数据集：

```powershell
python run_cv.py --dataset OncologyScreen --epochs 150 --lr 0.001 --patience 25 --output-dir results_oneil_rerun
python run_cv.py --dataset DrugCombDB --epochs 150 --lr 0.001 --patience 25 --output-dir results_drugcomb_rerun
```

`--device auto` 默认有 CUDA 则用 CUDA，否则用 CPU；`--threads 4` 控制 CPU 线程。已经完成的结果目录默认禁止覆盖；重新运行请指定新目录。

## 代码结构

| 文件 | 职责 |
|---|---|
| `run_cv.py` | 命令行入口、配置和两个数据集调度 |
| `clhgsyn/data.py` | zip 提取、原始表读取、稳定 ID、重复三元组分组、蛋白邻域采样 |
| `clhgsyn/graph.py` | 只根据当前训练集阳性三元组构造归一化超图 |
| `clhgsyn/layers.py` | HGNN-ECA、LightGCN 均值聚合、双线性聚合、双线性注意力 |
| `clhgsyn/losses.py` | 药物和细胞系分别计算的跨视图对比损失 |
| `clhgsyn/model.py` | 双视图拼接、三路交互、MLP 分类 |
| `clhgsyn/train.py` | 五折划分、内部验证早停、分块训练、保存模型与预测 |
| `clhgsyn/metrics.py` | AUC、AUPR、AP、ACC、Precision、Recall、F1 |
| `predict.py` | 加载任一折模型，对已知药物/细胞系预测 |
| `verify_results.py` | 独立核对保存结果、划分、超边和指标 |
| `summarize_results.py` | 从实际运行结果生成中文汇总报告 |
| `tests/test_core.py` | 关键公式与梯度正确性检查 |

## 与论文的方法对应

1. **HTI / HGNN-ECA**：`D_v^-1/2 H D_e^-1 H^T D_v^-1/2`，超边权重均为 1。两层线性变换和 ReLU，每层加入节点全局均值池化、核宽 3 的一维卷积和 Sigmoid 通道注意力。孤立节点的归一化系数为 0，不添加未说明的协同超边。
2. **PTI / 双聚合器**：第一跳从 DPI/CPI 采样蛋白；第二跳从第一跳采样节点沿 PPI 扩展。每跳最多 32 个，固定种子、无放回，排除已访问节点，合并去重并使用掩码。LightGCN 按论文公式进行邻域均值聚合；双线性部分进行共享无偏置投影、两两逐元素乘积平均与 LeakyReLU，再以 `lambda=0.4` 融合。
3. **双线性聚合的等价加速**：论文对所有 `i,j` 求和且包含 `i=j`，因此 `mean_ij(W h_i ⊙ W h_j) = mean_i(W h_i)^2`。实现用这个精确恒等式，不枚举所有蛋白对；测试与显式枚举结果一致。由于该式非负，图中的 LeakyReLU 在此处实际为恒等映射，这是原公式本身的性质。
4. **对比损失**：HTI 作为锚点，相同实体的 PTI 为正样本，其他同类型实体的 HTI/PTI 都为负样本，屏蔽 HTI 自身。药物和细胞系分开计算；温度默认 0.5。按论文单向公式实现，没有额外加入反向损失。
5. **预测头**：每个实体将 HTI 和 PTI 拼接为 256 维，三组独立矩阵计算 drug–drug / drug–cell / drug–cell 分数，经 Softmax 对三组逐元素交互表示加权，输入 `256→128→1` MLP。训练使用 logits 和 BCE，输出时 Sigmoid。两个药物的 ID 先排序，使交换输入顺序时预测相同。
6. **联合损失**：`(sum(BCE) + eta * sum(CL_drug, CL_cell)) / N_train`，`eta=0.3`。这是对论文总和损失统一除以训练样本数，并非分别对 BCE 和 CL 求均值后直接相加。为了控制内存，分类头分块计算并累积梯度，每个 epoch 只进行一次完整 Adam 更新；分块实现与直接完整损失的梯度已核对。

## 数据适配与复现边界

- zip 没有 SMILES 分子图、原始细胞系表达矩阵、蛋白文本语料或 Word2Vec 向量。因此药物/细胞系初始特征使用可训练嵌入，蛋白使用单位标准差高斯初始化的可训练嵌入；**没有实现缺少输入的 SMILES-GCN、表达 FNN 或文本 Word2Vec 初始化**。DPI/CPI/PPI 都真实参与 PTI。
- 原始包中的 PPI 去重后为 217,160 行、15,970 个蛋白，与论文中整合资源的总规模不同；实际数据统计和 SHA-256 见各数据集 `data_report.json`。超图对重复阳性三元组只建立一条二值超边，分类损失仍保留全部实验行。
- 论文明确的参数保留为：两个视图均 128 维、2 跳、`lambda=0.4`、`eta=0.3`、dropout 0.3。层数 2、ECA 核宽 3、温度 0.5、每跳采样数 32、MLP 隐藏层 128、优化器 Adam、epoch 上限和早停细节属于实现补充。
- CLI 默认学习率保留论文的 **0.0001**，默认 epoch 上限 200；本次实际运行显式指定 **0.001、最多 150 轮、至少 30 轮、patience 25**，目的是先得到可运行的完整版本。没有用外层测试折或独立测试集选择超参数。
- 不进行基线比较、消融、细胞系留出实验或临床案例复现。当前只处理已有实体范围内的随机三元组预测，不能据此声称具有新药物或新细胞系泛化能力。

## 五折协议与防泄漏

采用论文的“约 90% 开发集 + 约 10% 独立测试集”总体结构。先按无序药物对与细胞系分组，用固定种子划出独立集；在开发集上进行 `StratifiedGroupKFold(5)`。由于重复样本成组，独立集占比不保证精确 10%，类别比例也不保证精确一致。

每个外层折的训练部分，再按组拿出约 10% 作为内部验证集。只用内部验证 AUC 选最优 checkpoint；外层测试折只在训练结束后评估一次。该额外内部验证步骤比原文简述更严格，也减少了实际训练样本。超图仅使用**当前实际训练部分的阳性记录**，内部验证、外层测试和独立测试三者都不参与超图。DPI/CPI/PPI 属于与协同标签无关的背景结构，在各折共享。

DrugCombDB 原始 69,436 行保留，不擅自对重复实验的分数取平均。相同药物对和细胞系，包括调换药物顺序的记录，被放入同一个组，不能跨训练/验证/测试；冲突标签的组也完整保留并记录在 `data_report.json`。所有节点 ID 由排序确定。

如果只希望在全数据上直接做五折，不留 10% 独立集：

```powershell
python run_cv.py --dataset all --holdout 0 --epochs 150 --lr 0.001 --output-dir results_full5cv
```

## 结果文件

- `results/<dataset>/summary.json`：五折均值、样本标准差（`ddof=1`）、汇总 OOF 指标、独立集五模型平均概率指标。
- `fold_metrics.csv`：每折指标、训练/验证/测试数量、最优轮次。
- `oof_predictions.csv`：开发集每条记录恰好一次的外层测试预测。
- `fold_1` 至 `fold_5`：`best.pt`、`history.csv`、`split_indices.npz`、`training_positive_hyperedges.npy`、测试与独立测试预测。
- `config.json`、`node_mapping.json`、`protein_neighbors.npz`：运行参数、节点映射与固定邻域，可恢复模型。
- `samples.csv`、`data_report.json`：原始行号、标签、组号、数据统计及原始文件 SHA-256。
- `results_training.log`：正式运行日志。开发时的两轮 smoke 结果未混入正式结果。

ACC/Precision/Recall/F1 固定阈值为 0.5。AUPR 使用 PR 曲线梯形积分，另行报告 `AP`（average precision），避免混用两种定义。汇总 OOF 指标与各折均值不同，分别存储。

## 验证与单独预测

```powershell
python -m unittest discover -s tests -v
python verify_results.py --results results
python summarize_results.py
python predict.py --result-dir results/OncologyScreen --fold 1 --input examples/triplets.csv --output predictions.csv
```

预测输入列为 `drug_a,drug_b,cell`，药物使用 DrugBank ID，细胞系名称与数据一致。仅加载本项目自行训练或可信来源的 checkpoint。全部训练、预测和核验命令均可单独运行。
