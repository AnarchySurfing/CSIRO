# CSIRO Biomass — 项目说明 ✅

本仓库实现了用于参与 Kaggle 竞赛 “CSIRO Biomass” 的训练、验证与推断流水线，包含：数据集处理、数据增强、模型定义（基于 ViT + Mamba fusion neck）、训练循环（包含 EMA、TTA、梯度累积）与评估脚本。

---

## 目录概览 🔧
- `config.py` — 全局配置（数据路径、超参、模型名等）
- `dataset.py` — 数据集与增强（Train / Val / TTA）
- `model.py` — 模型定义（Backbone + Fusion + Heads + 损失）
- `train.py` — 训练/验证函数（train_epoch / valid_epoch / TTA）
- `trainingloop.py` — 折/轮主循环（交叉验证、保存最优模型、OOF）
- `metric.py` — 自定义评估（加权 R² 等）
- `data/` — 原始与处理后数据（不提交仓库）
- `models/` — 保存/加载的模型权重
- `notebooks/` — 试验用笔记本
- `outputs/` — 预测结果与提交文件
- `requirements.txt` — 所需 Python 包

---

## 亮点 & 特性 ✨
- 使用 `timm` 的 ViT 系列 backbone，可加载本地/远端预训练权重
- 自定义的 Mamba-style Fusion Neck 用于左右图像 token 融合
- 支持 TTA（翻转/旋转）、EMA、梯度累积、warmup cosine LR 等训练技巧
- 支持分折训练（StratifiedGroupKFold）并输出 OOF 指标

---

## 快速开始 🚀
1. 克隆仓库并切换到项目目录
2. 建议创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.\.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

3. 准备数据：将竞赛数据解压并按 `config.py` 中的 `BASE_PATH` / `TRAIN_IMAGE_DIR` 指定放置，或直接修改 `config.CFG` 中路径指向本地数据。

4. 编辑 `config.py`（可修改 `FOLDS_TO_TRAIN`, `EPOCHS`, `BATCH_SIZE`, `DEVICE`, `PRETRAINED_DIR` 等），然后开始训练：

```bash
python trainingloop.py
```

> 想只训练某一折或快速调试：在 `config.py` 修改 `FOLDS_TO_TRAIN = [0]`、`EPOCHS=1`、`BATCH_SIZE` 等。

---

## 关键配置说明 ℹ️
- `CFG.BASE_PATH`、`CFG.TRAIN_CSV`、`CFG.TRAIN_IMAGE_DIR`：数据路径
- `CFG.MODEL_NAME` / `CFG.BACKBONE_PATH`：timm 模型名称与本地 backbone 权重（可选）
- `CFG.PRETRAINED_DIR`：用于恢复/微调的已保存模型目录
- `CFG.FOLDS_TO_TRAIN`：要训练的折号列表
- `CFG.DEVICE`：CPU 或 GPU（若系统上有 CUDA，会自动选择 GPU）

---

## 训练策略与保存 🔁
- 脚本使用 `StratifiedGroupKFold` 做分层分组拆分（基于 `Sampling_Date` 与 `State`）
- 每个 fold 使用 EMA 保存最佳权重到 `CFG.MODEL_DIR` 下 `best_model_fold{fold}.pth`
- 训练结束后会打印 OOF 全局/每目标 R² 与每折汇总

---

## 推断与提交（建议流程） 🧾
1. 使用保存的最佳权重加载模型（可参考 `trainingloop.py` 的加载示例）
2. 使用验证/测试集的 `BiomassDataset` 与 `get_tta_transforms` 做 TTA 推断
3. 将模型输出按竞赛要求格式化并保存到 `outputs/` 以便提交

---

## 依赖（重要） 📦
请参考 `requirements.txt`，主要依赖包括：
- `torch`, `timm`, `albumentations`, `opencv-python`, `rasterio`（如需读取地理栅格）、`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`

---

## 常见问题（FAQ） ❓
Q: 模型显存不足怎么办？
- A: 本项目启用了 ViT 的 gradient checkpointing（请在 `model.py` 中留意输出），可显著降低显存占用；同时可减小 `BATCH_SIZE`、使用 `GRAD_ACC` 梯度累积或更小的 backbone。

Q: 如何快速验证代码是否能跑通？
- A: 在 `config.py` 中设置 `EPOCHS=1`、`FOLDS_TO_TRAIN=[0]`、`BATCH_SIZE=1`，并确保 `TRAIN_CSV` 指向一个小子集（或造个小 CSV）进行 smoke test。

Q: 要如何恢复训练或微调？
- A: 将目标权重放到 `CFG.PRETRAINED_DIR` 并设置 `CFG.PRETRAINED=True`，脚本会尝试加载 `best_model_fold{fold}.pth`。

---

## 贡献 & 联系 🤝
欢迎提交 issue 或 pull request，用于修复 bug、改进配置或添加训练/推断脚本。

---

感谢使用本仓库，祝你在竞赛中取得好成绩！🏆
