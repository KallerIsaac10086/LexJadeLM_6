# 📘 LexJadeLM6 README

> 一份面向中文场景的轻量-高效语言模型全家桶，从 5 M 到 600 M 参数，一键预训练 / 微调 / 部署。

---

## 🌟 特性一览
| 特性 | 说明 |
|---|---|
| 🧩 **8 种模型** | Tiny → Ultra-M，覆盖 5 M–600 M 参数 |
| ⚙️ **多种训练范式** | 预训练、SFT、LoRA、DPO、蒸馏 |
| 🔥 **训练加速** | DDP、BF16、梯度累积、KV-Cache |
| 🚀 **即开即用** | Web Demo、OpenAI-API 服务脚本 |
| 📝 **中文优化** | 专为中文语料训练的 BPE 分词器 |

---

## 📊 模型规格总览

| 模型名称 | 参数量 | 隐藏维度 | 层数 | 注意力头 | 词表大小 | 最大长度 | MoE | 备注 |
|---|---|---|---|---|---|---|---|---|
| **LexJade6-Tiny** | 5 M | 256 | 4 | 4 (GQA-2) | 10 k | 4096 | - | 极致轻量，IoT/边缘 |
| **LexJade6-Flash** | 25 M | 256 | 8 | 4 (GQA-2) | 10 k | 4096 | - | 低延迟推理 |
| **LexJade6-Cube** | 50 M | 512 | 8 | 8 (GQA-4) | 10 k | 4096 | - | 边缘友好 |
| **LexJade6-Large** | 100 M | 512 | 12 | 8 (GQA-4) | 10 k | 4096 | - | 通用小模型 |
| **LexJade6-Extreme** | 300 M | 768 | 16 | 8 (GQA-4) | 11.4 k | 4096 | - | 高性价比 |
| **LexJade6-Ultra** | 600 M | 1 024 | 16 | 12 (GQA-4) | 15.5 k | 4096 | - | Max版本 |
| **LexJade6-Extreme-M** | 300 M-MoE | 768 | 16 | 8 (GQA-4) | 11.4 k | 4096 | 6×4 | 稀疏激活，MoGE |
| **LexJade6-Ultra-M** | 600 M-MoE | 1 024 | 16 | 12 (GQA-4) | 15.5 k | 4096 | 6×4 | 稀疏激活，MoGE |

> MoE 栏 “6×4” 表示 6 个路由专家，每 token 激活 4 个并附带 1 个共享专家。

---

## 🛠️ 环境安装

```bash
git clone https://github.com/your-org/LexJadeLM6.git
cd LexJadeLM6
pip install -r requirements.txt       # torch >=2.0, transformers >=4.30
```

---

## 🚀 30 秒上手

### 1️⃣ 预训练（以 Cube 为例）
```bash
torchrun trainer_LexJade6/train_pretrain.py \
  --model_choice 3 \
  --epochs 3 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --device cuda \
  --dtype bfloat16
```

### 2️⃣ 监督微调（以 Ultra 为例）
```bash
torchrun trainer_LexJade6/train_full_sft.py \
  --model_choice 6 \
  --epochs 2 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --data_path ./dataset/sft_data.jsonl
```

### 3️⃣ Web Demo
```bash
python scripts/web_demo.py \
  --model_path ./sft_out/LexJade6-Ultra/final_sft_LexJade6-Ultra.pth \
  --tokenizer_path ./model/tokenizer_LexJadeLM6-Ultra
```
浏览器打开 http://localhost:7860 即刻体验！

---

## 📁 目录速览

```
minimind/
├── dataset/           # 数据集处理
├── model/             # 模型 & 分词器（含各型号专用 tokenizer）
├── out/               # 预训练 ckpt
├── sft_out/           # 微调 ckpt
├── trainer_LexJade6/  # 训练脚本
├── scripts/           # 推理 & 部署脚本
├── requirements.txt
└── README.md
```

---

## 📄 License & 致谢

- **License**: Apache-2.0  
- **Issues & PR**: 欢迎随时提交！

---

> 让每一颗 GPU 都能跑起「大模型」！
