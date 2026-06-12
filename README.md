# MythOS V6 Omega

**MythOS V6 Omega** 是一个自进化超级智能体大模型架构，以五大数学猜想为灵感驱动推理引擎，结合双模型反幻觉机制、自进化系统和多语言自适应参数激活。

> **目标**: 8GB 显存本地运行 | 知识蒸馏 | 反幻觉 | 深度推理 | 自进化

---

## 目录

- [核心特性](#核心特性)
- [架构总览](#架构总览)
- [五大数学引擎](#五大数学引擎)
- [双模型反幻觉系统](#双模型反幻觉系统)
- [自进化系统](#自进化系统)
- [GPU 动态适配](#gpu-动态适配)
- [多语言参数激活](#多语言参数激活)
- [知识系统](#知识系统)
- [模型预设规格](#模型预设规格)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [训练指南](#训练指南)
- [消融实验](#消融实验)
- [技术栈](#技术栈)
- [License](#license)

---

## 核心特性

| 特性 | 描述 |
|------|------|
| **五大数学引擎** | Riemann 推理引擎、Hodge 思维架构、Twin Prime 认知循环、P vs NP 验证引擎、Da Vinci 统一引擎 |
| **双模型反幻觉** | Ω_G (生成器) + Ω_V (验证器) + 一致性仲裁器，候选生成→验证→仲裁三阶段 |
| **自进化系统** | 架构进化、参数进化、知识进化、进化监控，训练中自动优化模型结构 |
| **GPU 动态适配** | 自动检测硬件能力，动态调整精度、专家数、批大小 |
| **多语言激活** | 通用参数 60% + 语系参数 25% + 语言专属参数 15% 三级参数分配 |
| **知识蒸馏** | 支持 MiMo v2.5 Pro / Qwen2.5 等教师模型蒸馏 |
| **消融实验框架** | 7 种配置 (C0-C6) 自动化消融实验 |
| **课程学习** | 多阶段难度递增训练 |
| **Colab 支持** | 提供 Google Colab 一键训练 notebook |

---

## 架构总览

```
输入 Token IDs
    │
    ▼
┌──────────────────────────────────────┐
│  语言检测 (LanguageClassifier)        │  ← 自动识别中/英/日/韩/阿拉伯/西里尔等
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Ω_G 生成器 (OmegaGenerator)         │  ← Transformer + 五大数学引擎
│  ├─ Riemann 推理引擎                  │    索引注意力 + 关键线推理
│  ├─ Hodge 思维架构                    │    代数循环分解 + 层级抽象
│  ├─ Twin Prime 认知循环               │    直觉 + 反思双阶段
│  ├─ P vs NP 验证引擎                  │    多维度正确性验证
│  └─ Da Vinci 统一引擎                 │    跨域映射 + 知识融合
│  → 输出 K 个候选 logits               │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Ω_V 验证器 (OmegaVerifier)          │  ← 独立轻量模型 (~1/3 参数)
│  ├─ 多维度验证头                      │    数学/代码/逻辑/事实
│  ├─ Token 级错误定位                  │    精确定位错误位置
│  └─ 修正建议生成                      │    反馈给生成器
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  一致性仲裁器 (ConsistencyArbiter)    │  ← 候选选择 + 重新生成决策
│  ├─ K 候选排名                        │    基于验证分数
│  ├─ 一致性检查                        │    ≥min_agreement_ratio 通过
│  └─ 反馈信号生成                      │    用于重新生成
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  自进化系统 (Evolution System)        │  ← 训练中自动优化
│  ├─ 架构进化                          │    专家生长/死亡、注意力头合并
│  ├─ 参数进化                          │    参数裁剪/扩展
│  ├─ 知识进化                          │    跨域知识迁移
│  └─ 进化监控                          │    健康度评估 + 异常检测
└──────────────┬───────────────────────┘
               │
               ▼
          输出 logits / loss
```

---

## 五大数学引擎

### 1. Riemann 推理引擎

灵感来源: **黎曼猜想 (Riemann Hypothesis)**

| 机制 | 实现 |
|------|------|
| 索引注意力 (Prime-Indexed Attention) | 使用素数分布加权 token 重要性，素数 token 承载更高推理权重 |
| 关键线推理 (Critical Line Reasoning) | 在确定性与不确定性之间寻求平衡，模拟 Re(s) = 1/2 的临界行为 |
| 零点深化 (Zero-Point Deepening) | 遇到矛盾或高不确定性时递归精化，类似 ζ(s) 零点标记关键点 |
| 素数定理资源分配 | 按 π(x) ~ x/ln(x) 比例分配计算资源 |

核心实现: `mythos/core/v6/math_engines/riemann_engine.py` (21KB)

### 2. Hodge 思维架构

灵感来源: **霍奇猜想 (Hodge Conjecture)**

| 机制 | 实现 |
|------|------|
| 代数循环分解 | 将推理分解为不可约的 "代数循环"：事实检索、逻辑演绎、类比映射、反事实推理、元认知、抽象 |
| Hodge 过滤 | 从具体到抽象的层级抽象，通过门控残差连接实现 |
| 有理组合器 | 量化权重组合组件，提高可解释性 |
| Hodge 对偶性 | 同时计算分析和综合两个视角 |

核心实现: `mythos/core/v6/math_engines/hodge_thinking.py` (15KB)

### 3. Twin Prime 认知循环

灵感来源: **孪生素数猜想 (Twin Prime Conjecture)**

| 阶段 | 模拟 | 实现 |
|------|------|------|
| 直觉阶段 (System 1) | 快速、并行、模式匹配 | 模板匹配 + 置信度估计 |
| 反思阶段 (System 2) | 慢速、串行、逻辑验证 | 逻辑一致性检查 |
| 一致性检查 | 确保两阶段结果一致 | 可调阈值 |
| 间距适应 | 根据问题复杂度调整思考深度 | 动态间距参数 |

核心实现: `mythos/core/v6/math_engines/twin_cognition.py` (14KB)

### 4. P vs NP 验证引擎

灵感来源: **P vs NP 问题**

| 维度 | 验证内容 |
|------|----------|
| 数学 | 方程平衡、数值精度、量词一致性 |
| 代码 | 类型安全、定义-使用一致性、控制流 |
| 逻辑 | 前提-结论有效性、矛盾检测、反例生成 |
| 事实 | 知识一致性 |

核心特性:
- **错误定位**: Token 级错误位置精确定位
- **修正反馈**: 生成修正建议反馈给生成器
- **复杂度适应**: 根据问题难度调整验证深度

核心实现: `mythos/core/v6/math_engines/pnp_verification.py` (12KB)

### 5. Da Vinci 统一引擎

灵感来源: **达芬奇的跨学科天才**

| 能力 | 实现 |
|------|------|
| 域签名编码 | 每个知识域的独特表征向量 |
| 跨域映射 | 不同知识域之间的翻译 |
| 知识融合 | 多域知识组合 |
| 创造性发现 | 发现新颖的跨域连接 |
| 元调度 | 决定激活哪些引擎 |

核心实现: `mythos/core/v6/math_engines/da_vinci_engine.py` (19KB)

---

## 双模型反幻觉系统

### Ω_G 生成器

- Transformer 架构，集成五大数学引擎
- 每层使用引擎增强推理能力
- 输出 K 个候选 (默认 K=3)

### Ω_V 验证器

- 独立轻量模型 (~1/3 Ω_G 参数)
- 无参数共享，专门训练用于正确性验证
- 多维度验证 + Token 级错误定位 + 修正建议

### 一致性仲裁器

```
if ≥ min_agreement_ratio 候选通过验证:
    → 接受最佳候选
elif 1 个候选通过:
    → 接受 + 低置信度警告
elif 0 个通过:
    → 触发重新生成 + 反馈信号
```

生成时支持验证循环 (`generate_verified`)：最多尝试 `max_attempts` 次，失败时提高温度重新生成。

---

## 自进化系统

### 架构进化 (ArchitectureEvolution)

- **专家生长**: 利用率超过阈值时分裂专家
- **专家死亡**: 利用率低于阈值时移除专家
- **注意力头合并**: 高相似度头合并以减少冗余
- **推理路径进化**: 优化推理路径选择

### 参数进化 (ParameterEvolution)

- 参数裁剪/扩展策略
- 基于贡献度的选择性更新

### 知识进化 (KnowledgeEvolution)

- 跨域知识迁移
- 知识图谱动态扩展

### 进化监控 (EvolutionMonitor)

- 健康度评估
- 异常检测
- 指标历史记录

---

## GPU 动态适配

`HardwareDetector` 自动检测硬件并分类:

| 层级 | VRAM | 精度 | 最大层数 | 最大专家数 | 批大小 |
|------|------|------|---------|-----------|--------|
| datacenter | ≥40GB | bf16 | 80 | 256 | 8 |
| high_end | ≥24GB | fp16 | 48 | 128 | 4 |
| mid_range | ≥12GB | fp16 | 24 | 64 | 2 |
| low_end | ≥6GB | int8 | 16 | 32 | 1 |
| minimal | >0GB | int4 | 8 | 8 | 1 |
| cpu_only | 0GB | int4 | 4 | 4 | 1 |

支持精度: `fp32`, `bf16`, `fp16`, `int8`, `int4`, `nf4`

---

## 多语言参数激活

三级参数分配:

```
┌─────────────────────────────────────┐
│  通用参数 (60%)                      │  ← 所有语言共享
├─────────────────────────────────────┤
│  语系参数 (25%)                      │  ← 汉藏/印欧/阿尔泰等 12 语系
├─────────────────────────────────────┤
│  语言专属参数 (15%)                  │  ← 中文/英文/日文等 64 种语言
└─────────────────────────────────────┘
```

支持语言检测: 中文、日文、韩文、阿拉伯文、西里尔文、天城文、拉丁文等。

---

## 知识系统

| 组件 | 功能 |
|------|------|
| DomainSignatureEncoder | 知识域签名编码 |
| KnowledgeGraphEmbedding | 知识图谱嵌入 |
| AnalogyReasoningNetwork | 类比推理网络 |
| StructureMappingEngine | 结构映射引擎 |
| BreakthroughThinkingModule | 突破性思维模块 (对抗训练: 生成器产生新颖组合，判别器评估有用性) |

---

## 模型预设规格

| 预设 | dim | layers | vocab | max_seq | 参数量 | VRAM | 适用场景 |
|------|-----|--------|-------|---------|--------|------|----------|
| `mythos-v6-omega-micro` | 512 | 8 | 16K | 2K | ~80M | ~4GB | 开发调试、Colab 免费版 |
| `mythos-v6-omega-1b` | 2048 | 24 | 128K | 131K | ~1B | ~16GB | 标准训练、Colab Pro |
| `mythos-v6-omega-70b` | 8192 | 80 | 128K | 1M | ~70B | 多卡 | 旗舰级性能 |

---

## 项目结构

```
Myth OS/
├── mythos/                          # 核心包
│   ├── core/
│   │   └── v6/                      # ★ V6 Omega 核心
│   │       ├── config_v6.py         # 完整配置系统 (所有预设)
│   │       ├── model_v6.py          # 主模型 MythosV6Omega
│   │       ├── tokenizer.py         # 语言感知 BPE 分词器
│   │       ├── train.py             # 训练入口
│   │       ├── ablation_*.py        # 消融实验框架
│   │       ├── math_engines/        # 五大数学引擎
│   │       │   ├── riemann_engine.py
│   │       │   ├── hodge_thinking.py
│   │       │   ├── twin_cognition.py
│   │       │   ├── pnp_verification.py
│   │       │   └── da_vinci_engine.py
│   │       ├── twin/                # 双模型系统
│   │       │   ├── omega_generator.py
│   │       │   ├── omega_verifier.py
│   │       │   ├── consistency_arbiter.py
│   │       │   └── twin_trainer.py
│   │       ├── evolution/           # 自进化系统
│   │       │   ├── architecture_evolution.py
│   │       │   ├── parameter_evolution.py
│   │       │   ├── knowledge_evolution.py
│   │       │   └── evolution_monitor.py
│   │       ├── adaptation/          # GPU 适配 + 语言激活
│   │       │   ├── hardware_detector.py
│   │       │   ├── dynamic_compute_graph.py
│   │       │   ├── precision_manager.py
│   │       │   ├── language_classifier.py
│   │       │   └── language_activator.py
│   │       ├── knowledge/           # 知识系统
│   │       │   ├── domain_signature.py
│   │       │   ├── knowledge_graph.py
│   │       │   ├── analogy_network.py
│   │       │   ├── structure_mapping.py
│   │       │   └── breakthrough.py
│   │       ├── training/            # 训练组件
│   │       │   ├── distill_trainer.py
│   │       │   ├── checkpoint.py
│   │       │   └── distributed.py
│   │       ├── data/                # 数据管线
│   │       │   ├── pipeline.py
│   │       │   ├── chinese_cleaner.py
│   │       │   └── curriculum.py
│   │       ├── evaluation/          # 评估系统
│   │       │   └── eval_suite.py
│   │       ├── inference/           # 推理引擎
│   │       │   └── engine.py
│   │       └── export/              # 量化导出
│   │           └── quantize.py
│   ├── distill/                     # 蒸馏管线
│   ├── skills/                      # Agent 技能
│   ├── think/                       # 深度思考
│   ├── verify/                      # 验证系统
│   └── training/                    # 通用训练
├── scripts/                         # 训练/推理脚本
│   ├── v6_mimo_distill.py           # V6 MiMo 蒸馏
│   ├── train_all_stages.py          # 多阶段训练
│   ├── evaluate_model.py            # 模型评估
│   └── ...
├── configs/                         # 配置文件
├── data/                            # 训练数据
├── tokenizer/                       # 分词器
├── docs/                            # 文档
├── tests/                           # 测试
├── juichi_cloud/                    # 云 GPU 训练指南
├── colab_train_v6.py                # Colab 训练脚本
├── colab_v6_final.ipynb             # Colab Notebook
├── config.yaml                      # 主配置
├── requirements.txt                 # 依赖
├── Dockerfile                       # Docker 部署
├── docker-compose.yml               # Docker Compose
├── install.bat                      # Windows 安装
└── install.sh                       # Linux 安装
```

---

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 训练)
- 8GB+ VRAM (micro 预设)

### 安装

**Windows:**
```bat
install.bat
```

**Linux/macOS:**
```bash
bash install.sh
```

**手动安装:**
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsuna.edu.cn/simple
```

### 快速验证

```python
from mythos.core.v6.model_v6 import build_omega

# 构建 micro 模型 (~80M 参数, ~4GB VRAM)
model = build_omega("mythos-v6-omega-micro")
print(model.get_model_info())

# 构建 1B 模型
model = build_omega("mythos-v6-omega-1b")
```

### 推理

```python
from mythos.core.v6.model_v6 import build_omega
from mythos.core.v6.tokenizer import LanguageAwareTokenizer

model = build_omega("mythos-v6-omega-micro")
tokenizer = LanguageAwareTokenizer(vocab_size=16000)

# 带验证的生成
input_ids = torch.tensor([tokenizer.encode("What is 2+2?")])
output = model.generate_verified(input_ids, max_new_tokens=128)
print(tokenizer.decode(output[0].tolist()))
```

---

## 训练指南

### 方式一: 标准训练

```bash
python -m mythos.core.v6.train \
    --preset mythos-v6-omega-micro \
    --mode standard \
    --data data/ \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 32 \
    --lr 2e-4
```

### 方式二: 知识蒸馏 (推荐)

```bash
# 从 MiMo v2.5 Pro 蒸馏
python -m mythos.core.v6.train \
    --preset mythos-v6-omega-1b \
    --mode distill \
    --teacher XiaomiMiMo/MiMo-v2.5-Pro \
    --data data/ \
    --alpha 0.5 --beta 0.5 --temperature 2.0
```

### 方式三: MiMo 蒸馏管线

```bash
# 一键生成蒸馏数据 + 训练
python scripts/v6_mimo_distill.py --mode full --samples 500 --preset mythos-v6-omega-micro
```

### 方式四: Google Colab

1. 上传 `colab_v6_final.ipynb` 到 Colab
2. 选择 T4 GPU (免费) 或 A100 (Pro)
3. 按 cell 顺序执行

### 方式五: 云 GPU (矩池云)

```bash
cd juichi_cloud/
bash run.sh           # 单卡 A100
bash run.sh --multi   # 双卡 A100
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--preset` | mythos-v6-omega-1b | 模型预设 |
| `--mode` | distill | 训练模式 (standard/distill) |
| `--teacher` | XiaomiMiMo/MiMo-v2.5-Pro | 教师模型 |
| `--alpha` | 0.5 | CE loss 权重 |
| `--beta` | 0.5 | KL loss 权重 |
| `--temperature` | 2.0 | 蒸馏温度 |
| `--lr` | 2e-4 | 学习率 |
| `--batch-size` | 4 | 每 GPU 批大小 |
| `--grad-accum` | 32 | 梯度累积步数 |
| `--max-steps` | 200000 | 最大训练步数 |
| `--gradient-checkpointing` | false | 梯度检查点 (省显存) |
| `--cpu-offload-teacher` | false | CPU 卸载教师模型 |

### DeepSpeed 训练

```bash
python -m mythos.core.v6.train \
    --preset mythos-v6-omega-1b \
    --mode distill \
    --data data/ \
    --deepspeed --ds-stage 2
```

---

## 消融实验

7 种消融配置，用于验证各引擎贡献:

| 配置 | 引擎 | 说明 |
|------|------|------|
| C0 | 无 | 基线 (纯 Transformer) |
| C1 | Riemann | 仅 Riemann 推理引擎 |
| C2 | Hodge | 仅 Hodge 思维架构 |
| C3 | Twin Cognition | 仅 Twin Prime 认知循环 |
| C4 | P vs NP | 仅 P vs NP 验证引擎 |
| C5 | Da Vinci | 仅 Da Vinci 统一引擎 |
| C6 | 全部 | 五大引擎全开 |

运行消融实验:

```bash
# 全部配置
python -m mythos.core.v6.ablation_runner --steps 100 --all

# 单个配置
python -m mythos.core.v6.ablation_runner --steps 50 --config C6
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 框架 | PyTorch 2.0+ |
| 分布式 | DeepSpeed ZeRO-2/3 |
| 精度 | bf16 / fp16 / int8 / int4 / nf4 |
| 分词器 | SentencePiece BPE + Byte-level fallback |
| 数据 | JSONL 流式处理 |
| 评估 | Perplexity / MMLU-style / Code / Math |
| 部署 | Docker / Docker Compose / FastAPI |
| 优化 | Flash Attention 2 / Triton / xformers / liger-kernel |

---

## 训练数据格式

JSONL 格式，每行一个样本:

```json
{
  "text": "训练文本内容",
  "task": "math_reasoning",
  "quality_score": 7.5,
  "conversations": [
    {"role": "system", "content": "系统提示"},
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "助手回答"}
  ]
}
```

蒸馏数据额外包含教师输出:

```json
{
  "input_text": "问题",
  "output_text": "教师回答",
  "reasoning_content": "教师推理过程",
  "quality_score": 8.0
}
```

---

## 版本历史

| 版本 | 代号 | 核心创新 |
|------|------|----------|
| V1-V3 | - | 基础 Transformer + MoE |
| V4 | Prometheus | MLA + 稀疏 MoE + 课程学习 |
| V5 | Titan | MHA** + UltraMoE + H³ 推理 + SSM 记忆 + MTP |
| **V6** | **Omega** | **五大数学引擎 + 双模型反幻觉 + 自进化 + 多语言激活** |

---

## License

本项目仅供研究和教育用途。

---

## 致谢

本项目的数学引擎灵感来源于以下数学猜想:
- **黎曼猜想** (Riemann Hypothesis) — 索引注意力与关键线推理
- **霍奇猜想** (Hodge Conjecture) — 代数循环分解与层级抽象
- **孪生素数猜想** (Twin Prime Conjecture) — 直觉-反思双阶段认知
- **P vs NP 问题** — 验证比生成更容易的核心洞见
- **达芬奇的跨学科方法** — 跨域知识统一与创造性发现
