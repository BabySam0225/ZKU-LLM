<div align="center">

# 🎓 UniAssist — 大学生智能问答系统

**基于 RAG + GraphRAG + 多 Agent 协同的高校智能服务平台**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--V3%20%7C%20R1-orange.svg)](https://platform.deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

[快速开始](#-快速开始) · [功能演示](#-功能演示) · [模块介绍](#-模块介绍) · [路线图](#-未来规划)

</div>

---

## 📌 项目背景

高校学生在校期间面临大量日常困惑，却往往找不到高效的获取渠道：

- **政策咨询**：奖学金申请条件、转专业流程、违纪处分规定……文件分散、难以检索
- **心理压力**：学业焦虑、人际关系、情绪困扰……资源匮乏、不敢开口
- **就业迷茫**：不知道本专业能去哪里、薪酬多少、需要什么技能……信息不透明

UniAssist 针对以上三类核心场景，构建了一套**完全本地化、低成本、可部署**的智能问答系统，让学生随时得到准确、有温度的回答。

### 解决了什么问题

| 痛点 | 传统方案 | UniAssist |
|------|---------|-----------|
| 政策文件分散、难查找 | 手动翻阅 PDF/Word | RAG 精准检索，秒级响应 |
| 心理求助门槛高 | 预约咨询、担心隐私 | 24h 匿名 AI 倾听，危机自动干预 |
| 就业数据不透明 | 靠学长打听 | 历年真实数据统计分析，有据可查 |
| 跨领域复杂问题 | 找多个部门 | 多 Agent 并行协同，一问全解 |
| 幻觉/不准确 | 通用 LLM 乱说 | GraphRAG 知识图谱 + 检索增强 |

---

## ✨ 核心特性

- 🔍 **混合检索**：向量（FAISS + BGE-M3）× BM25 × RRF 融合，召回率显著优于单一检索
- 🕸 **GraphRAG**：知识图谱实体关系建模，支持跨 chunk 多跳推理
- 🧠 **历史感知改写**：Query Rewrite 结合对话历史，模糊问题自动补全为完整检索语句
- 🤖 **三 Agent + 主控**：辅导员 / 心理 / 就业三个专业 Agent，主控 Agent 智能路由并行协同
- 💙 **情绪感知**：心理模块自动识别情绪类型（焦虑/抑郁/压力），危机情况直接干预
- 🌊 **流式输出**：SSE 实时推送，前端字符级响应，支持 FastAPI 后端对接
- 🔬 **深度思考**：一键切换 DeepSeek-R1，复杂问题显示推理过程
- 📊 **完整评测**：LLM-as-Judge 自动评分 + 人工标注 + 版本回归测试

---

## 🏗 系统架构

```
用户提问
    │
    ▼
┌─────────────────────────────────┐
│       Master Agent（主控）        │
│   意图识别 → 路由 → 并行调度      │
└────┬──────────┬──────────┬──────┘
     │          │          │
     ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│辅导员  │ │心理辅导│ │就业指导│
│ Agent  │ │ Agent  │ │ Agent  │
└────┬───┘ └────┬───┘ └────┬───┘
     └──────────┴──────────┘
                │
     ┌──────────▼──────────┐
     │    RAG 检索流水线     │
     │                     │
     │  Query Rewrite       │  ← 历史感知，模糊问题补全
     │  ↓                  │
     │  Hybrid Search       │  ← FAISS + BM25 + RRF
     │  ↓                  │
     │  GraphRAG 增强       │  ← 知识图谱实体扩展
     │  ↓                  │
     │  BGE Reranker        │  ← 精排
     │  ↓                  │
     │  Context Compress    │  ← Token 压缩
     │  ↓                  │
     │  DeepSeek 生成       │  ← 流式输出
     └─────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 推荐使用 Conda 创建独立环境

### 1. 安装依赖

```bash
git clone https://github.com/yourname/UniAssist.git
cd UniAssist
pip install -r requirements.txt
pip install networkx   # GraphRAG 依赖
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx      # https://platform.deepseek.com
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxx   # https://cloud.siliconflow.cn （注册送2000万Token）
```

### 3. 准备知识库并启动

**辅导员助手**（政策文档 → `docs/` 目录）

```bash
python build_kb.py build --dir ./docs
python main.py chat
```

**心理辅导助手**（JSON 数据集 → `mental_docs/` 目录）

```bash
python build_mental_kb.py build --dir ./mental_docs
python mental_main.py chat
```

**就业指导助手**（Excel 放项目根目录）

```bash
python build_career_kb.py build --files 就业数据2025年.xlsx
python career_main.py chat
```

**综合主控 Agent**（需先完成以上三个建库）

```bash
python master_agent.py chat
```

> 对话中输入 `think` 切换深度思考（R1），`clear` 清记忆，`exit` 退出

---

## 💡 功能演示

### 跨领域复杂问题

```
你：我挂科了很焦虑，不知道还能不能毕业，以后怎么找工作

🧠 意图识别
  📚辅导员  ✓  置信度: 0.9  挂科政策/毕业条件
  💙心理    ✓  置信度: 0.8  焦虑情绪
  💼就业    ✓  置信度: 0.7  求职建议

→ 并行调用 3 个模块...
  ✅ 💙心理  2.1s
  ✅ 📚辅导员 1.8s  
  ✅ 💼就业  2.3s

🎯 综合回答：
  听到你说的，这段时间一定承受了很多压力...
  关于毕业条件：根据学校规定，单科挂科可申请补考...
  关于就业：即便有挂科记录，计算机类岗位更注重实际能力...
```

### 历史感知模糊问题

```
你：网络专业薪酬怎么样？
助手：据近三年数据，网络专业平均薪酬约 5500 元...

你：那升学呢？
# Query Rewrite 自动还原 → "网络专业升学情况，考研率和主要院校"
助手：网络专业升学率约 18%，主要考研院校有...
```

### 危机干预

```
你：[表达极度痛苦]
💙 暖心（自动跳过检索，直接响应）：
  我听到你说的了，谢谢你愿意告诉我...
  📞 全国心理援助热线：400-161-9995（24小时）
```

---

## 📦 模块介绍

### 核心检索层

| 文件 | 功能 |
|------|------|
| `hybrid_search.py` | FAISS 向量检索 + BM25 关键词检索 + RRF 融合，查询阶段批量 Embedding 单次 API 调用 |
| `reranker.py` | BGE-Reranker-V2-M3 精排（硅基流动），Cohere 格式 HTTP 直调 |
| `query_rewriter.py` | 历史感知查询改写，模糊问题自动补全，支持 DeepSeek / Anthropic / OpenAI |
| `context_compressor.py` | 规则裁剪冗余上下文，减少生成阶段 Token 消耗 |

### GraphRAG 层

| 文件 | 功能 |
|------|------|
| `graph_extractor.py` | DeepSeek 驱动的实体关系抽取，每个 chunk → 实体 + 关系 JSON |
| `graph_store.py` | NetworkX 有向图存储，支持 N 跳邻居遍历、模糊实体匹配、持久化 |
| `graph_retriever.py` | 问题实体识别 → 图谱扩展 → chunk 关联，与 Hybrid Search 做 RRF 融合 |
| `build_graph.py` | 知识图谱建库脚本，支持采样测试和统计查看 |

### 三大 Agent

| 文件 | 功能 |
|------|------|
| `pipeline.py` | 辅导员 RAG 流水线，GraphRAG 可选增强，支持流式 / 非流式 |
| `mental_pipeline.py` | 心理辅导流水线，含情绪识别（焦虑/抑郁/压力）和危机干预直通 |
| `career_pipeline.py` | 就业指导流水线，Excel 统计摘要检索，技能建议融合自身知识 |
| `master_agent.py` | 主控路由 Agent，意图识别 + 并行调度 + 汇总生成 |

### 建库脚本

| 文件 | 功能 |
|------|------|
| `build_kb.py` | 辅导员知识库建库，支持 Word 文档解析（图片自动提取并绑定 chunk）|
| `build_mental_kb.py` | 心理知识库，兼容两种 JSON 格式（对话数组 / JSONL）+ Word 混合 |
| `build_career_kb.py` | 就业知识库，Excel 统计分析生成语义摘要，覆盖薪酬/行业/公司/升学维度 |

### 评测体系

| 文件 | 功能 |
|------|------|
| `eval_dataset.py` | 测试集管理，支持版本控制、标签过滤、统计分析 |
| `eval_metrics.py` | LLM-as-Judge 自动评分，6 个专业指标（忠实度/相关性/共情/安全/数据准确性等）|
| `eval_autogen.py` | 基于知识库自动生成测试用例，一键生成 + 评测 |
| `eval_runner.py` | 自动化评测运行器，带版本标签，支持按系统/难度过滤 |
| `eval_annotator.py` | 人工标注 CLI，计算人机一致性 |
| `eval_compare.py` | Prompt / 版本对比，定位改进和退步用例 |
| `eval_report.py` | 错误聚类分析、回归测试、HTML 可视化报告 |

### 后端服务

| 文件 | 功能 |
|------|------|
| `api_server.py` | FastAPI + SSE 流式接口，支持四个系统、深度思考开关、CORS |

---

## 📊 评测指标

系统使用 **LLM-as-Judge** 自动评测，各系统指标不同：

| 指标 | 满分 | 适用系统 | 权重 |
|------|------|---------|------|
| `faithfulness` 忠实度 | 5 | 辅导员、就业 | 1.0× |
| `relevance` 相关性 | 5 | 全部 | 1.2× |
| `completeness` 完整性 | 5 | 辅导员、就业 | 1.0× |
| `empathy` 共情度 | 5 | 心理 | 1.5× |
| `safety` 安全性 | 5 | 心理 | **2.0×** |
| `data_accuracy` 数据准确性 | 5 | 就业 | 1.2× |

综合分 ≥ 3.0 为通过，安全性权重最高且一票否决。

---

## 📁 项目结构

```
UniAssist/
├── 📂 核心检索
│   ├── hybrid_search.py       # 混合检索
│   ├── reranker.py            # BGE 精排
│   ├── query_rewriter.py      # 历史感知查询改写
│   ├── context_compressor.py  # 上下文压缩
│   └── generator.py           # DeepSeek 流式生成（支持 R1）
│
├── 📂 GraphRAG
│   ├── graph_extractor.py     # 实体关系抽取
│   ├── graph_store.py         # 知识图谱存储
│   ├── graph_retriever.py     # 图谱增强检索
│   └── build_graph.py         # 图谱建库脚本
│
├── 📂 三大 Agent
│   ├── pipeline.py            # 辅导员流水线
│   ├── mental_pipeline.py     # 心理辅导流水线
│   ├── career_pipeline.py     # 就业指导流水线
│   └── master_agent.py        # 主控路由 Agent
│
├── 📂 建库脚本
│   ├── build_kb.py            # 辅导员知识库
│   ├── build_mental_kb.py     # 心理知识库
│   └── build_career_kb.py     # 就业知识库
│
├── 📂 入口
│   ├── main.py                # 辅导员助手
│   ├── mental_main.py         # 心理辅导助手
│   ├── career_main.py         # 就业指导助手
│   └── api_server.py          # FastAPI 后端（SSE）
│
├── 📂 评测
│   ├── eval_dataset.py        # 测试集管理
│   ├── eval_metrics.py        # 评测指标
│   ├── eval_autogen.py        # 自动生成用例
│   ├── eval_runner.py         # 自动化评测
│   ├── eval_annotator.py      # 人工标注
│   ├── eval_compare.py        # 版本对比
│   └── eval_report.py         # 错误分析 / 回归测试
│
├── 📂 配置
│   ├── rag_config.py          # 辅导员系统配置
│   ├── mental_config.py       # 心理系统配置
│   └── career_config.py       # 就业系统配置
│
├── requirements.txt
└── .env.example
```

---

## ⚙️ 配置说明

所有配置均在 `rag_config.py` 的 dataclass 中定义，无需修改代码，修改配置类字段即可：

```python
# 切换为深度思考模式（R1）
GeneratorConfig(
    api_provider="deepseek",
    deepseek_model="deepseek-chat",           # 普通对话
    deepseek_reasoner_model="deepseek-reasoner",  # 深度思考
)

# 调整检索参数
SearchConfig(
    top_k_vector=15,   # 向量检索 Top-K
    top_k_bm25=15,     # BM25 检索 Top-K
    top_k_merged=15,   # 合并后 Top-K
)

# 减少子查询数量（提升速度）
QueryRewriterConfig(
    num_sub_queries=1,  # 建议 1~3
)
```

---

## 🗺 未来规划

### 近期（v1.x）
- [ ] **多模态支持**：PDF 扫描件 OCR、图片内容理解（当前图片仅作插图绑定）
- [ ] **增量建库**：新增文档时无需全量重建，支持增量更新索引
- [ ] **用户画像**：记录用户专业、年级，个性化检索与回答
- [ ] **Web UI**：基于 React 的前端界面，支持思考过程折叠展示

### 中期（v2.x）
- [ ] **多校部署**：配置化多租户支持，不同学校独立知识库
- [ ] **主动推送**：重要政策变更时主动通知相关学生
- [ ] **联网搜索**：岗位技能类问题实时查询招聘网站补充知识
- [ ] **语音交互**：ASR 输入 + TTS 输出，支持手机端语音问答

### 远期（v3.x）
- [ ] **Agent 工具链**：集成校园系统 API（教务、图书馆、校历），从"回答"升级为"办事"
- [ ] **私有化大模型**：支持本地部署 Qwen / DeepSeek，零数据外流
- [ ] **知识图谱可视化**：提供图谱管理后台，支持人工编辑实体关系

---

## 🤝 贡献指南

欢迎 PR 和 Issue！提交前请确保：

1. 通过所有评测（`python eval_runner.py run --tag pre-merge`）
2. 回归测试无退步（`python eval_report.py regression --baseline main`）
3. 新功能附带测试用例（`python eval_autogen.py generate --system main --n 10`）

---

## 📄 License

MIT License — 欢迎学术引用和商业使用，保留署名即可。

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

Made with ❤️ for university students

</div>
