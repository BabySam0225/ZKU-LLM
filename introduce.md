
---

## 第三章 系统总体设计

### 3.1 系统架构设计

本系统采用**前后端分离的微服务架构**，自顶向下划分为四个核心层次，各层通过RESTful API或gRPC通信，支持独立部署与水平扩展。

1. **Web前端层**  
   - 基于 **Vue 3 / React 18** + TypeScript 构建单页应用（SPA）  
   - 状态管理：Pinia / Redux Toolkit  
   - UI组件库：Element Plus / Ant Design  
   - 网络请求：Axios + 请求取消机制（防止重复请求）  
   - 流式接收：EventSource（SSE）或 WebSocket API  

2. **后端服务层**（核心业务网关）  
   - 开发框架：**FastAPI**（异步支持）+ Nginx 反向代理  
   - 功能模块：  
     - 用户认证与鉴权（JWT）  
     - 请求路由与限流（令牌桶算法）  
     - **查询改写**：调用大模型（DeepSeek / 本地Qwen3）将口语化问题改写为 `main_query` + `sub_queries`  
     - **混合检索调度器**：并行调用向量检索（FAISS）与关键词检索（BM25），结果经RRF融合  
     - **重排序模块**：调用BGE Reranker API（硅基流动）或本地模型对候选文档精排  
     - 多轮对话状态管理（Redis会话存储）  
     - 大模型调用与流式响应封装  

3. **知识库检索层**  
   - 向量检索：**FAISS**（IndexFlatIP索引，内积相似度）  
   - 关键词检索：**BM25**（`rank_bm25.BM25Okapi`，中文使用jieba分词）  
   - 融合算法：**RRF（Reciprocal Rank Fusion）**，参数 k=60  
   - 数据存储：文档元数据及原始内容以JSON/Pickle形式本地持久化  

4. **模型生成层**  
   - 大语言模型：支持API调用（DeepSeek-V3 / GPT-4o）  
   - 推理加速：vLLM / TensorRT-LLM  
   - 流式输出：异步生成器 + SSE 推送  

**系统工作流（含数据流与异步边界）**：

```text
[用户] → 前端 → API Gateway → 查询改写 → 混合检索(FAISS + BM25) → RRF融合 → 重排序(BGE Reranker) → 上下文融合 → LLM生成(流式) → 前端逐字渲染
                                                                                              ↑
                                                                                       Redis会话存储
```

**关键设计指标**：
- 检索延迟 ≤ 500ms（P95）  
- 首字生成延迟 ≤ 6.2s  
- 支持并发请求 ≥ 100 QPS（后端水平扩展）  

### 3.2 技术路线

本系统以**检索增强生成（RAG）**为核心技术框架，并针对高校就业咨询场景进行专项优化。技术栈全景如下：

| 技术方向 | 具体实现 | 选型理由 |
|---------|----------|----------|
| 前端框架 | Vue 3 + Vite | 高性能、生态完善、开发效率高 |
| 后端框架 | FastAPI | 异步支持、自动API文档、类型提示 |
| 向量检索 | FAISS (IndexFlatIP) | 高维内积检索，支持GPU，与归一化嵌入向量完美配合 |
| 关键词检索 | BM25Okapi (rank_bm25) | 经典概率模型，中文jieba分词，无需外部API |
| 嵌入模型 | BAAI/bge-m3 (硅基流动API) | 多语言（中英日韩），1024维，语义表征SOTA |
| 重排模型 | BAAI/bge-reranker-v2-m3 (硅基流动API) | CrossEncoder架构，精准相关性评分 |
| 查询改写模型 | DeepSeek-V3 (API) / Qwen3-7B (本地) | 中文理解强，支持JSON模式输出，成本低 |
| 大语言模型 | Qwen3-7B-Instruct (本地部署) | 中文理解强、支持32K上下文、可商用 |
| 对话状态管理 | Redis + 自定义会话窗口 | 高并发、低延迟、支持TTL |
| 流式通信 | SSE（Server-Sent Events） | 简单、自动重连、HTTP兼容 |

**核心创新技术**：
- **混合检索 + RRF融合**：避免单一检索模式缺陷，鲁棒性优于加权求和  
- **多查询改写（main_query + sub_queries）**：生成多个语义不同的检索入口，提升召回率  
- **三级级联检索**：向量检索（粗排）→ BM25（补充）→ RRF融合 → BGE重排（精排）  
- **可解释性反馈**：每次回答附带引用的知识库源文档（文件路径、内容片段）  

---

## 第四章 系统详细设计

### 4.1 私有知识库构建

**知识来源**：
- 高校政策文件（PDF/Word）：奖学金评定、学分转换、毕业要求  
- 就业指导资料（网页/Excel）：招聘信息、面试技巧、签约流程  
- 常见问答对（结构化CSV）：历年学生咨询记录脱敏后整理  

**预处理流程**：

1. **文档解析**：  
   - PDF → PyMuPDF / PDFPlumber 提取文本及表格  
   - Word → python-docx  
   - HTML → BeautifulSoup  

2. **清洗与归一化**：  
   - 去除页眉页脚、特殊符号、多余空白  
   - 统一日期、数字格式  
   - 敏感信息脱敏（如身份证号、手机号）  

3. **文档切分（Chunking）**：  
   - 策略：基于语义边界（段落、标题、句子） + 滑动窗口重叠（overlap=64字符）  
   - 块大小：256/512 tokens（针对不同嵌入模型可配置）  
   - 保留元数据：来源文件、章节标题、时间戳  

4. **向量化**：  
   - 嵌入模型：**BAAI/bge-m3**（通过硅基流动API调用），输出1024维向量  
   - 批量编码：batch_size=64，自动限速（每批间隔≥2秒，避免触发TPM限制）  
   - 向量归一化：L2归一化后存入FAISS，检索时使用内积等价于余弦相似度  

5. **关键词索引**：  
   - 对每个文档块的分词结果（jieba分词）构建 **BM25Okapi** 索引  
   - 索引序列化：使用 `pickle` 保存 `(bm25对象, doc_ids列表)` 至本地文件  

**增量更新机制**：  
- 每日定时任务检测知识库文件变动  
- 新文档经过相同流程后：重新生成向量矩阵并重建FAISS索引，同时更新BM25索引  
- 删除文档通过元数据标记实现软删除，重建索引时过滤  

### 4.2 混合检索机制

系统并行执行以下两种检索通道，并在检索层实现RRF融合。代码实现位于 `hybrid_search.py`。

**1. 向量检索（语义召回）**  
- 嵌入后端：默认使用 **BGE API**（硅基流动），备选OpenAI或本地FlagModel  
- 将用户查询使用同一嵌入模型编码为查询向量 `q`，并进行L2归一化  
- 检索器：**FAISS IndexFlatIP**（内积索引）  
- 检索参数：`top_k=20`（可配置）  
- 返回结果列表 `vec_results = [(doc_id, score_vec), ...]`，其中 `score_vec` 为内积相似度（等价于余弦相似度）  

**2. 关键词检索（精确匹配）**  
- 模型：**BM25Okapi**（来自 `rank_bm25` 库）  
- 分词器：中文使用 `jieba.cut`，英文按空格分词（`text.lower().split()`）  
- 检索参数：`top_k=20`  
- 返回结果列表 `bm25_results = [(doc_id, score_bm25), ...]`，其中 `score_bm25` 为BM25原始得分  

**3. 结果融合（RRF算法）**  
- 采用**倒数排序融合（Reciprocal Rank Fusion）**，避免参数调优且鲁棒性好  
  \[
  score_{RRF}(d) = \sum_{s \in \{vec, bm25\}} \frac{1}{k + rank_s(d)}
  \]
  其中 **k=60**（经验值，稳定融合效果），`rank_s(d)` 是文档d在检索结果s中的排名（从0开始，代码中加1）  
- 融合后按 `score_RRF` 降序取前 `top_k_merged=20` 条作为候选文档集合  

**实现细节**：  
- 向量检索器与BM25检索器分别实现 `build_index`、`search`、`save`、`load` 方法  
- `HybridRetriever` 类统一管理两者，`_rrf_merge` 私有方法执行融合  
- 支持多查询检索：`multi_query_search(queries: List[str])` 合并去重  

### 4.3 结果重排

为提高生成阶段的输入质量，候选集需要进一步精准排序。代码实现位于 `reranker.py`。

**重排模型**：默认使用 **BAAI/bge-reranker-v2-m3**（通过硅基流动API调用），备选Cohere或本地FlagReranker。  
- 类型：CrossEncoder架构，输入 (query, document) 对，输出相关性分数（0~1）  
- 优点：比双编码器更精确，能捕捉query与doc的深层交互  

**执行流程**：  
1. 取混合检索融合后的候选文档（通常为20条）  
2. 构造请求体（与Cohere Rerank API格式完全兼容）：  
   ```json
   {
     "model": "BAAI/bge-reranker-v2-m3",
     "query": "用户原始查询（或改写后的main_query）",
     "documents": [doc.content for doc in docs],
     "top_n": 5,
     "return_documents": false
   }
   ```  
3. 发送POST请求至硅基流动   
4. 解析返回的 `results` 列表（已按 `relevance_score` 降序排列），截取 `top_k=5` 作为最终生成输入  

**优化策略**：  
- 若候选文档数 ≤10，跳过重排直接进入生成（节省延迟）  
- 支持本地模型（`FlagReranker`）作为离线备选，避免API依赖  

**效果指标**：  
- 重排后第一位的命中率（Recall@1）相比原始检索提升约15%~20%（在自建数据集上验证）  

### 4.4 查询意图转换（查询改写）

用户问题口语化、省略上下文现象普遍，直接检索效果不佳。本系统实现**查询改写**而非传统意图分类，代码位于 `query_rewriter.py`。

**核心设计**：使用大语言模型将原始问题改写为 **`main_query` + `sub_queries`** 的结构化形式。  
- `main_query`：清晰、结构化，补充缺失上下文  
- `sub_queries`：3个语义相关但表达方式不同的变体（换词、换角度、换问法）  

**Prompt模板**（`REWRITE_PROMPT`）：  
```text
你是一个专业的问题改写助手，请将用户问题改写为更适合检索的形式。
要求：
1. main_query：将问题改写得更清晰、结构化，补充可能缺失的上下文
2. sub_queries：生成 {num} 个语义相关但表达方式不同的查询（换词、换角度、换问法）
用户问题：{question}
请严格输出JSON，不要输出任何其他内容：
{"main_query": "改写后的主查询", "sub_queries": ["子查询1", "子查询2", "子查询3"]}
```

**后端支持**：  
- **默认**：DeepSeek API（`deepseek-chat`），中文理解强、成本低、支持JSON模式  
- 备选：Anthropic Claude Haiku、OpenAI GPT-4o-mini、本地Qwen3-7B  

**实现细节**：  
- 统一接口 `QueryRewriter.rewrite(question: str) -> RewriteResult`  
- 解析LLM输出时使用正则提取JSON块，失败时降级为原始查询  
- 最终生成 `all_queries() = [main_query] + sub_queries` 用于多查询检索  

**示例**：  
- 原问题：“我学分不够还能毕业吗？”  
- 改写结果：  
  ```json
  {
    "main_query": "毕业学分要求 学分不足 处理办法",
    "sub_queries": ["学分不够能否毕业", "毕业条件 学分规定", "未修满学分毕业政策"]
  }
  ```  
改写后的多查询被送入 `HybridRetriever.multi_query_search()`，显著提升召回率。

### 4.5 多轮对话上下文管理


**上下文处理策略**：  
1. 当用户发起新问题时，从Redis拉取最近k轮对话（初始k=3，可配置）  
2. 将历史对话拼接为LLM的system prompt的一部分：  
   ```
   以下是用户与助手的对话历史：
   Q:  ...
   A:  ...
   当前问题：...
   ```  
3. 对于超长上下文（超过模型最大长度），采用**滑动窗口**（保留最近n轮）或**摘要压缩**（调用LLM总结历史）  

**实体继承**：  
- 从对话历史中使用规则+NER提取关键实体（如“国家奖学金”、“计算机学院”），存入entities  
- 当前问题若包含指代词（“它”、“这个”），用最近实体替换后再送入查询改写模块  

**对话结束检测**：  
- 若用户发送clear指令，清除会话状态；同时设置TTL=30分钟自动过期）  

### 4.6 流式响应机制

为提升用户体验，避免长时间白屏等待，全链路支持流式输出。

**技术实现**：  
- 后端：FastAPI 使用 `StreamingResponse` 结合异步生成器  
- LLM推理：启用vLLM的流式模式，逐个token产出  
- 协议：SSE（`text/event-stream`），数据格式：  
  ```
  event: message
  data: {"token": "我", "finished": false}
  ```  
- 前端：使用 `EventSource` 或 `fetch` + `ReadableStream` 逐token渲染  

**优化措施**：  
- **首字延迟优化**：LLM推理使用前缀缓存（Prompt Cache），避免重复计算历史对话的KV Cache  
- **令牌级节流**：避免每个token都触发UI重绘，使用requestAnimationFrame批量更新  
- **超时与重连**：SSE自动重连机制，最大重试3次  

**性能指标**：  
- 平均首字时间（TTFT）：≤ 6.2s（DeepSeek-V3API）  
- 平均生成速率：≥ 100 tokens/s（本地GPU4060）  
- 用户无感知延迟（流畅度评分 ≥ 4.5/5.0）  

