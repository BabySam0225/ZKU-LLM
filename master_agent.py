"""
master_agent.py — 主控 Agent（ 路由 + 编排）

=============================================

负责：

1. 意图识别： 判断问题涉及哪些领域（ 辅导员/⼼理/就业）

2. 路由：  单领域直接转发，  多领域并⾏调⽤

3. 汇总： 把多个 Agent 的回答整合成⼀个连贯的回复

4. 多轮记忆： 跨 Agent 的对话历史统⼀维护

典型复杂问题示例：

"我挂科了很焦虑，  不知道还能不能毕业，  也不知道以后怎么找⼯作"

→ 涉及： 辅导员（ 挂科政策） + ⼼理（ 焦虑情绪） + 就业（ 求职建议）

→ 三个 Agent 并⾏回答 → 汇总成⼀个有温度的完整回复

⽤法：

python master_agent.py chat

python master_agent.py query --q "我挂科了很焦虑..."
"""

import os
import time
import httpx
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

load_dotenv()
console = Console()

# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────
@dataclass
class AgentResponse:
    agent: str  # main / mental / career
    answer: str
    success: bool = True
    error: str = ""
    latency: float = 0.0

@dataclass
class MasterResponse:
    question: str
    agents_called: List[str]
    agent_responses: List[AgentResponse]
    final_answer: str
    intent: dict = field(default_factory=dict)

# ─────────────────────────────────────────────
# 意图识别
# ─────────────────────────────────────────────
INTENT_PROMPT = """你是⼀个智能分诊助⼿， 判断⼤学⽣的问题涉及哪些领域。

【 三个领域】
- main： 学校政策、 规章制度、 奖学⾦、 转专业、 请假、 成绩、 毕业要求等（ 辅导员知识库）
- mental： 情绪困扰、 ⼼理压⼒、 焦虑抑郁、 ⼈际关系、 ⾃我认知等（ ⼼理辅导）
- career： 就业⽅向、 薪酬、 推荐公司、 考研升学、 岗位技能、 求职建议等（ 就业指导）

【 ⽤户问题】
{question}

【 历史对话关键词】
{history_summary}

要求：
1. 分析问题真正需要哪些领域的知识来回答
2. ⼀个问题可以同时涉及多个领域
3. confidence 表示该领域的相关程度（ 0.0-1.0）
4. 只有 confidence >= 0.5 的领域才需要调⽤

只返回 JSON：
{
"main":{"needed": true/false, "confidence": 0.0-1.0, "reason": "简短说明"},
"mental": {"needed": true/false, "confidence": 0.0-1.0, "reason": "简短说明"},
"career": {"needed": true/false, "confidence": 0.0-1.0, "reason": "简短说明"},
"complexity": "simple/moderate/complex"
}"""

class IntentRouter:
    """⽤ DeepSeek 识别⽤户意图， 决定调⽤哪些 Agent"""
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(verify=False),
        )

    def route(self, question: str, history: List[dict] = None) -> dict:
        # 从历史中提取关键词， 帮助意图识别
        history_summary = ""
        if history:
            recent = history[-4:]  # 最近2轮
            history_summary = " | ".join(
                m["content"][:30] for m in recent if m["role"] == "user"
            )

        prompt = INTENT_PROMPT.format(
            question=question,
            history_summary=history_summary or "⽆"
        )

        import json
        import re as _re
        default = {
            "main":{"needed": True, "confidence": 0.8, "reason": "默认"},
            "mental": {"needed": False, "confidence": 0.0, "reason": ""},
            "career": {"needed": False, "confidence": 0.0, "reason": ""},
            "complexity": "simple",
        }

        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""

            # 清理 markdown 代码块和⾸尾空⽩
            text = _re.sub(r"```(?:json)?\s*", "", text).strip()

            # 找到第⼀个 { 的位置， 去掉前⾯多余内容（ 包括换⾏）
            brace_idx = text.find("{")
            if brace_idx > 0:
                text = text[brace_idx:]

            # 贪婪提取最外层 JSON
            m = _re.search(r"\{.*\}", text, _re.DOTALL)
            if m:
                text = m.group()

            result = json.loads(text)
            return result
        except Exception as e:
            console.print(f"[dim]⚠ 意图识别失败， 默认使⽤ main: {e}[/dim]")
            return default

# ─────────────────────────────────────────────
# 汇总⽣成器
# ─────────────────────────────────────────────
SYNTHESIS_PROMPT = """你是⼀个综合性⼤学⽣助⼿， 现在需要把多个专业模块的回答整合成⼀个连贯、温暖的最终回复。

【 ⽤户问题】
{question}

【 各模块回答】
{agent_answers}

整合要求：
1. 先照顾情绪（ 如果有⼼理模块的回答， 把共情部分放在最前⾯）
2. 再解答政策/信息类问题（ 辅导员模块内容）
3. 最后给出前进⽅向（ 就业/升学建议）
4. 回答要流畅⾃然， 不要显得是"拼接"的
5. 不要重复相同内容， 提炼每个模块的核⼼信息
6. 语⽓温暖， 像⼀个了解学⽣全⾯情况的好⽼师"""

class SynthesisGenerator:
    """把多个 Agent 的回答汇总成⼀个连贯的回复"""
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(verify=False),
        )

    def synthesize(self, question: str, responses: List[AgentResponse], stream: bool = True, enable_thinking: bool = False) -> str:
        # 构建各模块回答的汇总
        agent_names = {
            "main": "辅导员助⼿",
            "mental": "⼼理辅导",
            "career": "就业指导"
        }

        agent_answers = "\n\n".join(
            f"【 {agent_names.get(r.agent, r.agent)}】 \n{r.answer}"
            for r in responses if r.success and r.answer
        )

        model = "deepseek-reasoner" if enable_thinking else "deepseek-chat"
        messages = [{"role": "user", "content": SYNTHESIS_PROMPT.format(
            question=question,
            agent_answers=agent_answers,
        )}]

        if stream:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1500,
                temperature=0.3,
                stream=True,
            )
            full = ""
            in_thinking = False
            print("\n综合回答： ", end="", flush=True)
            for chunk in response:
                delta = chunk.choices[0].delta
                thinking = getattr(delta, "reasoning_content", None) or ""
                content_text = delta.content or ""
                if thinking:
                    if not in_thinking:
                        print("\n\033[2m[深度思考] ", end="", flush=True)
                        in_thinking = True
                    print(thinking, end="", flush=True)
                if content_text:
                    if in_thinking:
                        print("\033[0m\n综合回答： ", end="", flush=True)
                        in_thinking = False
                    print(content_text, end="", flush=True)
                    full += content_text
            print()
            return full
        else:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1500,
                temperature=0.3,
            )
            return resp.choices[0].message.content

# ─────────────────────────────────────────────
# 各 Agent 调⽤封装
# ─────────────────────────────────────────────
def call_agent(agent_name: str, question: str, history: List[dict], enable_thinking: bool) -> AgentResponse:
    """调⽤单个 Agent， 返回结构化结果"""
    t0 = time.time()
    try:
        if agent_name == "main":
            from pipeline import RAGPipeline
            from rag_config import DEFAULT_CONFIG
            p = RAGPipeline(DEFAULT_CONFIG)
            p.load_index()
            result = p.query(question, history=history, enable_thinking=enable_thinking)
            answer = result.answer if hasattr(result, 'answer') else str(result)

        elif agent_name == "mental":
            from mental_pipeline import MentalPipeline
            p = MentalPipeline()
            p.load_index()
            answer = p.query(question, history=history, enable_thinking=enable_thinking)

        elif agent_name == "career":
            from career_pipeline import CareerPipeline
            p = CareerPipeline()
            p.load_index()
            answer = p.query(question, history=history, enable_thinking=enable_thinking)

        else:
            return AgentResponse(agent_name, "", False, f"未知 Agent: {agent_name}")

        return AgentResponse(
            agent=agent_name,
            answer=answer,
            success=True,
            latency=time.time() - t0,
        )
    except Exception as e:
        return AgentResponse(
            agent=agent_name,
            answer="",
            success=False,
            error=str(e),
            latency=time.time() - t0,
        )

# ─────────────────────────────────────────────
# 主控 Pipeline
# ─────────────────────────────────────────────
class MasterAgent:
    def __init__(self):
        self.router = IntentRouter()
        self.synthesizer = SynthesisGenerator()
        # 缓存已加载的 pipeline， 避免重复加载
        self._pipelines: Dict = {}

    def query(self, question: str, history: List[dict] = None, enable_thinking: bool = False) -> MasterResponse:
        history = history or []
        t0 = time.time()

        # ── Step 1: 意图识别 ──────────────────
        console.print(f"\n[bold cyan]意图识别[/bold cyan]： {question[:50]}...")
        intent = self.router.route(question, history)

        needed_agents = [
            name for name in ["main", "mental", "career"]
            if intent.get(name, {}).get("needed") and
               intent.get(name, {}).get("confidence", 0) >= 0.5
        ]

        if not needed_agents:
            needed_agents = ["main"]  # 兜底

        # 打印意图识别结果
        t = Table(show_header=False, box=None, padding=(0, 1))
        agent_labels = {
            "main": "辅导员",
            "mental": "⼼理",
            "career": "就业"
        }
        complexity = intent.get("complexity", "simple")

        for name in ["main", "mental", "career"]:
            info = intent.get(name, {})
            needed = name in needed_agents
            conf = info.get("confidence", 0)
            reason = info.get("reason", "")
            status = f"[green]✓ 调⽤[/green]" if needed else "[dim]跳过[/dim]"
            t.add_row(
                agent_labels[name],
                status,
                f"置信度:{conf:.1f}",
                f"[dim]{reason}[/dim]"
            )
        console.print(t)
        console.print(f" [dim]复杂度： {complexity}  调⽤： {', '.join(needed_agents)}[/dim]")

        # ── Step 2: 并⾏调⽤各 Agent ──────────
        agent_responses = []

        if len(needed_agents) == 1:
            # 单 Agent： 直接调⽤， 保留流式输出
            agent = needed_agents[0]
            console.print(f"\n[bold cyan]→ 转发给 {agent_labels[agent]}[/bold cyan]")
            resp = call_agent(agent, question, history, enable_thinking)
            agent_responses.append(resp)
            final_answer = resp.answer if resp.success else "抱歉， 处理时遇到了问题。"
        else:
            # 多 Agent： 并⾏调⽤（ ⾮流式） ， 最后统⼀汇总
            console.print(f"\n[bold cyan]→ 并⾏调⽤ {len(needed_agents)} 个模块...[/bold cyan]")
            with ThreadPoolExecutor(max_workers=len(needed_agents)) as executor:
                futures = {
                    executor.submit(
                        call_agent, agent, question, history, False
                    ): agent
                    for agent in needed_agents
                }

                for future in as_completed(futures):
                    resp = future.result()
                    agent_responses.append(resp)
                    status = "✅" if resp.success else "❌"
                    console.print(
                        f" {status} {agent_labels[resp.agent]} "
                        f"({resp.latency:.1f}s)"
                        + (f" [red]{resp.error[:40]}[/red]" if resp.error else "")
                    )

            # 按固定顺序排（ mental 优先， 影响汇总语⽓）
            order = {"mental": 0, "main": 1, "career": 2}
            agent_responses.sort(key=lambda r: order.get(r.agent, 9))

        # ── Step 3: 汇总 ────────────────────
        console.print(f"\n[bold cyan]→ 汇总回答...[/bold cyan]")
        final_answer = self.synthesizer.synthesize(
            question,
            agent_responses,
            stream=True,
            enable_thinking=enable_thinking,
        )

        elapsed = time.time() - t0
        console.print(f"\n[dim]总耗时 {elapsed:.1f}s[/dim]")

        return MasterResponse(
            question=question,
            agents_called=needed_agents,
            agent_responses=agent_responses,
            final_answer=final_answer,
            intent=intent,
        )

    # ─────────────────────────────────────────
    # 对话主循环
    # ─────────────────────────────────────────
    def chat(self):
        console.print(Panel(
            "[bold green]综合助⼿已就绪[/bold green]\n\n"
            "我整合了三个专业模块：\n"
            "辅导员助⼿（校规政策）\n"
            "⼼理辅导（情绪⽀持）\n"
            "就业指导（职业规划）\n\n"
            "⽆论你的问题涉及哪个⽅⾯， 都可以直接问我。\n"
            "输⼊ clear 清除记忆， exit 退出， think 切换深度思考模式",
            border_style="green"
        ))

        history = []
        enable_thinking = False

        while True:
            try:
                user_input = input("\n你： ").strip()
                if not user_input:
                    continue

                if user_input.lower() in {"exit", "quit", "q"}:
                    console.print("\n[green]再⻅！ 有问题随时来找我[/green]")
                    break

                if user_input.lower() == "clear":
                    history.clear()
                    console.print("[cyan]✓ 对话记忆已清除[/cyan]")
                    continue

                if user_input.lower() == "think":
                    enable_thinking = not enable_thinking
                    status = "开启" if enable_thinking else "关闭"
                    console.print(f"[cyan]✓ 深度思考模式已{status}[/cyan]")
                    continue

                result = self.query(
                    user_input,
                    history=history,
                    enable_thinking=enable_thinking,
                )

                # 存⼊历史
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": result.final_answer})

                # 只保留最近8条（ 4轮） ， 避免 token 积累
                if len(history) > 8:
                    history = history[-8:]

                round_num = len(history) // 2
                thinking_tag = " [深度思考]" if enable_thinking else ""
                console.print(
                    f"[dim]（ {round_num} 轮对话 | "
                    f"调⽤模块： {', '.join(result.agents_called)}"
                    f"{thinking_tag}）[/dim]"
                )

            except KeyboardInterrupt:
                console.print("\n[green]再⻅！[/green]")
                break
            except Exception as e:
                console.print(f"[red]出错了: {e}[/red]")

# ─────────────────────────────────────────────
# ⼊⼝
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="综合助⼿（ 三 Agent 编排） ")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("chat", help="启动对话")

    q = sub.add_parser("query", help="单次提问")
    q.add_argument("--q", required=True)
    q.add_argument("--think", action="store_true", help="开启深度思考")

    args = parser.parse_args()

    if args.cmd is None:
        parser.print_help()
    elif args.cmd == "chat":
        MasterAgent().chat()
    elif args.cmd == "query":
        result = MasterAgent().query(args.q, enable_thinking=args.think)
        console.print(Panel(result.final_answer, title="综合回答", border_style="green"))