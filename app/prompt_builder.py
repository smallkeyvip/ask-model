"""
Prompt 工程：企业角色 + 知识库约束 + 历史对话 的结构化 Prompt 构建
"""
from __future__ import annotations

from typing import List


SYSTEM_PROMPT = """你是企业智能知识助手，专注于为企业提供准确、专业的知识问答服务。

## 核心原则
1. **基于知识库回答**：你的回答必须基于提供的知识库内容，不要编造不存在的信息
2. **标注引用来源**：在回答中引用知识库内容时，使用 [来源X] 标注
3. **承认未知**：如果知识库中没有相关信息，明确告知用户"根据当前知识库未找到相关信息"
4. **结构化回答**：使用清晰的层级结构，善用列表、表格等格式
5. **专业严谨**：保持企业级的专业表述，避免口语化和随意表述

## 回答格式
- 先给出直接答案/结论
- 再展开解释（如有需要）
- 最后列出引用来源
"""


class PromptBuilder:
    """
    构建结构化 Prompt，融合：
    - 企业角色 system prompt
    - 检索到的知识库上下文
    - 三层记忆（短期对话 + 中期摘要 + 长期记忆）
    - 当前用户问题
    """

    def build_messages(
        self,
        query: str,
        knowledge_docs: List[dict],
        memory_context: dict,
    ) -> List[dict]:
        """
        构建完整的 messages 列表供 LLM 调用。

        参数:
            query: 用户当前问题
            knowledge_docs: 检索+重排后的知识文档
            memory_context: 三层记忆上下文
                - short_term_messages: 最近对话
                - mid_term_summary: 滚动摘要
                - long_term_memories: 长期记忆召回
        """
        messages = []

        # 1. System Prompt（企业角色）
        system_content = self._build_system_with_context(
            knowledge_docs,
            memory_context,
        )
        messages.append({"role": "system", "content": system_content})

        # 2. 中期摘要记忆（作为较早的上下文）
        mid_summary = memory_context.get("mid_term_summary", "")
        if mid_summary:
            messages.append({
                "role": "system",
                "content": f"以下是之前对话的摘要记录：\n{mid_summary}",
            })

        # 3. 长期记忆（跨会话相关历史）
        long_term = memory_context.get("long_term_memories", [])
        if long_term:
            lt_text = "\n".join(
                f"- {m['content']}" for m in long_term[:3]
            )
            messages.append({
                "role": "system",
                "content": f"以下是与当前问题相关的历史记忆：\n{lt_text}",
            })

        # 4. 短期对话历史（最近 N 轮）
        short_term_msgs = memory_context.get("short_term_messages", [])
        messages.extend(short_term_msgs)

        # 5. 当前用户问题
        messages.append({"role": "user", "content": query})

        return messages

    def _build_system_with_context(
        self,
        knowledge_docs: List[dict],
        memory_context: dict,
    ) -> str:
        """将 system prompt 与知识库上下文融合"""
        parts = [SYSTEM_PROMPT]

        # 注入知识库上下文
        if knowledge_docs:
            parts.append("\n## 知识库参考内容")
            for i, doc in enumerate(knowledge_docs, 1):
                source = doc.get("source", "知识库")
                parts.append(f"\n### [来源{i}] {source}")
                parts.append(doc["content"])
            parts.append(f"\n共参考 {len(knowledge_docs)} 条知识库内容。")
        else:
            parts.append("\n## 注意：本次未检索到相关知识库内容，请基于自身知识谨慎回答，并明确告知用户。")

        return "\n".join(parts)
