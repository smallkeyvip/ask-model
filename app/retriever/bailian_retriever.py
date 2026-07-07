"""
阿里百炼远程知识库检索器
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, List

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings


class BailianRetriever:
    """
    调用阿里百炼知识库 API 进行远程检索。
    支持普通响应和 SSE 流式响应两种模式。
    """

    def __init__(self):
        s = settings()
        self.api_url = s.BAILIAN_API_URL
        self.api_key = s.BAILIAN_API_KEY
        self.agent_id = s.BAILIAN_AGENT_ID

    async def retrieve(self, query: str) -> List[dict]:
        """
        调用百炼知识库，返回检索结果。
        百炼 API 强制要求 stream=True，这里用流式接收后聚合。
        """
        payload = {
            "input": {
                "messages": [{"role": "user", "content": query}]
            },
            "parameters": {
                "agent_options": {"agent_id": self.agent_id}
            },
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=30, trust_env=True) as client:
                async with client.stream(
                    "POST",
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        logger.error(
                            f"[百炼检索] HTTP {resp.status_code}: {body.decode()[:500]}"
                        )
                        return []
                    # 收集 SSE 事件中的 data 行
                    all_events = []
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            chunk = line[5:].strip()
                            if chunk == "[DONE]":
                                break
                            try:
                                event_data = json.loads(chunk)
                                all_events.append(event_data)
                            except Exception:
                                pass

                    # 从所有事件中提取包含知识库结果的事件
                    return self._parse_response(all_events)
        except Exception as e:
            logger.error(f"[百炼检索] 请求失败: {e}")
            return []

    async def retrieve_stream(self, query: str) -> AsyncGenerator[str, None]:
        """流式调用百炼知识库，逐步返回内容"""
        payload = {
            "input": {
                "messages": [{"role": "user", "content": query}]
            },
            "parameters": {
                "agent_options": {"agent_id": self.agent_id}
            },
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=60, trust_env=True) as client:
                async with client.stream(
                    "POST",
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            chunk = line[5:].strip()
                            if chunk == "[DONE]":
                                break
                            yield chunk
        except Exception as e:
            logger.error(f"[百炼检索-流式] 请求失败: {e}")

    @staticmethod
    def _parse_response(events: list) -> List[dict]:
        """
        解析百炼 API SSE 事件流，提取知识库检索结果。
        events: 所有 SSE data 行解析后的 JSON 列表
        """
        results = []

        for event_data in events:
            output = event_data.get("output", {})
            choices = output.get("choices", [])
            for choice in choices:
                msg = choice.get("message", {})
                role = msg.get("role", "")
                content = msg.get("content", "")

                # 提取 tool 角色的知识库检索结果
                if role == "tool" and content:
                    results.append({
                        "content": content,
                        "source": "百炼知识库",
                        "score": 1.0,
                        "origin": "bailian",
                    })

                    # 提取详细文档引用
                    extra = msg.get("additional_kwargs", {}).get("extra_json", {})
                    docs = extra.get("docs", [])
                    for doc in docs:
                        results.append({
                            "content": doc.get("content", ""),
                            "source": doc.get("doc_name", doc.get("title", "百炼知识库")),
                            "score": float(doc.get("_score", 0.8)),
                            "origin": "bailian",
                            "doc_url": doc.get("doc_url", ""),
                        })

                # 提取 assistant 角色的最终回答
                elif role == "assistant" and content and content.strip() != "用户":
                    results.append({
                        "content": content,
                        "source": "百炼知识库",
                        "score": 1.0,
                        "origin": "bailian",
                    })

        return results
