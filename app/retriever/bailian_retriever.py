"""
阿里百炼远程知识库检索器
"""
from __future__ import annotations

import json
import logging
import asyncio
from typing import List

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings


class BailianRetriever:
    """
    调用阿里百炼知识库 API 进行远程检索。
    通过 SSE 接口聚合知识库检索结果。
    """

    def __init__(self):
        """从全局配置创建百炼端点凭据和可复用异步 HTTP 客户端。

        客户端默认超时 30 秒并信任环境代理设置；原始 API 密钥只用于请求头，
        不会写入检索结果。
        """
        s = settings()
        self.api_url = s.BAILIAN_API_URL
        self.api_key = s.BAILIAN_API_KEY
        self.agent_id = s.BAILIAN_AGENT_ID
        self._client = httpx.AsyncClient(timeout=30, trust_env=True)

    async def aclose(self) -> None:
        """关闭百炼 HTTP 连接池；已经关闭时不执行操作。"""
        if not self._client.is_closed:
            await self._client.aclose()

    async def retrieve(self, query: str) -> List[dict]:
        """通过百炼强制流式接口聚合一次知识库检索结果。

        Args:
            query: 发送给百炼应用的单条检索问题。

        Returns:
            从全部 SSE 事件解析出的统一文档列表。服务端错误或网络异常会在
            250 毫秒后重试一次；第二次仍失败、非成功客户端状态或无可解析事件时
            返回空列表，不向混合检索器抛出错误。

        Notes:
            格式错误的单个 SSE ``data`` 行会被忽略；HTTP 5xx 会触发重试，
            其他非 200 状态记录响应正文并直接降级。
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

        for attempt in range(2):
            try:
                async with self._client.stream(
                    "POST",
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        if resp.status_code >= 500:
                            raise httpx.HTTPStatusError(
                                f"Bailian server error: {resp.status_code}",
                                request=resp.request,
                                response=resp,
                            )
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
                if attempt == 0:
                    await asyncio.sleep(0.25)
                    continue
                logger.error(f"[百炼检索] 请求失败: {e}")
        return []

    @staticmethod
    def _parse_response(events: list) -> List[dict]:
        """从已解码的百炼 SSE 事件中提取工具角色知识文档。

        Args:
            events: 每个 SSE ``data`` 行 JSON 解码后的字典列表。

        Returns:
            统一检索结果列表。工具消息含 ``docs`` 时逐项保留正文、来源、分数和
            链接；没有 ``docs`` 时把工具消息正文作为分数 0.7 的兜底文档。
            无效分数使用 0.8，非工具角色被忽略。
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
                    extra = msg.get("additional_kwargs", {}).get("extra_json", {})
                    docs = extra.get("docs", []) if isinstance(extra, dict) else []
                    for doc in docs:
                        try:
                            score = float(doc.get("_score", 0.8))
                        except (TypeError, ValueError):
                            score = 0.8
                        results.append({
                            "content": doc.get("content", ""),
                            "source": doc.get("doc_name", doc.get("title", "百炼知识库")),
                            "score": score,
                            "origin": "bailian",
                            "doc_url": doc.get("doc_url", ""),
                        })
                    if not docs:
                        results.append({
                            "content": content,
                            "source": "百炼知识库",
                            "score": 0.7,
                            "origin": "bailian",
                        })

        return results
