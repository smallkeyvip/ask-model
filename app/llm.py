"""
LLM 调用模块：对接 DeepSeek Pro（兼容 OpenAI API 格式）
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, List, Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings


class LLMClient:
    """
    DeepSeek Pro LLM 客户端，支持：
    - 普通响应（非流式）
    - SSE 流式响应
    """

    def __init__(self, timeout: float = 120.0):
        """创建延迟初始化的 OpenAI 兼容异步客户端包装器。

        Args:
            timeout: 默认请求超时秒数，同时作为新建 ``httpx`` 客户端的默认超时。

        Notes:
            HTTP 连接池直到首次请求才创建；并发首次调用由异步锁保证只创建一个实例。
        """
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """返回可复用的开放连接池，并在缺失或已关闭时安全地重建。

        Returns:
            当前可用的 ``httpx.AsyncClient``。多个并发调用共享同一个实例。
        """
        if self._client is not None and not self._client.is_closed:
            return self._client

        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=self._timeout,
                    trust_env=True,
                )
        return self._client

    async def aclose(self) -> None:
        """关闭已创建的 HTTP 连接池。

        客户端尚未初始化或已经关闭时不执行操作，因此可安全重复调用。
        """
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
    ) -> str:
        """调用非流式聊天补全接口并提取第一条候选文本。

        Args:
            messages: OpenAI 兼容的角色/内容消息列表。
            temperature: 模型采样温度。
            max_tokens: 允许生成的最大 token 数。
            timeout: 本次请求的超时秒数；为 ``None`` 时使用实例默认值。

        Returns:
            响应 ``choices[0].message.content`` 中的完整回答文本。

        Raises:
            httpx.HTTPStatusError: 服务端返回非成功 HTTP 状态。
            Exception: 网络失败、JSON 解码失败或响应结构不符合预期。错误会记录后原样抛出。
        """
        s = settings()
        try:
            client = await self._get_client()
            resp = await client.post(
                s.DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {s.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": s.DEEPSEEK_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=self._timeout if timeout is None else timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            logger.info(
                f"[LLM] 调用成功 model={s.DEEPSEEK_MODEL} "
                f"prompt_tokens={usage.get('prompt_tokens')} "
                f"completion_tokens={usage.get('completion_tokens')}"
            )
            return content
        except httpx.HTTPStatusError as e:
            logger.error(f"[LLM] HTTP错误 {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"[LLM] 调用失败: {e}")
            raise

    async def chat_stream(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """调用流式聊天补全接口并逐段产出模型文本。

        Args:
            messages: OpenAI 兼容的角色/内容消息列表。
            temperature: 模型采样温度。
            max_tokens: 允许生成的最大 token 数。
            timeout: 本次流式请求的超时秒数；为 ``None`` 时使用实例默认值。

        Yields:
            每个有效 SSE ``data`` 事件中非空的增量 ``content`` 文本。

        Raises:
            Exception: 建连、HTTP 状态或读取流失败。单个格式错误、缺字段的 SSE
                数据块会被跳过，不会终止整个流。
        """
        s = settings()
        try:
            client = await self._get_client()
            async with client.stream(
                "POST",
                s.DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {s.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": s.DEEPSEEK_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                timeout=self._timeout if timeout is None else timeout,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except Exception as e:
            logger.error(f"[LLM-流式] 调用失败: {e}")
            raise
