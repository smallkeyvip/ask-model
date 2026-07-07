"""
LLM 调用模块：对接 DeepSeek Pro（兼容 OpenAI API 格式）
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, List

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

    async def chat(self, messages: List[dict], temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """非流式调用，返回完整文本"""
        s = settings()
        try:
            async with httpx.AsyncClient(timeout=120, trust_env=True) as client:
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
        self, messages: List[dict], temperature: float = 0.7, max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        """流式调用，逐步返回文本 chunk"""
        s = settings()
        try:
            async with httpx.AsyncClient(timeout=120, trust_env=True) as client:
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
