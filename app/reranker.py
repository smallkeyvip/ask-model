"""
重排 + 上下文压缩：过滤无关文档，控制 Token 数量
"""
from __future__ import annotations

import logging
from typing import List

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings

RERANK_SYSTEM_PROMPT = """你是一个文档相关性评估专家。给定一个用户问题和一组候选文档，
请为每个文档评估与问题的相关性，输出格式为：
文档编号:相关性分数(0-10)

只输出评估结果，不要解释。示例：
1:8
2:3
3:9"""


def count_tokens_approx(text: str) -> int:
    """近似 Token 计数（中文约 1.5 字/token，英文约 4 字符/token）"""
    # 简单估算：中文按字数，英文按空格分词
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 0.7 + other_chars / 4)


class Reranker:
    """
    两阶段重排：
    1. 快速过滤：基于原始 score 阈值过滤低分文档
    2. LLM 精排：调用 LLM 对候选文档进行相关性打分重排
    3. Token 压缩：截断超出 max_context_tokens 的文档
    """

    def __init__(self, top_k: int = 5, max_context_tokens: int = 3000):
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

    async def rerank(self, query: str, documents: List[dict]) -> List[dict]:
        """
        对检索结果进行重排和压缩。
        返回: 重排后的文档列表（不超过 top_k 条，总 token 不超过 max_context_tokens）
        """
        if not documents:
            return []

        # Stage 1: 快速过滤（score < 0.3 的直接过滤）
        filtered = [d for d in documents if d.get("score", 0) >= 0.3]
        if not filtered:
            filtered = documents[:3]  # 兜底：至少保留前3条

        # Stage 2: LLM 精排（候选文档过多时启用）
        if len(filtered) > self.top_k:
            filtered = await self._llm_rerank(query, filtered)

        # 取 top_k
        reranked = filtered[:self.top_k]

        # Stage 3: Token 压缩
        reranked = self._truncate_to_token_limit(reranked)

        logger.info(f"[重排] 输入={len(documents)}, 过滤后={len(filtered)}, 最终={len(reranked)}")
        return reranked

    async def _llm_rerank(self, query: str, documents: List[dict]) -> List[dict]:
        """使用 LLM 对文档进行精排"""
        try:
            # 构建文档列表文本
            doc_texts = []
            for i, doc in enumerate(documents[:10], 1):  # 最多评估10条
                content = doc["content"][:200]  # 截取前200字
                doc_texts.append(f"文档{i}: {content}")

            user_msg = f"问题：{query}\n\n" + "\n".join(doc_texts)

            s = settings()
            async with httpx.AsyncClient(timeout=20, trust_env=True) as client:
                resp = await client.post(
                    s.DEEPSEEK_API_URL,
                    headers={
                        "Authorization": f"Bearer {s.DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": s.DEEPSEEK_MODEL,
                        "messages": [
                            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 200,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                result_text = data["choices"][0]["message"]["content"]

                # 解析评分结果
                scores = self._parse_rerank_scores(result_text, len(documents))
                for i, doc in enumerate(documents):
                    if i < len(scores):
                        doc["rerank_score"] = scores[i]
                    else:
                        doc["rerank_score"] = doc.get("score", 0) * 5

                # 按 rerank_score 降序排序
                documents.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
                return documents

        except Exception as e:
            logger.warning(f"[重排] LLM精排失败，使用原始排序: {e}")
            return documents

    @staticmethod
    def _parse_rerank_scores(text: str, expected_count: int) -> List[float]:
        """解析 LLM 输出的评分（格式：编号:分数）"""
        import re
        scores = []
        for match in re.finditer(r'(\d+)\s*[:：]\s*([\d.]+)', text):
            scores.append(float(match.group(2)))
        return scores

    def _truncate_to_token_limit(self, documents: List[dict]) -> List[dict]:
        """截断文档使总 token 不超过 max_context_tokens"""
        result = []
        total_tokens = 0
        for doc in documents:
            doc_tokens = count_tokens_approx(doc["content"])
            if total_tokens + doc_tokens > self.max_context_tokens:
                # 截断当前文档
                remaining = self.max_context_tokens - total_tokens
                if remaining > 50:  # 至少保留 50 token
                    approx_chars = int(remaining * 1.5)  # 粗略转换
                    doc = {**doc, "content": doc["content"][:approx_chars] + "...(已截断)"}
                    result.append(doc)
                break
            total_tokens += doc_tokens
            result.append(doc)
        return result
