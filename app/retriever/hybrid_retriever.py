"""
混合检索器：多路召回（百炼远程知识库 + 本地 Chroma 向量库）并合并去重
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import List

from app.config import get_settings
from app.retriever.bailian_retriever import BailianRetriever
from app.retriever.local_retriever import LocalRetriever

logger = logging.getLogger(__name__)
settings = get_settings


class HybridRetriever:
    """
    多路召回策略：
    1. 并发请求百炼远程知识库和本地 Chroma
    2. 对每个改写 query 都执行召回
    3. 合并去重，按 score 降序返回
    """

    def __init__(self):
        self.bailian = BailianRetriever()
        self.local = LocalRetriever()

    async def retrieve(self, queries: List[str], top_k: int = 10) -> List[dict]:
        """
        多路召回：对每个 query 并发请求远程+本地，合并去重后返回 top_k。
        """
        remote_tasks = [asyncio.create_task(self.bailian.retrieve(q)) for q in queries]
        local_task = asyncio.create_task(
            asyncio.to_thread(self.local.search_multi, queries, top_k)
        )
        gathered = await asyncio.gather(
            *remote_tasks,
            local_task,
            return_exceptions=True,
        )
        remote_results_list = gathered[:-1]
        local_result = gathered[-1]
        ranked_lists: List[List[dict]] = []
        all_results: List[dict] = []

        for results in remote_results_list:
            if isinstance(results, Exception):
                logger.warning(f"[混合检索] 远程检索异常: {results}")
                continue
            ranked_lists.append(results)
            all_results.extend(results)
        if isinstance(local_result, Exception):
            logger.warning(f"[混合检索] 本地检索异常: {local_result}")
        else:
            ranked_lists.append(local_result)
            all_results.extend(local_result)

        fused = self._reciprocal_rank_fusion(ranked_lists)
        final = self._ensure_source_coverage(fused, top_k)
        logger.info(
            f"[混合检索] queries={len(queries)}, "
            f"总召回={len(all_results)}, 融合后={len(fused)}, "
            f"最终返回={len(final)}"
        )
        return final

    async def aclose(self) -> None:
        await self.bailian.aclose()

    @staticmethod
    def _content_key(result: dict) -> str:
        content = re.sub(r"\s+", " ", result.get("content", "")).strip().casefold()
        return hashlib.sha256(content.encode("utf-8")).hexdigest() if content else ""

    @classmethod
    def _reciprocal_rank_fusion(cls, ranked_lists: List[List[dict]]) -> List[dict]:
        by_key: dict[str, dict] = {}
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, result in enumerate(ranked, 1):
                key = cls._content_key(result)
                if not key:
                    continue
                by_key.setdefault(key, dict(result))
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
        if not scores:
            return []
        max_score = max(scores.values())
        fused = []
        for key, raw_score in scores.items():
            item = by_key[key]
            item["retrieval_score"] = item.get("score", 0)
            item["score"] = raw_score / max_score
            fused.append(item)
        fused.sort(key=lambda item: item["score"], reverse=True)
        return fused

    @staticmethod
    def _ensure_source_coverage(fused: List[dict], top_k: int) -> List[dict]:
        selected = fused[:top_k]
        available_origins = {item.get("origin") for item in fused}
        selected_origins = {item.get("origin") for item in selected}
        for origin in available_origins - selected_origins:
            candidate = next((item for item in fused if item.get("origin") == origin), None)
            if candidate is None:
                continue
            if len(selected) < top_k:
                selected.append(candidate)
            elif selected:
                selected[-1] = candidate
        return selected
