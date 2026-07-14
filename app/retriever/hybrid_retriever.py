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
        """创建远程百炼检索器和本地 Chroma 检索器。"""
        self.bailian = BailianRetriever()
        self.local = LocalRetriever()

    async def retrieve(self, queries: List[str], top_k: int = 10) -> List[dict]:
        """并发执行每条查询的远程召回和一次本地批量召回，再融合排序。

        Args:
            queries: 查询改写器生成的一至多条检索文本。
            top_k: 融合后最多返回的文档数量。

        Returns:
            内容去重、RRF 归一化评分且尽量覆盖可用来源的文档列表。任一路异常
            都会被单独记录并忽略；所有路均无结果时返回空列表。
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
        """关闭混合检索器持有的异步远程连接池。

        本地 Chroma 客户端为同步持久化对象，没有对应的异步关闭操作。
        """
        await self.bailian.aclose()

    @staticmethod
    def _content_key(result: dict) -> str:
        """为正文生成不暴露内容的稳定去重键。

        Args:
            result: 包含可选 ``content`` 字段的检索结果。

        Returns:
            正文经过空白压缩、首尾清理和大小写折叠后的 SHA-256 十六进制摘要；
            正文为空时返回空字符串并由融合逻辑丢弃该项。
        """
        content = re.sub(r"\s+", " ", result.get("content", "")).strip().casefold()
        return hashlib.sha256(content.encode("utf-8")).hexdigest() if content else ""

    @classmethod
    def _reciprocal_rank_fusion(cls, ranked_lists: List[List[dict]]) -> List[dict]:
        """按内容去重并对多个排序列表应用倒数排名融合。

        Args:
            ranked_lists: 各检索路径按相关性降序排列的结果列表。

        Returns:
            按融合分降序排列的新字典列表。相同正文的各路贡献按
            ``1 / (60 + rank)`` 累加，最高融合分归一化为 1；原始 ``score``
            保存为 ``retrieval_score``。没有有效正文时返回空列表。
        """
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
        """在数量限制内尽量让最终结果包含每种已召回来源。

        Args:
            fused: 已按融合分降序排列的文档列表。
            top_k: 最终结果数量上限。

        Returns:
            初始前 ``top_k`` 条的浅列表。若某来源未入选，则追加到空余位置，
            或用该来源排名最高的文档替换当前末项；``top_k`` 小于等于零时返回空列表。
        """
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
