"""
混合检索器：多路召回（百炼远程知识库 + 本地 Chroma 向量库）并合并去重
"""
from __future__ import annotations

import asyncio
import logging
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
        all_results: List[dict] = []

        # 并发执行所有 query 的远程和本地检索
        tasks = []
        for q in queries:
            tasks.append(self.bailian.retrieve(q))
        # 本地检索同步执行（Chroma 是本地操作，不需要 await）
        local_results = self.local.search_multi(queries, top_k=top_k)

        # 等待远程结果
        remote_results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并远程结果
        for results in remote_results_list:
            if isinstance(results, Exception):
                logger.warning(f"[混合检索] 远程检索异常: {results}")
                continue
            all_results.extend(results)

        # 合并本地结果
        all_results.extend(local_results)

        # 去重（按内容前100字符）
        seen = set()
        unique_results = []
        for r in all_results:
            key = r.get("content", "")[:100]
            if key and key not in seen:
                seen.add(key)
                unique_results.append(r)

        # 按 score 降序排序
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        final = unique_results[:top_k]
        logger.info(
            f"[混合检索] queries={len(queries)}, "
            f"总召回={len(all_results)}, 去重后={len(unique_results)}, "
            f"最终返回={len(final)}"
        )
        return final
