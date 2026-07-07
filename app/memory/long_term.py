"""
长期实体/主题记忆（向量持久记忆）：跨会话的人物、业务、主题，存入 Chroma
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings

# 长期记忆的 Chroma collection 名称
LONG_TERM_COLLECTION = "long_term_memory"


class LongTermMemory:
    """
    基于 Chroma 的向量持久记忆：
    - 存储对话中提取的实体、主题、关键结论
    - 跨会话召回相关历史记忆
    - 支持增量更新（同一实体覆盖旧记录）
    """

    def __init__(self, persist_dir: Optional[str] = None, top_k: int = 5):
        s = settings()
        self.top_k = top_k
        self._client = chromadb.PersistentClient(
            path=persist_dir or s.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=LONG_TERM_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"[长期记忆] Chroma collection '{LONG_TERM_COLLECTION}' 已加载，"
            f"当前记录数: {self._collection.count()}"
        )

    # ── 写入 ─────────────────────────────────────────────────────────
    def store(
        self,
        content: str,
        category: str = "general",
        session_id: str = "",
        entities: Optional[List[str]] = None,
    ) -> str:
        """
        向长期记忆存入一条记录。
        - content: 记忆正文（关键结论/实体描述/主题摘要）
        - category: 类别标签（entity/topic/conclusion/preference）
        - session_id: 所属会话
        - entities: 关联实体列表，用于元数据过滤
        """
        doc_id = str(uuid.uuid4())
        metadata = {
            "category": category,
            "session_id": session_id,
            "entities": ",".join(entities or []),
            "timestamp": time.time(),
        }
        self._collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata],
        )
        logger.debug(f"[长期记忆] 存入记录 {doc_id}，category={category}")
        return doc_id

    def store_batch(
        self,
        items: List[dict],
        session_id: str = "",
    ) -> List[str]:
        """
        批量存入。每个 item: {"content": str, "category": str, "entities": list[str]}
        """
        if not items:
            return []
        ids = [str(uuid.uuid4()) for _ in items]
        documents = [it["content"] for it in items]
        metadatas = [
            {
                "category": it.get("category", "general"),
                "session_id": session_id,
                "entities": ",".join(it.get("entities", [])),
                "timestamp": time.time(),
            }
            for it in items
        ]
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(f"[长期记忆] 批量存入 {len(ids)} 条记录")
        return ids

    # ── 召回 ─────────────────────────────────────────────────────────
    def recall(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """语义召回与 query 最相关的历史记忆"""
        k = top_k or self.top_k
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
        )
        return self._format_results(results)

    def recall_with_filter(
        self, query: str, category: str, top_k: Optional[int] = None
    ) -> List[dict]:
        """按类别过滤后语义召回"""
        k = top_k or self.top_k
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
            where={"category": category},
        )
        return self._format_results(results)

    # ── 工具方法 ─────────────────────────────────────────────────────
    @staticmethod
    def _format_results(results) -> List[dict]:
        formatted = []
        if not results or not results["documents"]:
            return formatted
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else None
            formatted.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
            })
        return formatted

    @property
    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(LONG_TERM_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=LONG_TERM_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
