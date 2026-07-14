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

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        top_k: int = 5,
        user_id: str = "",
        session_id: str = "",
    ):
        """打开持久化 Chroma 客户端并绑定当前记忆所有者。

        Args:
            persist_dir: Chroma 持久化目录；未提供时使用全局配置。
            top_k: 召回方法未显式指定数量时使用的默认上限。
            user_id: 所有读写过滤使用的用户标识，防止跨成员访问。
            session_id: 未显式传入会话时写入或删除使用的默认会话标识。

        Notes:
            构造会创建或打开共享 collection，并执行一次当前用户记录数查询用于日志。
        """
        s = settings()
        self.top_k = top_k
        self.user_id = user_id
        self.session_id = session_id
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
            f"当前用户记录数: {self.count}"
        )

    # ── 写入 ─────────────────────────────────────────────────────────
    def store(
        self,
        content: str,
        category: str = "general",
        session_id: str = "",
        entities: Optional[List[str]] = None,
    ) -> str:
        """向当前用户的长期记忆写入一条带元数据的记录。

        Args:
            content: 记忆正文，例如关键结论、实体描述或主题摘要。
            category: 类别标签，例如 ``entity``、``topic`` 或 ``conclusion``。
            session_id: 所属会话；为空时使用构造时绑定的默认会话。
            entities: 关联实体列表，以逗号连接后写入 Chroma 元数据。

        Returns:
            为本条记录生成的 UUID 字符串。

        Raises:
            Exception: Chroma 持久化或嵌入计算失败。
        """
        doc_id = str(uuid.uuid4())
        metadata = {
            "category": category,
            "user_id": self.user_id,
            "session_id": session_id or self.session_id,
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
        """批量写入当前用户的长期记忆。

        Args:
            items: 记录字典列表；每项必须包含 ``content``，可选 ``category``
                和 ``entities``。
            session_id: 本批记录的会话；为空时使用构造时绑定的默认会话。

        Returns:
            与输入顺序对应的新 UUID 列表；输入为空时不访问存储并返回空列表。

        Raises:
            KeyError: 任一记录缺少 ``content``。
            Exception: Chroma 批量写入或嵌入计算失败。
        """
        if not items:
            return []
        ids = [str(uuid.uuid4()) for _ in items]
        documents = [it["content"] for it in items]
        metadatas = [
            {
                "category": it.get("category", "general"),
                "user_id": self.user_id,
                "session_id": session_id or self.session_id,
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
        """在当前用户范围内语义召回与查询最相关的历史记忆。

        Args:
            query: 用于向量查询的当前问题。
            top_k: 本次最大返回数量；未提供或为零时使用实例默认值。

        Returns:
            含正文、元数据和距离的结果列表。当前用户无记录时返回空列表。
        """
        k = top_k or self.top_k
        owner_count = self.count
        if owner_count == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, owner_count),
            where=self._owner_filter(),
        )
        return self._format_results(results)

    def recall_with_filter(
        self, query: str, category: str, top_k: Optional[int] = None
    ) -> List[dict]:
        """在当前用户和指定类别的交集内执行语义召回。

        Args:
            query: 用于向量查询的当前问题。
            category: 必须匹配的记忆类别元数据。
            top_k: 本次最大返回数量；未提供或为零时使用实例默认值。

        Returns:
            含正文、元数据和距离的结果列表；当前用户无任何记录时返回空列表。
        """
        k = top_k or self.top_k
        owner_count = self.count
        if owner_count == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, owner_count),
            where={
                "$and": [
                    self._owner_filter(),
                    {"category": category},
                ]
            },
        )
        return self._format_results(results)

    def _owner_filter(self) -> dict:
        """构造所有长期记忆查询必须携带的用户所有权过滤器。

        Returns:
            可直接传给 Chroma ``where`` 参数的 ``user_id`` 等值条件。
        """
        return {
            "user_id": self.user_id,
        }

    def clear_session(self, session_id: Optional[str] = None) -> None:
        """删除当前用户指定会话的全部持久记忆。

        Args:
            session_id: 目标会话；未提供或为空时使用构造时绑定的默认会话。

        Notes:
            删除条件同时包含 ``user_id`` 与 ``session_id``，不会影响其他用户。
        """
        self._collection.delete(
            where={
                "$and": [
                    self._owner_filter(),
                    {"session_id": session_id or self.session_id},
                ]
            }
        )

    # ── 工具方法 ─────────────────────────────────────────────────────
    @staticmethod
    def _format_results(results) -> List[dict]:
        """把 Chroma 单查询的嵌套响应展平为项目统一的结果结构。

        Args:
            results: Chroma ``query`` 返回值，预期第一维对应唯一查询文本。

        Returns:
            含 ``content``、``metadata`` 和 ``distance`` 的字典列表；响应为空时
            返回空列表。缺少元数据或距离时分别使用空字典或 ``None``。
        """
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
        """查询并返回当前用户拥有的长期记忆记录数。

        此属性会访问 Chroma 并按所有者过滤，不是单纯的内存计数。
        """
        records = self._collection.get(
            where=self._owner_filter(),
            include=["metadatas"],
        )
        return len(records.get("ids", []))

    def clear(self) -> None:
        """删除当前用户的全部长期记忆，不限制会话且不影响其他成员。"""
        self._collection.delete(where=self._owner_filter())
