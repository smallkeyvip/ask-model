"""
本地 Chroma 向量知识库检索器
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings


class LocalRetriever:
    """
    基于 Chroma 的本地向量检索器：
    - 使用 sentence-transformers 生成 Embedding
    - 支持文档入库（add）和语义检索（search）
    """

    def __init__(self):
        s = settings()
        self._client = chromadb.PersistentClient(
            path=s.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=s.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_fn = None  # 延迟加载
        logger.info(
            f"[本地检索] collection '{s.CHROMA_COLLECTION}'，"
            f"当前文档数: {self._collection.count()}"
        )

    def _get_embedding_fn(self):
        """延迟加载 Embedding 模型（首次调用时下载/加载）"""
        if self._embedding_fn is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            s = settings()
            self._embedding_fn = HuggingFaceEmbeddings(
                # model_name 支持本地绝对路径，直接加载无需联网
                model_name=s.EMBEDDING_MODEL,
                model_kwargs={
                    "device": "cpu",
                    "local_files_only": True,
                },
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embedding_fn



    # ── 文档入库 ─────────────────────────────────────────────────────
    def add_documents(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ) -> int:
        """将文档批量写入本地向量库"""
        import uuid
        if not documents:
            return 0
        doc_ids = ids or [str(uuid.uuid4()) for _ in documents]
        embed_fn = self._get_embedding_fn()
        embeddings = embed_fn.embed_documents(documents)

        self._collection.add(
            ids=doc_ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in documents],
        )
        logger.info(f"[本地检索] 新增 {len(documents)} 条文档，总计 {self._collection.count()} 条")
        return len(documents)

    # ── 语义检索 ─────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """语义检索，返回最相关的 top_k 条文档"""
        if self._collection.count() == 0:
            return []

        embed_fn = self._get_embedding_fn()
        query_embedding = embed_fn.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
        )
        return self._format_results(results)

    def search_multi(self, queries: List[str], top_k: int = 10) -> List[dict]:
        """多 query 检索并去重合并"""
        seen_contents = set()
        merged = []
        for q in queries:
            hits = self.search(q, top_k=top_k)
            for hit in hits:
                key = hit["content"][:100]  # 简单去重
                if key not in seen_contents:
                    seen_contents.add(key)
                    merged.append(hit)
        return merged[:top_k]

    @staticmethod
    def _format_results(results) -> List[dict]:
        formatted = []
        if not results or not results["documents"]:
            return formatted
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 0
            formatted.append({
                "content": doc,
                "source": meta.get("source", "本地知识库"),
                "score": max(0, 1 - dist),  # cosine distance → similarity
                "metadata": meta,
                "origin": "local",
            })
        return formatted

    @property
    def count(self) -> int:
        return self._collection.count()
