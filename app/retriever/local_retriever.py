"""
本地 Chroma 向量知识库检索器
"""
from __future__ import annotations

import logging
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
        """打开本地持久化 Chroma collection，并延迟准备嵌入模型。

        构造使用全局持久化目录、collection 名和余弦距离配置；只查询当前文档数
        用于日志，不会在此时加载或下载 Embedding 模型。
        """
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
        """获取进程内复用的 Hugging Face Embedding 实例。

        Returns:
            已配置 CPU、向量归一化和本地文件策略的 ``HuggingFaceEmbeddings``。

        Raises:
            ImportError: 未安装嵌入依赖。
            Exception: 首次加载时模型目录无效，或禁止联网且本地模型不存在。

        Notes:
            首次调用可能下载模型；后续调用返回缓存实例。本方法未提供并发初始化锁。
        """
        if self._embedding_fn is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            s = settings()
            self._embedding_fn = HuggingFaceEmbeddings(
                # model_name 支持本地绝对路径，直接加载无需联网
                model_name=s.EMBEDDING_MODEL,
                model_kwargs={
                    "device": "cpu",
                    "local_files_only": s.EMBEDDING_LOCAL_FILES_ONLY,
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
        """生成文档向量并批量写入本地知识 collection。

        Args:
            documents: 待入库的正文列表。
            ids: 可选的稳定文档 ID；未提供时为每条正文生成 UUID。
            metadatas: 可选的逐文档元数据；未提供时写入对应数量的空字典。

        Returns:
            成功提交给 Chroma 的文档数量；输入为空时返回 0 且不加载模型。

        Raises:
            Exception: ID/元数据数量与正文不匹配、嵌入失败或 Chroma 写入失败。
        """
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
        """对单条查询执行本地向量检索。

        Args:
            query: 需要编码并检索的文本。
            top_k: 最大返回数量，实际值不会超过 collection 文档数。

        Returns:
            按 Chroma 相关性顺序排列的统一文档字典列表；空库直接返回空列表。

        Raises:
            Exception: 模型加载、查询编码或 Chroma 查询失败。
        """
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
        """批量检索多条改写查询，并按正文前缀去重合并最佳结果。

        Args:
            queries: 需要同时编码的查询列表。
            top_k: 每条查询和最终合并结果的最大数量。

        Returns:
            按相似度降序排列、最多 ``top_k`` 条的文档列表。正文前 100 个字符
            相同的命中只保留分数最高者；查询为空或库为空时返回空列表。

        Raises:
            Exception: 模型加载、批量编码或 Chroma 查询失败。
        """
        if not queries or self._collection.count() == 0:
            return []
        embed_fn = self._get_embedding_fn()
        query_embeddings = embed_fn.embed_documents(queries)
        results = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=min(top_k, self._collection.count()),
        )
        best_by_content = {}
        for query_index in range(len(queries)):
            hits = self._format_results(results, query_index)
            for hit in hits:
                key = hit["content"][:100]  # 简单去重
                previous = best_by_content.get(key)
                if previous is None or hit.get("score", 0) > previous.get("score", 0):
                    best_by_content[key] = hit
        merged = sorted(
            best_by_content.values(),
            key=lambda hit: hit.get("score", 0),
            reverse=True,
        )
        return merged[:top_k]

    @staticmethod
    def _format_results(results, query_index: int = 0) -> List[dict]:
        """把 Chroma 指定查询的嵌套响应转换为统一检索结果。

        Args:
            results: Chroma ``query`` 返回的批量响应。
            query_index: 需要提取的查询结果下标。

        Returns:
            含 ``content``、``source``、``score``、``metadata`` 和 ``origin``
            的列表。空响应或下标越界时返回空列表；余弦距离转换为不小于零的相似度。
        """
        formatted = []
        if not results or not results["documents"]:
            return formatted
        if query_index >= len(results["documents"]):
            return formatted
        for i, doc in enumerate(results["documents"][query_index]):
            meta = results["metadatas"][query_index][i] if results["metadatas"] else {}
            dist = results["distances"][query_index][i] if results["distances"] else 0
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
        """查询并返回本地知识 collection 的当前文档总数。"""
        return self._collection.count()
