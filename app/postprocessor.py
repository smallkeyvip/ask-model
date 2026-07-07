"""
结果后处理：引用溯源、格式整理、敏感词过滤
"""
from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# 默认敏感词列表（生产环境应从配置/数据库加载）
DEFAULT_SENSITIVE_WORDS = [
    "密码", "password", "secret_key", "api_key", "token",
    "银行卡号", "身份证号", "手机号", "手机号"
]


class PostProcessor:
    """
    LLM 输出后处理：
    1. 引用溯源：将 [来源X] 替换为实际的来源链接
    2. 格式整理：清理多余空白、规范化 Markdown
    3. 敏感词过滤：脱敏处理敏感信息
    """

    def __init__(self, sensitive_words: List[str] = None):
        self.sensitive_words = sensitive_words or DEFAULT_SENSITIVE_WORDS

    def process(self, text: str, sources: List[dict]) -> dict:
        """
        后处理 LLM 输出文本。

        返回:
            {
                "answer": str,           # 处理后的回答
                "sources": list,         # 引用来源列表
                "filtered_words": list,  # 被过滤的敏感词
            }
        """
        filtered_words = []

        # 1. 敏感词过滤
        text, filtered_words = self._filter_sensitive(text)

        # 2. 引用溯源处理
        text, cited_sources = self._process_citations(text, sources)

        # 3. 格式整理
        text = self._format_text(text)

        return {
            "answer": text,
            "sources": cited_sources,
            "filtered_words": filtered_words,
        }

    def _filter_sensitive(self, text: str) -> tuple[str, List[str]]:
        """敏感词脱敏"""
        filtered = []
        for word in self.sensitive_words:
            if word.lower() in text.lower():
                filtered.append(word)
                # 保留首尾字符，中间用 *** 替换
                if len(word) > 2:
                    masked = word[0] + "***" + word[-1]
                else:
                    masked = "***"
                text = re.sub(re.escape(word), masked, text, flags=re.IGNORECASE)
        return text, filtered

    def _process_citations(self, text: str, sources: List[dict]) -> tuple[str, List[dict]]:
        """处理引用标注，将 [来源X] 关联到实际来源"""
        cited = []
        cited_indices = set()

        # 查找所有 [来源X] 格式的引用
        pattern = r'\[来源(\d+)\]'
        matches = re.findall(pattern, text)

        for match in matches:
            idx = int(match) - 1  # 转为 0-based
            if 0 <= idx < len(sources):
                cited_indices.add(idx)

        # 构建引用来源列表
        for idx in sorted(cited_indices):
            source = sources[idx]
            cited.append({
                "index": idx + 1,
                "source": source.get("source", "知识库"),
                "origin": source.get("origin", "unknown"),
            })

        # 如果 LLM 没有主动引用，但提供了来源，追加来源列表
        if not cited and sources:
            source_list = []
            for i, s in enumerate(sources, 1):
                src_name = s.get("source", "知识库")
                source_list.append(f"[来源{i}] {src_name}")
            text += "\n\n---\n**参考来源：**\n" + "\n".join(source_list)
            cited = [
                {"index": i + 1, "source": s.get("source", ""), "origin": s.get("origin", "")}
                for i, s in enumerate(sources)
            ]

        return text, cited

    @staticmethod
    def _format_text(text: str) -> str:
        """格式整理"""
        # 清理连续空行（最多保留1个）
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 清理行尾空白
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        # 确保 Markdown 标题前有空行
        text = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', text)
        return text.strip()
