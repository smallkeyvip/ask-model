"""
结果后处理：引用溯源、格式整理、敏感词过滤
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# 默认敏感词列表（生产环境应从配置/数据库加载）
DEFAULT_SENSITIVE_WORDS: List[str] = []


class PostProcessor:
    """
    LLM 输出后处理：
    1. 引用溯源：将 [来源X] 替换为实际的来源链接
    2. 格式整理：清理多余空白、规范化 Markdown
    3. 敏感词过滤：脱敏处理敏感信息
    """

    def __init__(self, sensitive_words: Optional[List[str]] = None):
        """配置需要按字面值额外屏蔽的敏感词。

        Args:
            sensitive_words: 自定义词表。传入 ``None`` 时使用模块默认词表；
                传入空列表则明确禁用额外词表匹配。列表按引用保存，不会复制。
        """
        self.sensitive_words = sensitive_words if sensitive_words is not None else DEFAULT_SENSITIVE_WORDS

    def process(self, text: str, sources: List[dict]) -> dict:
        """依次对 LLM 输出执行脱敏、引用解析和 Markdown 空白整理。

        Args:
            text: 模型生成的原始回答文本。
            sources: 与 ``[来源N]`` 编号顺序一致的重排文档列表。

        Returns:
            包含 ``answer``、``sources`` 和 ``filtered_words`` 的字典。
            ``sources`` 仅含实际引用项；当模型未主动引用时，会追加完整参考来源列表。
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
        """对常见凭证、个人数据值和配置词表执行确定性脱敏。

        Args:
            text: 待过滤的回答文本。

        Returns:
            二元组：替换后的文本，以及命中的规则标签或敏感词列表。
            同一标签只按对应正则追加一次；自定义词按遍历顺序记录。
        """
        filtered = []
        value_patterns = [
            (
                "credential",
                re.compile(
                    r"(?i)\b(api[_-]?key|access[_-]?token|secret[_-]?key|password)\b(\s*[:=]\s*)([^\s,;]{8,})"
                ),
                lambda match: f"{match.group(1)}{match.group(2)}***",
            ),
            (
                "phone",
                re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)"),
                lambda match: f"{match.group(1)}****{match.group(2)}",
            ),
            (
                "id_number",
                re.compile(r"(?<!\d)(\d{6})\d{8}([\dXx]{4})(?!\d)"),
                lambda match: f"{match.group(1)}********{match.group(2)}",
            ),
            (
                "bank_card",
                re.compile(r"(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)"),
                lambda match: f"{match.group(1)}********{match.group(2)}",
            ),
        ]
        for label, pattern, replacement in value_patterns:
            text, count = pattern.subn(replacement, text)
            if count:
                filtered.append(label)

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
        """解析 ``[来源N]`` 标注并构建允许返回给浏览器的来源元数据。

        Args:
            text: 已脱敏但尚未整理格式的回答文本。
            sources: 按提示中编号顺序排列的候选来源。

        Returns:
            处理后的文本和去重、按编号排序的引用来源列表。越界编号会被忽略；
            若没有有效引用但存在候选来源，文本末尾会追加完整参考来源段落。
        """
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
            url = self._safe_source_url(source.get("doc_url", ""))
            cited.append({
                "index": idx + 1,
                "source": source.get("source", "知识库"),
                "origin": source.get("origin", "unknown"),
                "url": url,
            })

        # 如果 LLM 没有主动引用，但提供了来源，追加来源列表
        if not cited and sources:
            source_list = []
            for i, s in enumerate(sources, 1):
                src_name = s.get("source", "知识库")
                source_list.append(f"[来源{i}] {src_name}")
            text += "\n\n---\n**参考来源：**\n" + "\n".join(source_list)
            cited = [
                {
                    "index": i + 1,
                    "source": s.get("source", ""),
                    "origin": s.get("origin", ""),
                    "url": self._safe_source_url(s.get("doc_url", "")),
                }
                for i, s in enumerate(sources)
            ]

        return text, cited

    @staticmethod
    def _safe_source_url(value: str) -> str:
        """仅保留具有网络位置的 HTTP(S) 来源链接。

        Args:
            value: 来源记录中的候选 URL；非字符串也会被安全拒绝。

        Returns:
            通过协议和主机校验的原始 URL，否则返回空字符串。
        """
        if not isinstance(value, str):
            return ""
        parsed = urlparse(value)
        return value if parsed.scheme in {"http", "https"} and parsed.netloc else ""

    @staticmethod
    def _format_text(text: str) -> str:
        """规范化回答中的空行、行尾空白和 Markdown 标题间距。

        Args:
            text: 已完成脱敏和引用处理的回答文本。

        Returns:
            首尾无空白、连续空行不超过一行且标题前有空行的文本。
        """
        # 清理连续空行（最多保留1个）
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 清理行尾空白
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        # 确保 Markdown 标题前有空行
        text = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', text)
        return text.strip()
