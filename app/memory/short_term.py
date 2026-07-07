"""
短期记忆（窗口记忆）：保留最近 N 轮完整问答
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class Turn:
    role: str          # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)   # 引用来源等附加信息


class ShortTermMemory:
    """
    双端队列式窗口记忆，保留最近 window_size 轮对话（1轮 = 1 user + 1 assistant）。
    超出窗口的对话会被弹出并交给中期摘要模块处理。
    """

    def __init__(self, window_size: int = 6):
        self.window_size = window_size   # 单位：轮（turns），每轮 = user + assistant
        self._history: List[Turn] = []

    # ── 写入 ─────────────────────────────────────────────────────────
    def add(self, role: str, content: str, **metadata) -> None:
        self._history.append(Turn(role=role, content=content, metadata=metadata))

    # ── 读取最近 N 轮（2*N 条消息）──────────────────────────────────
    def get_window(self) -> List[Turn]:
        max_msgs = self.window_size * 2
        return self._history[-max_msgs:]

    def get_window_as_messages(self) -> List[dict]:
        return [{"role": t.role, "content": t.content} for t in self.get_window()]

    # ── 溢出检测 ─────────────────────────────────────────────────────
    def overflowed_turns(self) -> List[Turn]:
        """返回超出窗口的历史轮次（最旧部分），供中期摘要消费。"""
        max_msgs = self.window_size * 2
        if len(self._history) <= max_msgs:
            return []
        return self._history[:-max_msgs]

    def trim_to_window(self) -> List[Turn]:
        """裁剪并返回被移除的 turns。"""
        max_msgs = self.window_size * 2
        if len(self._history) <= max_msgs:
            return []
        removed = self._history[:-max_msgs]
        self._history = self._history[-max_msgs:]
        return removed

    # ── 工具方法 ─────────────────────────────────────────────────────
    def get_all_turns(self) -> List[Turn]:
        """返回全部历史 turns（供会话归档时使用）"""
        return list(self._history)

    @property
    def total_turns(self) -> int:
        return len(self._history)

    def clear(self) -> None:
        self._history.clear()
