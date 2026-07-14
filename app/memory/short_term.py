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
        """创建仅驻留内存的对话窗口。

        Args:
            window_size: 需要保留的完整问答轮数；内部容量按每轮两条消息计算。
        """
        self.window_size = window_size   # 单位：轮（turns），每轮 = user + assistant
        self._history: List[Turn] = []

    # ── 写入 ─────────────────────────────────────────────────────────
    def add(self, role: str, content: str, **metadata) -> None:
        """把一条消息追加到历史末尾。

        Args:
            role: 消息角色，调用约定为 ``user`` 或 ``assistant``。
            content: 消息正文。
            **metadata: 与该条消息一起保存的来源等附加字段。

        Notes:
            本方法不自动裁剪窗口；裁剪由 :meth:`trim_to_window` 显式触发。
        """
        self._history.append(Turn(role=role, content=content, metadata=metadata))

    # ── 读取最近 N 轮（2*N 条消息）──────────────────────────────────
    def get_window(self) -> List[Turn]:
        """读取窗口容量内最新的消息。

        Returns:
            新列表，最多包含 ``window_size * 2`` 个 ``Turn``；其中的 ``Turn``
            对象仍与内部历史共享。
        """
        max_msgs = self.window_size * 2
        return self._history[-max_msgs:]

    def get_window_as_messages(self) -> List[dict]:
        """把当前窗口转换为 LLM 接口所需的角色/内容字典。

        Returns:
            每项只含 ``role`` 和 ``content`` 的新字典列表，不携带时间戳与元数据。
        """
        return [{"role": t.role, "content": t.content} for t in self.get_window()]

    # ── 溢出检测 ─────────────────────────────────────────────────────
    def trim_to_window(self) -> List[Turn]:
        """删除超出窗口容量的最旧消息并返回被删部分。

        Returns:
            按原顺序排列的被移除消息；没有溢出时返回空列表且不修改历史。
        """
        max_msgs = self.window_size * 2
        if len(self._history) <= max_msgs:
            return []
        removed = self._history[:-max_msgs]
        self._history = self._history[-max_msgs:]
        return removed

    # ── 工具方法 ─────────────────────────────────────────────────────
    def get_all_turns(self) -> List[Turn]:
        """获取当前仍在短期存储中的全部消息。

        Returns:
            内部列表的浅拷贝，适用于会话归档；其中的 ``Turn`` 对象仍共享。
        """
        return list(self._history)

    @property
    def total_turns(self) -> int:
        """返回当前保存的消息条数，而不是完整问答轮数。"""
        return len(self._history)

    def clear(self) -> None:
        """原地删除全部短期消息；窗口大小配置保持不变。"""
        self._history.clear()
