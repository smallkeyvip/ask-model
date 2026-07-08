"""
用户输入预处理模块

功能：
1. 敏感词过滤 —— 关键词匹配，命中直接拦截
2. 业务相关性判断 —— LLM 轻量判断，非业务问题提示引导
3. 指代消解 + 省略补全 —— LLM + 历史对话，产出完整独立问题
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from app.config import get_settings
from app.sensitive_words import (
    get_default_sensitive_config,
    merge_sensitive_config,
)

try:
    from opencc import OpenCC
except Exception:  # pragma: no cover - optional dependency
    OpenCC = None

try:
    from zhconv import convert as zhconv_convert
except Exception:  # pragma: no cover - optional dependency
    zhconv_convert = None

logger = logging.getLogger(__name__)
settings = get_settings

_OPENCC_T2S = None
if OpenCC is not None:
    try:
        _OPENCC_T2S = OpenCC("t2s")
    except Exception:  # pragma: no cover - optional dependency
        _OPENCC_T2S = None

_TRADITIONAL_TO_SIMPLIFIED = str.maketrans({
    "臺": "台",
    "灣": "湾",
    "獨": "独",
    "東": "东",
    "輪": "轮",
    "顛": "颠",
    "顔": "颜",
    "顏": "颜",
    "國": "国",
    "組": "组",
    "織": "织",
    "襲": "袭",
    "擊": "击",
    "極": "极",
    "製": "制",
    "彈": "弹",
    "遙": "遥",
    "藥": "药",
    "槍": "枪",
    "殺": "杀",
    "滅": "灭",
    "門": "门",
    "報": "报",
    "會": "会",
    "縱": "纵",
    "氣": "气",
    "體": "体",
    "綁": "绑",
    "質": "质",
    "黃": "黄",
    "兒": "儿",
    "強": "强",
    "姦": "奸",
    "媽": "妈",
    "販": "贩",
    "運": "运",
    "嗎": "吗",
    "賭": "赌",
    "馬": "马",
    "樂": "乐",
    "錢": "钱",
    "賣": "卖",
    "買": "买",
    "貸": "贷",
    "鈔": "钞",
    "偽": "伪",
    "僞": "伪",
    "兇": "凶",
    "網": "网",
    "詐": "诈",
    "騙": "骗",
    "庫": "库",
    "學": "学",
    "歷": "历",
    "證": "证",
    "辦": "办",
})

# ── LLM 提示词 ──────────────────────────────────────────────────────

BUSINESS_CHECK_SYSTEM_PROMPT = """你是一个企业知识问答系统的前置过滤器。
企业业务领域包括但不限于：
{business_keywords}

请判断用户问题是否与企业业务相关。

判断标准：
- 询问公司业务流程、规则、系统操作、数据查询 → BUSINESS
- 询问公司产品、服务、合作方、设备 → BUSINESS
- 问题完全与以上业务领域无关（闲聊、通用编程、娱乐、政治等） → NOT_BUSINESS
- 问题可能涉及业务领域但表述模糊 → BUSINESS（允许通过，宁可放过也不误拦）

仅回复 BUSINESS 或 NOT_BUSINESS，不要解释。"""


REFERENCE_RESOLVE_SYSTEM_PROMPT = """你是一个对话消解助手。根据对话历史，将用户的当前问题进行指代消解和省略补全，
使其成为一个独立完整的自然语言问题。

消解规则：
1. 代词替换：将"它/他/她/这/那/这个/那个/其/该"替换为对话历史中出现的具体实体
2. 省略补全：如"那拆分呢？"→ 结合上文补全为"返还单的拆分规则是什么？"
3. 保持意图：仅补全省略信息和替换代词，不改变用户问题的核心意图
4. 直接输出：仅输出消解后的完整问题，不要解释、不要标记、不要额外文字
5. 无需消解：如果当前问题已经是独立完整的问题，直接原文输出"""

REFERENCE_RESOLVE_EXAMPLE_USER = """对话历史：
用户：返还单的合并规则是什么？
助手：返还单合并主要有三种场景：自动合并、手动合并、批量合并。

当前问题：那拆分呢？
消解结果：返还单的拆分规则是什么？

对话历史：
用户：渠道商A最近的退赔数据怎么样？
助手：渠道商A上月退赔金额为12万元，主要集中在线下门店设备损坏。

当前问题：它的统计报表怎么导出？
消解结果：渠道商A的退赔统计报表怎么导出？

对话历史：
用户：帮我查一下SNABC123456789的状态
助手：该设备目前处于待维修状态，持有人为渠道商B。

当前问题：什么时候能修好？
消解结果：SNABC123456789什么时候能修好？"""

# ── 默认业务领域关键词 ──────────────────────────────────────────────

_DEFAULT_BUSINESS_KEYWORDS = (
    "怪兽充电,充电宝,供应链,供应链资产,退赔,赔偿,返修,报废,"
    "渠道商,商户,门店,BD,SN,设备,资产,PCN,SPU,SKU,"
    "返还单,赔付,维修,物流,入库,出库,到货登记,"
    "阿里百炼,eboss,PMS,NPS,ASQ,ACQ,FMS,"
    "退赔明细,赔偿申请,退赔申请,退赔统计"
)


# ── 数据结构 ────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    """预处理结果"""
    blocked: bool = False
    reason: str = ""  # 拦截原因（仅 blocked=True 时有效）
    original_query: str = ""  # 原始问题
    processed_query: str = ""  # 预处理后的问题（blocked=False 时有效）


# ── 主类 ────────────────────────────────────────────────────────────

class InputPreprocessor:
    """
    用户输入预处理器。

    使用方式：
        preprocessor = InputPreprocessor()
        result = await preprocessor.preprocess(query, history_messages)
        if result.blocked:
            return blocked_response(result.reason)
        query = result.processed_query  # 消解后的完整问题
    """

    def __init__(self, business_keywords: Optional[str] = None):
        s = settings()

        # ── 敏感词库 ──
        extra_config = self._load_extra_sensitive_config(s)
        self._sensitive_config = merge_sensitive_config(
            get_default_sensitive_config(),
            extra_config,
        )
        self._sensitive: Dict[str, List[str]] = self._sensitive_config["keywords"]
        # 构建快速查找集合（大小写、繁简、全半角、空格/符号归一化）
        self._sensitive_flat: Dict[str, str] = {}
        for category, keywords in self._sensitive.items():
            for kw in keywords:
                normalized = self._normalize_sensitive_text(kw)
                if normalized:
                    self._sensitive_flat[normalized] = category

        self._sensitive_lists = self._normalize_sensitive_lists(
            self._sensitive_config.get("lists", {})
        )
        self._sensitive_rules = self._normalize_sensitive_rules(
            self._sensitive_config.get("rules", {})
        )

        logger.info(
            f"[预处理] 敏感词库加载完成，共 {len(self._sensitive_flat)} 个词，"
            f"分类: {list(self._sensitive.keys())}"
        )

        # ── 业务关键词 ──
        self._business_keywords = business_keywords or _DEFAULT_BUSINESS_KEYWORDS
        logger.info(f"[预处理] 业务关键词: {self._business_keywords[:100]}...")

    # ── 公开方法 ─────────────────────────────────────────────────────

    async def preprocess(
        self,
        query: str,
        history_messages: Optional[List[dict]] = None,
    ) -> PreprocessResult:
        """
        对用户输入执行完整预处理流程。

        参数:
            query: 用户原始问题
            history_messages: 短期记忆中的对话历史 [{"role":..., "content":...}, ...]

        返回:
            PreprocessResult
        """
        history_messages = history_messages or []

        # ── Step 1: 敏感词过滤（同步，快速失败）──
        sensitive_result = self._check_sensitive(query)
        if sensitive_result.blocked:
            return sensitive_result

        # ── Step 2: 业务相关性判断（LLM）──
        business_result = await self._check_business_relevance(query)
        if business_result.blocked:
            return business_result

        # ── Step 3: 指代消解（LLM + 历史）──
        resolved = query
        if history_messages:
            resolved = await self._resolve_references(query, history_messages)

        logger.info(
            f"[预处理] 通过: '{query[:50]}...'"
            + (f" -> '{resolved[:50]}...'" if resolved != query else "")
        )

        return PreprocessResult(
            blocked=False,
            original_query=query,
            processed_query=resolved,
        )

    # ── Step 1: 敏感词过滤 ───────────────────────────────────────────

    def _check_sensitive(self, query: str) -> PreprocessResult:
        """关键词包含匹配，命中任一敏感词即拦截"""
        q = self._normalize_sensitive_text(query)

        rule_hit = self._check_sensitive_rules(q)
        if rule_hit:
            category, rule_name = rule_hit
            return self._build_sensitive_block_result(
                query=query,
                category=category,
                hit_type=rule_name,
            )

        for keyword, category in self._sensitive_flat.items():
            if self._match_sensitive_keyword(keyword, q):
                return self._build_sensitive_block_result(
                    query=query,
                    category=category,
                    hit_type="keyword",
                )
        return PreprocessResult(blocked=False)

    def _check_sensitive_rules(self, normalized_query: str) -> Optional[tuple[str, str]]:
        """组合规则匹配，覆盖动作+目标、人物、活动等非连续关键词表达。"""
        for category, rules in self._sensitive_rules.items():
            for rule in rules:
                rule_type = rule.get("type")
                rule_name = rule.get("name", rule_type or "sensitive_rule")

                if rule_type == "direct_list":
                    words = self._sensitive_lists.get(rule.get("list", ""), [])
                    if self._contains_any(normalized_query, words):
                        return category, rule_name

                elif rule_type == "intent_list":
                    intents = self._sensitive_lists.get(rule.get("intents", ""), [])
                    words = self._sensitive_lists.get(rule.get("list", ""), [])
                    if (
                        self._contains_any(normalized_query, intents)
                        and self._contains_any(normalized_query, words)
                    ):
                        return category, rule_name

                elif rule_type == "action_target":
                    if (
                        self._contains_any(normalized_query, rule.get("actions", []))
                        and self._contains_any(normalized_query, rule.get("targets", []))
                    ):
                        return category, rule_name

        return None

    @staticmethod
    def _contains_any(normalized_text: str, words: List[str]) -> bool:
        return any(word and word in normalized_text for word in words)

    @staticmethod
    def _build_sensitive_block_result(
        query: str,
        category: str,
        hit_type: str,
    ) -> PreprocessResult:
        logger.warning(
            f"[预处理-拦截] 敏感词分类={category}, hit={hit_type}, "
            f"query='{query[:80]}...'"
        )
        return PreprocessResult(
            blocked=True,
            reason="抱歉，输入包含不当内容，无法处理。",
            original_query=query,
        )

    @staticmethod
    def _normalize_sensitive_text(text: str) -> str:
        """
        敏感词匹配专用归一化：
        1. 大小写统一
        2. 繁简转换
        3. 全半角转换
        4. 空格、标点、符号压缩
        """
        if not text:
            return ""

        normalized = text.casefold()

        if _OPENCC_T2S is not None:
            normalized = _OPENCC_T2S.convert(normalized)
        elif zhconv_convert is not None:
            normalized = zhconv_convert(normalized, "zh-cn")
        else:
            normalized = normalized.translate(_TRADITIONAL_TO_SIMPLIFIED)

        normalized = unicodedata.normalize("NFKC", normalized).casefold()

        compact_chars = []
        for ch in normalized:
            category = unicodedata.category(ch)
            if ch.isspace() or category[0] in {"P", "S", "Z"}:
                continue
            compact_chars.append(ch)
        return "".join(compact_chars)

    @staticmethod
    def _match_sensitive_keyword(keyword: str, normalized_query: str) -> bool:
        """短英文/数字词使用边界匹配，降低包含匹配误伤。"""
        if not keyword:
            return False

        if keyword.isascii() and keyword.isalnum() and len(keyword) <= 2:
            pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
            return re.search(pattern, normalized_query) is not None

        return keyword in normalized_query

    # ── Step 2: 业务相关性判断 ─────────────────────────────────────────

    async def _check_business_relevance(self, query: str) -> PreprocessResult:
        """LLM 判断问题是否属于公司业务范围"""
        system_prompt = BUSINESS_CHECK_SYSTEM_PROMPT.format(
            business_keywords=self._business_keywords,
        )
        try:
            result = await self._call_llm(
                system_prompt=system_prompt,
                user_msg=f"用户问题：{query}",
                max_tokens=10,
                temperature=0.0,
            )
            if result and "NOT_BUSINESS" in result.upper():
                logger.info(
                    f"[预处理-拦截] 非业务问题: '{query[:80]}...'"
                )
                return PreprocessResult(
                    blocked=True,
                    reason="请咨询公司业务相关问题，如退赔流程、供应链管理、设备状态查询等。",
                    original_query=query,
                )
        except Exception as e:
            # 业务判断 LLM 失败时放行，避免阻塞正常业务咨询
            logger.warning(f"[预处理] 业务相关性判断失败，放行: {e}")

        return PreprocessResult(blocked=False)

    # ── Step 3: 指代消解 ─────────────────────────────────────────────

    async def _resolve_references(
        self, query: str, history_messages: List[dict]
    ) -> str:
        """利用历史对话对当前问题进行指代消解和省略补全"""
        # 构建历史对话文本
        history_pairs = []
        user_turn = None
        for m in history_messages[-6:]:  # 最近 3 轮
            if m.get("role") == "user":
                user_turn = m["content"]
            elif m.get("role") == "assistant" and user_turn is not None:
                history_pairs.append(f"用户：{user_turn}")
                history_pairs.append(f"助手：{m['content']}")
                user_turn = None

        if not history_pairs:
            return query  # 没有有效历史，无需消解

        history_text = "\n".join(history_pairs)
        user_msg = f"对话历史：\n{history_text}\n\n当前问题：{query}"

        try:
            result = await self._call_llm(
                system_prompt=REFERENCE_RESOLVE_SYSTEM_PROMPT,
                user_msg=user_msg,
                max_tokens=100,
                temperature=0.2,
            )
            if result:
                logger.debug(
                    f"[预处理-消解] '{query[:50]}...' → '{result[:50]}...'"
                )
                return result.strip()
        except Exception as e:
            logger.warning(f"[预处理] 指代消解失败，使用原始query: {e}")

        return query

    # ── LLM 调用 ─────────────────────────────────────────────────────

    async def _call_llm(
        self,
        system_prompt: str,
        user_msg: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """调用 DeepSeek LLM，与项目现有 LLMClient 独立的轻量版本"""
        s = settings()
        async with httpx.AsyncClient(timeout=20, trust_env=True) as client:
            resp = await client.post(
                s.DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {s.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": s.DEEPSEEK_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    # ── 敏感词加载辅助 ───────────────────────────────────────────────

    def _load_extra_sensitive_config(self, s) -> Dict[str, Any]:
        """
        从配置/文件中加载额外敏感配置。

        兼容旧格式：
            {"political": [...], "illegal": [...]}
        统一格式：
            {
              "keywords": {"political": [...]},
              "lists": {"political_persons": [...], "political_activities": [...]},
              "rules": {"political": [...]}
            }
        """
        extra: Dict[str, Any] = {}
        file_path = s.SENSITIVE_KEYWORDS_EXTRA_FILE
        if file_path:
            try:
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    extra = data
                    logger.info(f"[预处理] 从文件加载额外敏感配置: {file_path}")
                else:
                    logger.error("[预处理] 额外敏感配置必须是 JSON object")
            except Exception as e:
                logger.error(f"[预处理] 加载额外敏感配置文件失败: {e}")
        return extra

    @classmethod
    def _normalize_sensitive_lists(cls, lists: Any) -> Dict[str, List[str]]:
        """归一化统一配置中的 lists。"""
        if not isinstance(lists, dict):
            if lists:
                logger.warning("[预处理] 忽略无效敏感名单配置，期望 dict")
            return {}

        return {
            str(name): cls._normalize_word_list(words)
            for name, words in lists.items()
        }

    @classmethod
    def _normalize_sensitive_rules(cls, rules: Any) -> Dict[str, List[Dict[str, Any]]]:
        """归一化统一配置中的 rules。"""
        if not isinstance(rules, dict):
            if rules:
                logger.warning("[预处理] 忽略无效敏感规则配置，期望 dict")
            return {}

        normalized_rules: Dict[str, List[Dict[str, Any]]] = {}
        for category, category_rules in rules.items():
            if not isinstance(category_rules, list):
                logger.warning(f"[预处理] 忽略无效敏感规则分类配置: {category}")
                continue

            normalized_rules[str(category)] = []
            for rule in category_rules:
                if not isinstance(rule, dict):
                    continue

                normalized_rule = dict(rule)
                if "actions" in normalized_rule:
                    normalized_rule["actions"] = cls._normalize_word_list(
                        normalized_rule["actions"]
                    )
                if "targets" in normalized_rule:
                    normalized_rule["targets"] = cls._normalize_word_list(
                        normalized_rule["targets"]
                    )
                normalized_rules[str(category)].append(normalized_rule)

        return normalized_rules

    @classmethod
    def _normalize_word_list(cls, words: Any) -> List[str]:
        """归一化词表。"""
        if not isinstance(words, list):
            if words:
                logger.warning("[预处理] 忽略无效敏感名单配置，期望 list")
            return []

        result = []
        seen = set()
        for word in words:
            if not isinstance(word, (str, int, float)):
                continue
            normalized = cls._normalize_sensitive_text(str(word))
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result
