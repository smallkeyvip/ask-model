"""
敏感词配置：从统一 JSON 文件加载关键词、名单、组合规则和业务关键词。

运行时代码不再保存内置词表；配置文件是敏感词与业务关键词的唯一数据源。
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List


SENSITIVE_CATEGORIES = ("political", "violence", "pornographic", "illegal")
DEFAULT_SENSITIVE_CONFIG_FILE = "./sensitive_words_extra.json"


def _as_word_list(value: Any) -> List[str]:
    """把配置中的标量列表转换为规范字符串词表。

    Args:
        value: 候选列表；非列表值视为无配置。

    Returns:
        仅保留字符串、整数和浮点数中转为文本后非空的项，并保持输入顺序。
        本方法不执行大小写、繁简或符号归一化。
    """
    if not isinstance(value, list):
        return []
    return [
        str(item)
        for item in value
        if isinstance(item, (str, int, float)) and str(item).strip()
    ]


def _load_word_mapping(value: Any, field_name: str) -> Dict[str, List[str]]:
    """校验并复制“名称到词表”的配置区段。

    Args:
        value: JSON 中的候选映射。
        field_name: 用于错误提示的顶层字段名。

    Returns:
        键统一转为字符串、值转换为字符串词表的新字典。

    Raises:
        ValueError: 区段不是对象，或任一词表不是数组。
    """
    if not isinstance(value, dict):
        raise ValueError(f"敏感词配置字段 {field_name!r} 必须是 JSON object")

    result: Dict[str, List[str]] = {}
    for name, words in value.items():
        if not isinstance(words, list):
            raise ValueError(
                f"敏感词配置字段 {field_name}.{name} 必须是 JSON array"
            )
        result[str(name)] = _as_word_list(words)
    return result


def load_sensitive_config(file_path: str) -> Dict[str, Any]:
    """从 UTF-8 JSON 文件加载并校验统一敏感词配置。

    统一格式包含 ``keywords``、``lists``、``rules``、``business_keywords``
    和 ``output_sensitive_words``。敏感分类必须完整出现，避免文件错误导致某类
    过滤被静默关闭。

    Args:
        file_path: 配置文件路径。

    Returns:
        与文件数据隔离的规范配置字典，可由调用方安全修改。

    Raises:
        ValueError: 路径为空、文件无法读取、JSON 无效或配置结构不完整。
    """
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("敏感词配置文件路径不能为空")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"加载敏感词配置文件失败: {file_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("敏感词配置顶层必须是 JSON object")

    keywords = _load_word_mapping(data.get("keywords"), "keywords")
    missing_keyword_categories = [
        category for category in SENSITIVE_CATEGORIES if category not in keywords
    ]
    if missing_keyword_categories:
        raise ValueError(
            "敏感词配置缺少关键词分类: "
            + ", ".join(missing_keyword_categories)
        )

    lists = _load_word_mapping(data.get("lists"), "lists")

    rule_data = data.get("rules")
    if not isinstance(rule_data, dict):
        raise ValueError("敏感词配置字段 'rules' 必须是 JSON object")
    rules: Dict[str, List[Dict[str, Any]]] = {}
    for category in SENSITIVE_CATEGORIES:
        category_rules = rule_data.get(category)
        if not isinstance(category_rules, list):
            raise ValueError(
                f"敏感词配置字段 rules.{category} 必须是 JSON array"
            )
        if any(not isinstance(rule, dict) for rule in category_rules):
            raise ValueError(
                f"敏感词配置字段 rules.{category} 只能包含 JSON object"
            )
        rules[category] = deepcopy(category_rules)

    business_keywords = data.get("business_keywords")
    if not isinstance(business_keywords, list):
        raise ValueError("敏感词配置字段 'business_keywords' 必须是 JSON array")
    business_keywords = _as_word_list(business_keywords)
    if not business_keywords:
        raise ValueError("敏感词配置字段 'business_keywords' 不能为空")

    output_sensitive_words = data.get("output_sensitive_words")
    if not isinstance(output_sensitive_words, list):
        raise ValueError(
            "敏感词配置字段 'output_sensitive_words' 必须是 JSON array"
        )

    return {
        "keywords": keywords,
        "lists": lists,
        "rules": rules,
        "business_keywords": business_keywords,
        "output_sensitive_words": _as_word_list(output_sensitive_words),
    }


def get_default_sensitive_config(
    file_path: str = DEFAULT_SENSITIVE_CONFIG_FILE,
) -> Dict[str, Any]:
    """从默认或指定配置文件读取系统敏感词配置。

    Args:
        file_path: 统一敏感词配置文件路径。

    Returns:
        由 :func:`load_sensitive_config` 校验并复制的完整配置。
    """
    return load_sensitive_config(file_path)
