"""
敏感词库：统一管理关键词、名单、组合规则，支持外部文件/配置动态加载

当前设计：代码内置 + 可选的配置文件覆盖
后续可扩展：从 Redis/数据库动态加载
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


SENSITIVE_CATEGORIES = ("political", "violence", "pornographic", "illegal")


# ── 统一敏感词库格式 ────────────────────────────────────────────────
# keywords: 分类关键词，命中直接拦截
# lists: 专用名单，供规则引用；political_persons 为严格直接拦截名单
# rules: 组合规则，任一 action + 任一 target 同时出现即拦截

_SENSITIVE_CORE: Dict[str, Any] = {
    "keywords": {
        "political": [
            "六四", "8964", "八九六四", "天安门事件", "天安门风波",
            "法轮功", "法轮大法", "falun", "falungong", "flg",
            "台独", "藏独", "疆独", "港独",
            "台湾独立", "西藏独立", "新疆独立", "香港独立",
            "分裂国家", "分裂主义", "民族分裂",
            "东突厥斯坦", "东突", "东伊运",
            "推翻", "颠覆", "颠覆政权", "煽动颠覆",
            "颜色革命", "街头革命", "暴力革命",
            "白纸革命", "四通桥",
            "港版国安法", "港独组织",
            "台湾建国", "台湾共和国",
        ],
        "violence": [
            "恐怖袭击", "恐怖主义", "恐怖分子", "极端组织",
            "炸弹制作", "制作炸弹", "爆炸物", "引爆装置", "遥控炸弹",
            "爆破", "引爆", "雷管",
            "枪击", "枪杀", "持枪", "砍人", "刺杀",
            "屠杀", "血洗", "灭门", "报复社会",
            "纵火", "放火", "投毒",
            "生化武器", "化学武器", "生物武器", "毒气",
            "人体炸弹", "自杀式袭击", "人肉炸弹",
            "袭警", "劫持", "绑架", "挟持人质",
        ],
        "pornographic": [
            "色情", "淫秽", "涉黄", "黄色内容",
            "裸聊", "裸照", "私密照", "成人视频",
            "成人影片", "成人网站", "毛片", "黄片",
            "一夜情", "约炮",
            "嫖娼", "卖淫", "援交",
            "性服务", "色情服务", "特殊服务", "上门服务",
            "儿童色情", "未成年色情", "儿童性剥削",
            "迷奸", "强奸", "性侵",
            "催情", "春药", "迷药",
            "操你妈", "操你", "你妈的", "草你妈", "艹你妈",
        ],
        "illegal": [
            "吸毒", "贩毒", "制毒", "运毒", "藏毒",
            "毒品",
            "海洛因", "冰毒", "大麻", "可卡因", "摇头丸",
            "罂粟", "k粉", "K粉", "麻古", "吗啡", "芬太尼",
            "赌场", "赌球", "赌马", "六合彩", "百家乐", "网赌",
            "博彩", "赌博", "棋牌赌博",
            "洗钱", "跑分", "资金盘", "走私",
            "人贩子", "拐卖", "买卖人口", "器官买卖",
            "高利贷", "裸贷", "套路贷", "校园贷",
            "假币", "假钞", "伪钞",
            "代孕", "非法代孕",
            "买凶", "买凶杀人", "雇凶",
            "暗网", "深网", "黑市",
            "电信诈骗", "网络诈骗", "杀猪盘",
            "盗号", "撞库", "脱库", "黑产",
            "办假证", "假证", "假身份证", "假学历",
        ],
    },
    "lists": {
        "political_persons": [],
        "political_activities": [],
        "political_activity_intents": [
            "支持", "宣传", "纪念", "悼念", "声援", "组织", "参与", "传播",
            "加入", "发起", "推动", "煽动", "动员", "串联",
            "怎么", "如何", "怎样", "办法", "方法", "教程", "攻略", "步骤",
        ],
    },
    "rules": {
        "political": [
            {
                "name": "political_ambiguous_number_context",
                "type": "keyword_context",
                "keywords": ["64"],
                "contexts": ["事件", "风波", "运动", "纪念", "悼念", "天安门"],
            },
            {
                "name": "political_person_direct",
                "type": "direct_list",
                "list": "political_persons",
            },
            {
                "name": "political_activity_intent",
                "type": "intent_list",
                "intents": "political_activity_intents",
                "list": "political_activities",
            },
            {
                "name": "political_separatism_action_target",
                "type": "action_target",
                "actions": ["分裂", "独立", "建国", "脱离", "颠覆", "推翻"],
                "targets": ["台湾", "香港", "西藏", "新疆", "国家", "政权"],
            },
            {
                "name": "political_sensitive_org_action_target",
                "type": "action_target",
                "actions": ["支持", "宣传", "纪念", "悼念", "声援", "组织", "参与", "传播", "加入"],
                "targets": ["台独", "港独", "藏独", "疆独", "法轮功", "法轮大法", "东突"],
            },
        ],
        "violence": [
            {
                "name": "violence_explosive_intent",
                "type": "keyword_context",
                "keywords": ["炸药"],
                "contexts": ["制作", "制造", "自制", "购买", "引爆", "投放", "配方", "教程"],
            },
            {
                "name": "violence_method_action_target",
                "type": "action_target",
                "actions": ["制作", "制造", "自制", "购买", "引爆", "投放"],
                "targets": ["炸弹", "炸药", "爆炸物", "毒气", "生化武器"],
            },
        ],
        "illegal": [
            {
                "name": "illegal_fraud_intent",
                "type": "keyword_context",
                "keywords": ["诈骗"],
                "contexts": ["实施", "组织", "参与", "教程", "方法", "话术", "骗钱", "团伙", "引流", "获利"],
            },
            {
                "name": "illegal_trade_action_target",
                "type": "action_target",
                "actions": ["购买", "售卖", "贩卖", "制作", "运输"],
                "targets": ["毒品", "冰毒", "海洛因", "可卡因", "假币", "假证"],
            },
        ],
    },
}


def _as_word_list(value: Any) -> List[str]:
    """把外部配置中的标量列表转换为可合并的字符串词表。

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


def get_default_sensitive_config() -> Dict[str, Any]:
    """获取可由调用方安全修改的默认敏感配置。

    Returns:
        内置关键词、名单和组合规则的深拷贝，修改结果不会污染模块常量。
    """
    return deepcopy(_SENSITIVE_CORE)


def get_default_sensitive_words() -> Dict[str, List[str]]:
    """获取旧调用方使用的分类关键词映射。

    Returns:
        内置 ``keywords`` 部分的深拷贝，不包含名单和组合规则。
    """
    return deepcopy(_SENSITIVE_CORE["keywords"])


def merge_sensitive_words(
    base: Dict[str, List[str]],
    additions: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """按分类合并旧格式敏感词并去重排序。

    Args:
        base: 基础的“分类到词表”映射。
        additions: 需要追加的同格式映射；可包含基础映射中不存在的分类。

    Returns:
        新建的合并映射。每类词表为两个输入集合的字典序排序并集，输入不会被修改。
    """
    result = {}
    for category in set(list(base.keys()) + list(additions.keys())):
        base_words = set(base.get(category, []))
        extra_words = set(additions.get(category, []))
        result[category] = sorted(base_words | extra_words)
    return result


def merge_sensitive_config(
    base: Dict[str, Any],
    additions: Dict[str, Any],
) -> Dict[str, Any]:
    """把新旧两种外部格式合并到统一敏感配置的深拷贝中。

    兼容旧外部 JSON：
        {"political": [...], "illegal": [...]}
    推荐外部 JSON：
        {
          "keywords": {"political": [...]},
          "lists": {"political_persons": [...], "political_activities": [...]},
          "rules": {"political": [...]}
        }

    Args:
        base: 包含 ``keywords``、``lists`` 和 ``rules`` 的基础配置。
        additions: 外部追加配置；非映射子项会被忽略。

    Returns:
        合并后的新配置。关键词和名单去重排序；合法规则字典按原顺序追加。
        ``base`` 与 ``additions`` 均不会被修改。
    """
    result = deepcopy(base)

    keyword_additions = additions.get("keywords", {})
    if not isinstance(keyword_additions, dict):
        keyword_additions = {}
    for category in SENSITIVE_CATEGORIES:
        legacy_words = additions.get(category, [])
        words = _as_word_list(keyword_additions.get(category, []))
        words.extend(_as_word_list(legacy_words))
        if words:
            current = set(result["keywords"].get(category, []))
            result["keywords"][category] = sorted(current | set(words))

    list_additions = additions.get("lists", {})
    if not isinstance(list_additions, dict):
        list_additions = {}
    for list_name in ("political_persons", "political_activities", "political_activity_intents"):
        words = _as_word_list(list_additions.get(list_name, []))
        words.extend(_as_word_list(additions.get(list_name, [])))
        if words:
            current = set(result["lists"].get(list_name, []))
            result["lists"][list_name] = sorted(current | set(words))

    rule_additions = additions.get("rules", {})
    if isinstance(rule_additions, dict):
        for category in SENSITIVE_CATEGORIES:
            rules = rule_additions.get(category, [])
            if isinstance(rules, list):
                result["rules"].setdefault(category, []).extend(
                    rule for rule in rules if isinstance(rule, dict)
                )

    return result
