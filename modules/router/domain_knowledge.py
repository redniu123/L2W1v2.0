#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geology domain knowledge detection for SH-DA++ v4.0

目标：
- 加载地质词典并进行高性能检索
- 识别精准命中与模糊命中（缺字/错字）风险
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


class GeologyKnowledge:
    """地质领域知识库检测器"""

    def __init__(self, dict_path: str, min_len: int = 2) -> None:
        self.dict_path = Path(dict_path)
        self.min_len = max(2, int(min_len))
        self._terms: Set[str] = set()
        self._terms_by_len: Dict[int, Set[str]] = defaultdict(set)
        self._pattern_map: Dict[int, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._delete_map: Dict[int, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._lengths: List[int] = []
        self._loaded: bool = False
        self._load_terms()

    def _load_terms(self) -> None:
        if not self.dict_path.exists():
            self._loaded = False
            return

        with self.dict_path.open("r", encoding="utf-8") as f:
            for raw in f:
                term = raw.strip()
                if not term:
                    continue
                if len(term) < self.min_len:
                    continue
                if term in self._terms:
                    continue
                self._terms.add(term)
                self._terms_by_len[len(term)].add(term)

        # 预计算模糊匹配索引（1 字符替换 / 1 字符删除）
        for term in self._terms:
            t_len = len(term)
            for i in range(t_len):
                # 替换 1 个字符的通配模式
                pattern = f"{term[:i]}*{term[i + 1:]}"
                self._pattern_map[t_len][pattern].add(term)
                # 删除 1 个字符的变体
                if t_len - 1 >= self.min_len:
                    deleted = f"{term[:i]}{term[i + 1:]}"
                    self._delete_map[t_len - 1][deleted].add(term)

        self._lengths = sorted(self._terms_by_len.keys())
        self._loaded = True

    def detect_geology_risk(self, text: str) -> Tuple[float, Dict]:
        """
        检测地质领域风险

        返回:
            risk_score: float (0.0/0.3/0.9)
            details: dict (命中词与匹配类型)
        """
        if not self._loaded or not text:
            return 0.0, {"exact_hits": [], "fuzzy_hits": []}

        normalized = text.replace(" ", "")
        if len(normalized) < self.min_len:
            return 0.0, {"exact_hits": [], "fuzzy_hits": []}

        exact_hits = self._find_exact_hits(normalized)
        fuzzy_hits = self._find_fuzzy_hits(normalized)

        if fuzzy_hits:
            score = 0.9
        elif exact_hits:
            score = 0.3
        else:
            score = 0.0

        return score, {
            "exact_hits": sorted(exact_hits),
            "fuzzy_hits": sorted(fuzzy_hits),
        }

    def _find_exact_hits(self, text: str) -> Set[str]:
        hits: Set[str] = set()
        n = len(text)
        for length in self._lengths:
            if n < length:
                continue
            term_set = self._terms_by_len[length]
            for i in range(n - length + 1):
                substr = text[i : i + length]
                if substr in term_set:
                    hits.add(substr)
        return hits

    def _find_fuzzy_hits(self, text: str) -> Set[str]:
        hits: Set[str] = set()
        n = len(text)

        # 1) 1 字符替换（同长度）
        for length in self._lengths:
            if n < length:
                continue
            pattern_map = self._pattern_map.get(length, {})
            if not pattern_map:
                continue
            for i in range(n - length + 1):
                substr = text[i : i + length]
                # 生成通配模式并命中
                for j in range(length):
                    pattern = f"{substr[:j]}*{substr[j + 1:]}"
                    if pattern in pattern_map:
                        hits.update(pattern_map[pattern])
                        if len(hits) >= 20:
                            return hits

        # 2) 1 字符缺失（长度 -1）
        for length in self._lengths:
            target_len = length - 1
            if target_len < self.min_len or n < target_len:
                continue
            delete_map = self._delete_map.get(target_len, {})
            if not delete_map:
                continue
            for i in range(n - target_len + 1):
                substr = text[i : i + target_len]
                if substr in delete_map:
                    hits.update(delete_map[substr])
                    if len(hits) >= 20:
                        return hits

        return hits
