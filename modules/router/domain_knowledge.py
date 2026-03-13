#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: 轻量级领域语义引擎

策略：ahocorasick 多模式匹配地质词典，计算语义风险分 r_d。

特征公式：r_d = min(n_match / 3, 1.0)
  其中 n_match 为识别文本中命中地质词典词条的总字符数。
"""

import sys
from pathlib import Path
from typing import Optional


class DomainKnowledgeEngine:
    """
    基于 ahocorasick 的地质领域语义风险引擎

    Args:
        dict_path: 地质词典路径（每行一个词条）
    """

    def __init__(self, dict_path: str):
        try:
            import ahocorasick
        except ImportError:
            raise ImportError(
                "请安装 ahocorasick 库：pip install pyahocorasick"
            )

        self._automaton = ahocorasick.Automaton()
        dict_path = Path(dict_path)

        if not dict_path.exists():
            raise FileNotFoundError(f"地质词典不存在: {dict_path}")

        n_words = 0
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    self._automaton.add_word(word, word)
                    n_words += 1

        self._automaton.make_automaton()
        print(f"[DomainKnowledgeEngine] 加载地质词典: {n_words} 个词条")

    def compute_r_d(self, text: str) -> float:
        """
        计算语义风险分 r_d

        Args:
            text: Agent A 识别文本

        Returns:
            r_d = min(n_match / 3, 1.0)
            其中 n_match 为命中词条的总字符数
        """
        if not text:
            return 0.0

        n_match = 0
        for _, word in self._automaton.iter(text):
            n_match += len(word)

        r_d = min(n_match / 3.0, 1.0)
        return float(r_d)
