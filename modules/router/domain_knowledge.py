#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: 轻量级多领域语义引擎

支持 geology / finance / medicine 三领域词典，按样本 domain 计算语义风险分 r_d。
特征公式：r_d = min(n_match / 3, 1.0)
"""

from pathlib import Path
from typing import Dict, Optional, Union


class DomainKnowledgeEngine:
    """基于 ahocorasick 的多领域语义风险引擎。"""

    def __init__(self, dict_source: Union[str, Dict[str, str]]):
        try:
            import ahocorasick
        except ImportError:
            raise ImportError("请安装 ahocorasick 库：pip install pyahocorasick")

        self._ahocorasick = ahocorasick
        self._automatons: Dict[str, object] = {}

        if isinstance(dict_source, dict):
            sources = dict_source
        else:
            sources = {"geology": dict_source}

        for domain, path_str in sources.items():
            dict_path = Path(path_str)
            if not dict_path.exists():
                raise FileNotFoundError(f"{domain} 词典不存在: {dict_path}")
            automaton = ahocorasick.Automaton()
            n_words = 0
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        automaton.add_word(word, word)
                        n_words += 1
            automaton.make_automaton()
            self._automatons[domain] = automaton
            print(f"[DomainKnowledgeEngine] 加载 {domain} 词典: {n_words} 个词条")

    def compute_r_d(self, text: str, domain: Optional[str] = None) -> float:
        """按 domain 计算语义风险分 r_d。未知 domain 自动回退 geology。"""
        if not text:
            return 0.0
        selected_domain = domain or "geology"
        automaton = self._automatons.get(selected_domain) or self._automatons.get("geology")
        if automaton is None:
            return 0.0
        n_match = 0
        for _, word in automaton.iter(text):
            n_match += len(word)
        return float(min(n_match / 3.0, 1.0))
