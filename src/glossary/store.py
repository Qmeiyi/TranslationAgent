#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glossary存储与冲突解决模块
C同学实现：冲突解决、版本化、多义词处理、候选翻译评分
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.config import get_llm


class GlossaryStore:
    """术语表存储与冲突解决管理器"""

    def __init__(self, glossary_path: Optional[Path] = None):
        self.glossary_path = glossary_path
        self.glossary: Dict[str, Any] = {"version": 1, "entries": []}
        if glossary_path and glossary_path.exists():
            self.load(glossary_path)

    def load(self, path: Path) -> None:
        """加载术语表"""
        data = json.loads(path.read_text(encoding="utf-8"))
        self.glossary = data
        if "version" not in self.glossary:
            self.glossary["version"] = 1

    def save(self, path: Optional[Path] = None) -> Path:
        """保存术语表，自动版本号+1，文件名格式：glossary_v{version}.json"""
        target_path = path or self.glossary_path
        if target_path is None:
            raise ValueError("No path specified for saving glossary")

        # 版本号递增
        current_version = self.glossary.get("version", 1)
        new_version = current_version + 1
        self.glossary["version"] = new_version

        # 如果路径是glossary.json，改为glossary_v{version}.json格式
        if target_path.name == "glossary.json":
            versioned_path = target_path.parent / f"glossary_v{new_version}.json"
        else:
            # 如果已经是版本化文件名，更新版本号
            versioned_path = target_path.parent / f"glossary_v{new_version}.json"

        versioned_path.parent.mkdir(parents=True, exist_ok=True)
        versioned_path.write_text(
            json.dumps(self.glossary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return versioned_path

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """从文本中提取关键词（简单实现：基于长度和频率）"""
        # 移除标点，分词
        words = re.findall(r"\b\w+\b", text.lower())
        # 过滤停用词和短词
        stopwords = {"的", "了", "在", "是", "和", "有", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        words = [w for w in words if len(w) > 1 and w not in stopwords]
        # 统计频率
        from collections import Counter
        counter = Counter(words)
        return [word for word, _ in counter.most_common(max_keywords)]

    def _compute_context_signature(self, evidence_span: str) -> str:
        """计算上下文签名（用于判断上下文相似性）"""
        keywords = self._extract_keywords(evidence_span)
        return "|".join(sorted(keywords))

    def _are_contexts_similar(
        self, sig1: str, sig2: str, threshold: float = 0.4
    ) -> bool:
        """判断两个上下文签名是否相似"""
        if not sig1 or not sig2:
            return False
        words1 = set(sig1.split("|"))
        words2 = set(sig2.split("|"))
        if not words1 or not words2:
            return False
        intersection = words1 & words2
        union = words1 | words2
        similarity = len(intersection) / len(union) if union else 0.0
        return similarity >= threshold

    def _score_candidate(
        self, candidate: str, frequency: int, source: str, has_evidence: bool
    ) -> float:
        """为候选翻译评分"""
        score = 0.0
        # 频率加分
        score += min(frequency * 0.1, 0.5)
        # 来源加分
        if source == "extraction":
            score += 0.2
        elif source == "manual":
            score += 0.4
        # 有证据加分
        if has_evidence:
            score += 0.1
        return min(score, 1.0)

    def add_terms(
        self, new_terms: List[Dict[str, Any]], use_llm_disambiguation: bool = False
    ) -> Dict[str, Any]:
        """
        添加新术语，自动处理冲突
        
        Args:
            new_terms: 新提取的术语列表，每个术语包含term, type, candidates, evidence_span等
            use_llm_disambiguation: 是否使用LLM进行消歧（否则使用关键词规则）
        
        Returns:
            冲突解决报告
        """
        conflicts = []
        merged = []

        # 按术语分组
        term_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for term in new_terms:
            term_groups[term.get("term", "")].append(term)

        # 处理每个术语
        for term_text, occurrences in term_groups.items():
            existing_entry = self._find_entry(term_text)

            if existing_entry is None:
                # 新术语，直接添加
                entry = self._create_entry_from_extraction(occurrences[0])
                self.glossary["entries"].append(entry)
                merged.append(term_text)
            else:
                # 存在冲突，需要解决
                conflict_result = self._resolve_conflict(
                    existing_entry, occurrences, use_llm_disambiguation
                )
                conflicts.append(conflict_result)
                if conflict_result["action"] == "merged":
                    merged.append(term_text)
                elif conflict_result["action"] == "split":
                    merged.append(term_text)

        return {
            "conflicts": conflicts,
            "merged": merged,
            "total_new": len(new_terms),
        }

    def _find_entry(self, term: str) -> Optional[Dict[str, Any]]:
        """查找现有条目"""
        for entry in self.glossary.get("entries", []):
            if entry.get("term") == term:
                return entry
            # 检查别名
            if term in entry.get("aliases", []):
                return entry
        return None

    def _create_entry_from_extraction(self, term_data: Dict[str, Any]) -> Dict[str, Any]:
        """从提取结果创建条目"""
        candidates = term_data.get("candidates", [])
        candidate_list = []
        for i, cand in enumerate(candidates):
            candidate_list.append(
                {
                    "translation": cand if isinstance(cand, str) else cand.get("translation", ""),
                    "score": 1.0 - (i * 0.1),
                    "source": "extraction",
                }
            )

        return {
            "term": term_data.get("term", ""),
            "type": term_data.get("type", ""),
            "final": candidate_list[0]["translation"] if candidate_list else "",
            "aliases": [],
            "senses": [],
            "candidates": candidate_list,
            "evidence_span": term_data.get("evidence_span"),
            "score": None,
        }

    def _resolve_conflict(
        self,
        existing_entry: Dict[str, Any],
        new_occurrences: List[Dict[str, Any]],
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        解决冲突：合并或拆分义项
        
        Returns:
            {
                "action": "merged" | "split" | "kept_existing",
                "details": ...
            }
        """
        existing_type = existing_entry.get("type", "")
        existing_final = existing_entry.get("final", "")
        existing_evidence = existing_entry.get("evidence_span", "")

        # 对于命名实体（NE），默认强制唯一翻译
        if existing_type.startswith("NE:"):
            # 检查新出现的翻译是否与现有final一致
            all_new_translations = []
            for occ in new_occurrences:
                all_new_translations.extend(occ.get("candidates", []))

            # 如果新翻译与现有final不一致，记录冲突但保持现有翻译
            if existing_final and existing_final not in all_new_translations:
                # 更新候选列表，但保持final不变
                for occ in new_occurrences:
                    for cand in occ.get("candidates", []):
                        cand_str = cand if isinstance(cand, str) else cand.get("translation", "")
                        if cand_str not in [c["translation"] for c in existing_entry.get("candidates", [])]:
                            existing_entry.setdefault("candidates", []).append({
                                "translation": cand_str,
                                "score": 0.5,
                                "source": "conflict",
                            })

                return {
                    "action": "kept_existing",
                    "term": existing_entry.get("term"),
                    "reason": "NE must have unique translation",
                    "existing": existing_final,
                    "new_candidates": all_new_translations,
                }

            # 如果一致，合并候选并更新评分
            return self._merge_similar_translations(existing_entry, new_occurrences)

        # 对于非NE术语，检查上下文相似性
        existing_sig = self._compute_context_signature(existing_evidence) if existing_evidence else ""

        all_similar = True
        for occ in new_occurrences:
            new_evidence = occ.get("evidence_span", "")
            new_sig = self._compute_context_signature(new_evidence) if new_evidence else ""
            if not self._are_contexts_similar(existing_sig, new_sig):
                all_similar = False
                break

        if all_similar:
            # 上下文相似，合并翻译
            return self._merge_similar_translations(existing_entry, new_occurrences)
        else:
            # 上下文差异大，拆分为不同义项
            return self._split_into_senses(existing_entry, new_occurrences, use_llm)

    def _merge_similar_translations(
        self, existing_entry: Dict[str, Any], new_occurrences: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """合并相似翻译"""
        # 收集所有候选翻译
        translation_freq: Dict[str, int] = defaultdict(int)
        for occ in new_occurrences:
            for cand in occ.get("candidates", []):
                cand_str = cand if isinstance(cand, str) else cand.get("translation", "")
                translation_freq[cand_str] += 1

        # 更新候选列表
        existing_candidates = {c["translation"]: c for c in existing_entry.get("candidates", [])}
        for trans, freq in translation_freq.items():
            if trans in existing_candidates:
                existing_candidates[trans]["score"] += 0.1 * freq
            else:
                existing_candidates[trans] = {
                    "translation": trans,
                    "score": 0.5 + (0.1 * freq),
                    "source": "extraction",
                }

        # 选择最高分的作为final
        best_candidate = max(existing_candidates.values(), key=lambda c: c.get("score", 0))
        existing_entry["final"] = best_candidate["translation"]
        existing_entry["candidates"] = list(existing_candidates.values())

        return {
            "action": "merged",
            "term": existing_entry.get("term"),
            "new_final": existing_entry["final"],
        }

    def _split_into_senses(
        self,
        existing_entry: Dict[str, Any],
        new_occurrences: List[Dict[str, Any]],
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        """拆分为不同义项"""
        if use_llm:
            return self._split_into_senses_llm(existing_entry, new_occurrences)
        else:
            return self._split_into_senses_keyword(existing_entry, new_occurrences)

    def _split_into_senses_keyword(
        self,
        existing_entry: Dict[str, Any],
        new_occurrences: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """基于关键词规则拆分义项"""
        term_text = existing_entry.get("term", "")

        # 创建新义项
        sense_id = len(existing_entry.get("senses", [])) + 1
        new_sense = {
            "sense_id": f"{term_text}#{sense_id}",
            "context_signature": self._compute_context_signature(
                new_occurrences[0].get("evidence_span", "")
            ),
            "final": new_occurrences[0].get("candidates", [""])[0]
            if new_occurrences[0].get("candidates")
            else "",
            "evidence_span": new_occurrences[0].get("evidence_span"),
        }

        existing_entry.setdefault("senses", []).append(new_sense)

        return {
            "action": "split",
            "term": term_text,
            "sense_id": new_sense["sense_id"],
            "new_final": new_sense["final"],
        }

    def _split_into_senses_llm(
        self,
        existing_entry: Dict[str, Any],
        new_occurrences: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """使用LLM进行消歧并拆分义项"""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        term_text = existing_entry.get("term", "")
        existing_evidence = existing_entry.get("evidence_span", "")
        new_evidence = new_occurrences[0].get("evidence_span", "")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一位术语专家。请判断两个上下文中的术语是否表示不同含义。",
                ),
                (
                    "user",
                    f"术语：{term_text}\n\n"
                    f"现有上下文：{existing_evidence}\n\n"
                    f"新上下文：{new_evidence}\n\n"
                    "请判断这是同一个含义还是不同含义。如果不同，请简要说明区别。",
                ),
            ]
        )

        llm = get_llm(temperature=0.1)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({})

        # 简单判断：如果LLM明确说"不同"，则拆分
        should_split = "不同" in result or "different" in result.lower()

        if should_split:
            return self._split_into_senses_keyword(existing_entry, new_occurrences)
        else:
            return self._merge_similar_translations(existing_entry, new_occurrences)

    def get_relevant_terms(
        self, text: str, max_terms: int = 20
    ) -> List[Dict[str, Any]]:
        """根据文本检索相关术语（用于翻译时注入）"""
        text_lower = text.lower()
        relevant = []

        for entry in self.glossary.get("entries", []):
            term = entry.get("term", "")
            if term in text or term.lower() in text_lower:
                relevant.append(entry)

        # 按类型和重要性排序
        ne_terms = [e for e in relevant if e.get("type", "").startswith("NE:")]
        other_terms = [e for e in relevant if not e.get("type", "").startswith("NE:")]

        return (ne_terms + other_terms)[:max_terms]

    def check_violations(
        self, original_text: str, translated_text: str
    ) -> List[Dict[str, Any]]:
        """
        检查翻译是否违反术语一致性
        
        Returns:
            违规列表，每个违规包含term, expected, found, severity等
        """
        violations = []
        relevant_terms = self.get_relevant_terms(original_text)

        for entry in relevant_terms:
            term = entry.get("term", "")
            expected = entry.get("final", "")

            if not expected or term not in original_text:
                continue

            # 检查翻译中是否使用了expected
            if expected.lower() not in translated_text.lower():
                # 检查是否有其他候选翻译被使用
                candidates = [c["translation"] for c in entry.get("candidates", [])]
                found_any = any(cand.lower() in translated_text.lower() for cand in candidates if cand)

                if not found_any:
                    violations.append(
                        {
                            "term": term,
                            "expected": expected,
                            "found": "NOT_FOUND",
                            "severity": "high" if entry.get("type", "").startswith("NE:") else "medium",
                            "type": entry.get("type"),
                        }
                    )
                else:
                    # 使用了候选但不是final，标记为警告
                    violations.append(
                        {
                            "term": term,
                            "expected": expected,
                            "found": "WRONG_CANDIDATE",
                            "severity": "low",
                            "type": entry.get("type"),
                        }
                    )

        return violations

    def format_for_prompt(self, terms: Optional[List[Dict[str, Any]]] = None) -> str:
        """格式化术语表用于注入到翻译prompt"""
        if terms is None:
            terms = self.glossary.get("entries", [])

        if not terms:
            return "No glossary available."

        lines = ["## 强制术语表 (Strict Glossary):"]
        lines.append("**必须严格遵守以下术语翻译，不得随意更改：**\n")

        for entry in terms:
            term = entry.get("term", "")
            final = entry.get("final", "")
            term_type = entry.get("type", "")
            if final:
                lines.append(f"- **{term}** → {final} ({term_type})")

        world_summary = self.glossary.get("world_summary")
        if world_summary:
            lines.append(f"\n## 世界观背景:\n{world_summary}")

        return "\n".join(lines)

