from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import Levenshtein

_FORMAT_EQUIVALENCE = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "｛": "{",
        "｝": "}",
        "，": ",",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "。": ".",
    }
)


class RouteType(Enum):
    """Route type."""

    NONE = "NONE"
    BOUNDARY = "BOUNDARY"
    AMBIGUITY = "AMBIGUITY"
    BOTH = "BOTH"


class RejectionReason(Enum):
    """Rejection reason."""

    ACCEPTED = "accepted"
    GLOBAL_ED_EXCEEDED = "rejected_global_ed_exceeded"
    GLOBAL_LENGTH_CHANGE = "rejected_global_length_change"
    PURE_FORMATTING_EDIT = "rejected_pure_formatting_edit"
    # Kept for compatibility; v5.1 unified mode does not trigger these.
    BOUNDARY_VIOLATION = "rejected_boundary_violation"
    AMBIGUITY_VIOLATION = "rejected_ambiguity_violation"
    TOP2_MISMATCH = "rejected_top2_mismatch"
    MULTIPLE_CHANGES = "rejected_multiple_changes"


@dataclass
class BackfillConfig:
    """Backfill configuration (v5.1 keeps only global red lines)."""

    strict_mode: bool = True
    max_edit_distance: int = 3
    max_length_change_ratio: float = 0.2
    boundary_K: int = 2
    unified_prompt_mode: bool = True
    reject_pure_formatting_edit: bool = True


@dataclass
class BackfillResult:
    """Backfill result."""

    T_final: str
    is_rejected: bool
    rejection_reason: RejectionReason
    edit_distance: int
    length_change_ratio: float


class StrictBackfillController:
    """Strict backfill controller (v5.1)."""

    def __init__(self, config: BackfillConfig | None = None):
        self.config = config or BackfillConfig()

    def apply_backfill(
        self,
        T_A: str,
        T_cand: str,
        route_type: RouteType,
        idx_susp: int | None = None,
        top2_chars: list[str] | None = None,
    ) -> BackfillResult:
        """
        Apply strict backfill rules.

        v5.1 logic:
          1. Global red lines (ED>3 or length change>20%).
          2. Reject pure formatting normalization edits.
          3. Skip route-specific constraints when unified_prompt_mode=True.
        """
        if not self.config.strict_mode:
            return BackfillResult(
                T_final=T_cand,
                is_rejected=False,
                rejection_reason=RejectionReason.ACCEPTED,
                edit_distance=Levenshtein.distance(T_A, T_cand),
                length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
            )

        rejection = self._check_global_rejection(T_A, T_cand)
        if rejection is not None:
            return rejection

        formatting_rejection = self._check_pure_formatting_edit(T_A, T_cand)
        if formatting_rejection is not None:
            return formatting_rejection

        if not self.config.unified_prompt_mode:
            path_rejection = None
            if route_type == RouteType.BOUNDARY:
                path_rejection = self._check_boundary_constraint(T_A, T_cand)
            elif route_type == RouteType.AMBIGUITY:
                path_rejection = self._check_ambiguity_constraint(T_A, T_cand, idx_susp, top2_chars)
            if path_rejection is not None:
                return path_rejection

        return BackfillResult(
            T_final=T_cand,
            is_rejected=False,
            rejection_reason=RejectionReason.ACCEPTED,
            edit_distance=Levenshtein.distance(T_A, T_cand),
            length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
        )

    def _check_global_rejection(self, T_A: str, T_cand: str) -> BackfillResult | None:
        ed = Levenshtein.distance(T_A, T_cand)
        len_change_ratio = self._compute_length_change_ratio(T_A, T_cand)

        if ed > self.config.max_edit_distance:
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.GLOBAL_ED_EXCEEDED,
                edit_distance=ed,
                length_change_ratio=len_change_ratio,
            )

        if len_change_ratio > self.config.max_length_change_ratio:
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.GLOBAL_LENGTH_CHANGE,
                edit_distance=ed,
                length_change_ratio=len_change_ratio,
            )

        return None

    def _check_pure_formatting_edit(self, T_A: str, T_cand: str) -> BackfillResult | None:
        if not self.config.reject_pure_formatting_edit:
            return None
        if not T_cand or T_cand == T_A:
            return None
        if self._normalize_format_equivalence(T_A) != self._normalize_format_equivalence(T_cand):
            return None
        return BackfillResult(
            T_final=T_A,
            is_rejected=True,
            rejection_reason=RejectionReason.PURE_FORMATTING_EDIT,
            edit_distance=Levenshtein.distance(T_A, T_cand),
            length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
        )

    def _check_boundary_constraint(self, T_A: str, T_cand: str) -> BackfillResult | None:
        K = self.config.boundary_K
        N = len(T_A)
        ops = Levenshtein.editops(T_A, T_cand)

        for op_type, pos_A, pos_cand in ops:
            _ = pos_cand
            if op_type in ["replace", "delete"]:
                if not (pos_A < K or pos_A >= N - K):
                    return BackfillResult(
                        T_final=T_A,
                        is_rejected=True,
                        rejection_reason=RejectionReason.BOUNDARY_VIOLATION,
                        edit_distance=Levenshtein.distance(T_A, T_cand),
                        length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
                    )
            elif op_type == "insert":
                if not (pos_A <= K or pos_A >= N - K):
                    return BackfillResult(
                        T_final=T_A,
                        is_rejected=True,
                        rejection_reason=RejectionReason.BOUNDARY_VIOLATION,
                        edit_distance=Levenshtein.distance(T_A, T_cand),
                        length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
                    )
        return None

    def _check_ambiguity_constraint(
        self,
        T_A: str,
        T_cand: str,
        idx_susp: int | None,
        top2_chars: list[str] | None,
    ) -> BackfillResult | None:
        if idx_susp is None or top2_chars is None:
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.AMBIGUITY_VIOLATION,
                edit_distance=Levenshtein.distance(T_A, T_cand),
                length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
            )

        if len(T_A) != len(T_cand):
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.AMBIGUITY_VIOLATION,
                edit_distance=Levenshtein.distance(T_A, T_cand),
                length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
            )

        changed_positions = [i for i, (a, b) in enumerate(zip(T_A, T_cand)) if a != b]

        if len(changed_positions) > 1:
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.MULTIPLE_CHANGES,
                edit_distance=Levenshtein.distance(T_A, T_cand),
                length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
            )

        if len(changed_positions) == 1:
            changed_pos = changed_positions[0]
            if changed_pos != idx_susp:
                return BackfillResult(
                    T_final=T_A,
                    is_rejected=True,
                    rejection_reason=RejectionReason.AMBIGUITY_VIOLATION,
                    edit_distance=Levenshtein.distance(T_A, T_cand),
                    length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
                )
            new_char = T_cand[changed_pos]
            if new_char not in top2_chars:
                return BackfillResult(
                    T_final=T_A,
                    is_rejected=True,
                    rejection_reason=RejectionReason.TOP2_MISMATCH,
                    edit_distance=Levenshtein.distance(T_A, T_cand),
                    length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
                )

        return None

    def _compute_length_change_ratio(self, T_A: str, T_cand: str) -> float:
        if len(T_A) == 0:
            return 0.0
        return abs(len(T_cand) - len(T_A)) / len(T_A)

    def _normalize_format_equivalence(self, text: str) -> str:
        if not text:
            return ""
        return text.translate(_FORMAT_EQUIVALENCE)


def apply_strict_backfill(
    T_A: str,
    T_cand: str,
    route_type: RouteType,
    idx_susp: int | None = None,
    top2_chars: list[str] | None = None,
    config: BackfillConfig | None = None,
) -> BackfillResult:
    """Apply strict backfill."""
    if config is None:
        config = BackfillConfig()
    controller = StrictBackfillController(config)
    return controller.apply_backfill(T_A, T_cand, route_type, idx_susp, top2_chars)
