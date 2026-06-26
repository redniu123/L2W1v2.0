import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text_for_eval(text: object) -> str:
    if text is None:
        return ""
    return _WHITESPACE_RE.sub("", str(text).strip())
