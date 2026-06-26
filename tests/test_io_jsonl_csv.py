import json

import pytest

from l2w1.io.csv_io import read_csv_rows, write_csv_rows
from l2w1.io.jsonl import iter_jsonl, read_jsonl, write_jsonl

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def test_read_jsonl_fixture_supports_unicode_and_blank_line_skipping(tmp_path):
    rows = read_jsonl(FIXTURES / "tiny_rows.jsonl")

    assert len(rows) == 3
    assert rows[0]["sample_id"] == "s1"
    assert rows[0]["gt_text"] == "中国地质"
    assert rows[2]["ocr_text"] == " 中国\t金融\n"

    path = tmp_path / "rows_with_blank.jsonl"
    path.write_text(
        json.dumps(rows[0], ensure_ascii=False)
        + "\n\n"
        + json.dumps(rows[1], ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    assert [row["sample_id"] for row in iter_jsonl(path)] == ["s1", "s2"]


def test_iter_jsonl_raises_json_decode_error(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"sample_id": "s1"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        list(iter_jsonl(path))


def test_write_jsonl_roundtrip_creates_parent(tmp_path):
    rows = read_jsonl(FIXTURES / "tiny_rows.jsonl")
    out_path = tmp_path / "nested" / "roundtrip.jsonl"

    write_jsonl(out_path, rows)

    assert out_path.exists()
    assert read_jsonl(out_path) == rows
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 3


def test_read_csv_fixture_and_write_roundtrip(tmp_path):
    rows = read_csv_rows(FIXTURES / "tiny_csv_rows.csv")
    out_path = tmp_path / "nested" / "roundtrip.csv"

    write_csv_rows(out_path, rows)

    assert len(rows) == 3
    assert rows[0]["gt_text"] == "中国地质"
    assert read_csv_rows(out_path) == rows


def test_write_csv_empty_rows_without_fieldnames_writes_empty_file(tmp_path):
    out_path = tmp_path / "empty.csv"

    write_csv_rows(out_path, [], fieldnames=None)

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == ""


def test_write_csv_empty_rows_with_fieldnames_writes_header(tmp_path):
    out_path = tmp_path / "empty_with_header.csv"

    write_csv_rows(out_path, [], fieldnames=["sample_id", "gt_text"])

    assert out_path.read_text(encoding="utf-8").splitlines()[0] == "sample_id,gt_text"
    assert read_csv_rows(out_path) == []
