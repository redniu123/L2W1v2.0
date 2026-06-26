import pytest

from l2w1.io.cache import index_rows_by_sample_id


def test_index_rows_by_sample_id_default_key():
    rows = [
        {"sample_id": "s1", "value": 1},
        {"sample_id": "s2", "value": 2},
    ]

    indexed = index_rows_by_sample_id(rows)

    assert indexed["s1"]["value"] == 1
    assert indexed["s2"]["value"] == 2


def test_index_rows_by_sample_id_duplicate_error():
    rows = [
        {"sample_id": "s1", "value": 1},
        {"sample_id": "s1", "value": 2},
    ]

    with pytest.raises(ValueError, match="duplicate sample_id"):
        index_rows_by_sample_id(rows)


def test_index_rows_by_sample_id_duplicate_first():
    rows = [
        {"sample_id": "s1", "value": 1},
        {"sample_id": "s1", "value": 2},
    ]

    indexed = index_rows_by_sample_id(rows, on_duplicate="first")

    assert indexed["s1"]["value"] == 1


def test_index_rows_by_sample_id_duplicate_last():
    rows = [
        {"sample_id": "s1", "value": 1},
        {"sample_id": "s1", "value": 2},
    ]

    indexed = index_rows_by_sample_id(rows, on_duplicate="last")

    assert indexed["s1"]["value"] == 2


def test_index_rows_by_sample_id_rejects_unknown_duplicate_policy():
    with pytest.raises(ValueError, match="on_duplicate"):
        index_rows_by_sample_id([], on_duplicate="replace")


def test_index_rows_by_sample_id_missing_key_raises():
    with pytest.raises(KeyError, match="missing required key"):
        index_rows_by_sample_id([{"value": 1}])


def test_index_rows_by_custom_key():
    rows = [{"id": "a", "value": 1}]

    indexed = index_rows_by_sample_id(rows, key="id")

    assert indexed["a"]["value"] == 1
