import json
import pytest
from pathlib import Path
from src.dataloader import load_txt, load_csv, load_json, load_single_file


def test_load_txt(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello RAG world!", encoding="utf-8")

    docs = load_txt(file_path)
    assert len(docs) == 1
    assert docs[0].page_content == "Hello RAG world!"
    assert docs[0].metadata["source"] == "test.txt"
    assert docs[0].metadata["type"] == "txt"


def test_load_csv(tmp_path: Path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")

    docs = load_csv(file_path)
    assert len(docs) == 1
    assert "Row 1: name: Alice, age: 30" in docs[0].page_content
    assert "Row 2: name: Bob, age: 25" in docs[0].page_content
    assert docs[0].metadata["type"] == "csv"
    assert docs[0].metadata["total_rows"] == 2


def test_load_json(tmp_path: Path):
    file_path = tmp_path / "test.json"
    data = {"key": "value", "list": [1, 2, 3]}
    file_path.write_text(json.dumps(data), encoding="utf-8")

    docs = load_json(file_path)
    assert len(docs) == 1
    assert '"key": "value"' in docs[0].page_content
    assert docs[0].metadata["type"] == "json"


def test_load_single_file_unsupported(tmp_path: Path):
    file_path = tmp_path / "test.unsupported"
    file_path.write_text("some content", encoding="utf-8")

    docs = load_single_file(file_path)
    assert docs == []
