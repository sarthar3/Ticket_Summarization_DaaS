import json
import pytest
from app.preprocessing.dataset import DatasetSplitter

def test_dataset_splitter_ratios(tmp_path):
    records = [
        {"ticket_id": f"T{i:03d}", "ticket_text": f"Ticket text {i}", "summary": f"Summary {i}"}
        for i in range(100)
    ]
    splitter = DatasetSplitter(seed=42)
    train, val, test = splitter.split_dataset(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert len(train) + len(val) + len(test) == 100

def test_dataset_splitter_reproducibility():
    records = [
        {"ticket_id": f"T{i:03d}", "ticket_text": f"Ticket text {i}", "summary": f"Summary {i}"}
        for i in range(20)
    ]
    splitter_1 = DatasetSplitter(seed=42)
    train1, val1, test1 = splitter_1.split_dataset(records)

    splitter_2 = DatasetSplitter(seed=42)
    train2, val2, test2 = splitter_2.split_dataset(records)

    assert [r["ticket_id"] for r in train1] == [r["ticket_id"] for r in train2]
    assert [r["ticket_id"] for r in val1] == [r["ticket_id"] for r in val2]
    assert [r["ticket_id"] for r in test1] == [r["ticket_id"] for r in test2]

def test_dataset_splitter_save(tmp_path):
    records = [
        {"ticket_id": "T001", "ticket_text": "Sample text", "summary": "Sample summary"}
    ]
    splitter = DatasetSplitter(seed=42)
    output_file = tmp_path / "test_split.jsonl"
    splitter.save_split(records, output_file)

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        loaded = [json.loads(line) for line in f if line.strip()]
    assert len(loaded) == 1
    assert loaded[0]["ticket_id"] == "T001"
