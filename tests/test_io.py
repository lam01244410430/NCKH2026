import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import BaseModel

from hsk_agent_bench.io import verify_shard, write_jsonl_shard_atomic
import hsk_agent_bench.io as shard_io


class Item(BaseModel):
    text: str


def test_write_utf8_jsonl_and_checksum(tmp_path):
    target = tmp_path / "shard.jsonl"
    sidecar = write_jsonl_shard_atomic([Item(text="你好"), Item(text="世界")], target)
    assert target.read_bytes() == '{"text":"你好"}\n{"text":"世界"}\n'.encode("utf-8")
    assert sidecar == Path(str(target) + ".sha256")
    assert sidecar.read_text().strip() == hashlib.sha256(target.read_bytes()).hexdigest()
    assert verify_shard(target)


def test_refuses_overwrite_and_keeps_original_pair(tmp_path):
    target = tmp_path / "shard.jsonl"
    sidecar = write_jsonl_shard_atomic([Item(text="first")], target)
    original = target.read_bytes(), sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        write_jsonl_shard_atomic([Item(text="second")], target)
    assert (target.read_bytes(), sidecar.read_bytes()) == original


def test_generator_failure_removes_all_temporary_files(tmp_path):
    def broken_records():
        yield Item(text="partial")
        raise RuntimeError("simulated interruption")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        write_jsonl_shard_atomic(broken_records(), tmp_path / "shard.jsonl")
    assert list(tmp_path.iterdir()) == []


def test_tampering_or_missing_or_malformed_checksum_is_false(tmp_path):
    target = tmp_path / "shard.jsonl"
    assert not verify_shard(target)
    sidecar = write_jsonl_shard_atomic([Item(text="original")], target)
    target.write_text("tampered\n")
    assert not verify_shard(target)
    sidecar.write_text("garbage")
    assert not verify_shard(target)
    sidecar.unlink()
    assert not verify_shard(target)


def test_orphan_checksum_is_not_overwritten(tmp_path):
    target = tmp_path / "shard.jsonl"
    sidecar = Path(str(target) + ".sha256")
    sidecar.write_text("existing")
    with pytest.raises(FileExistsError):
        write_jsonl_shard_atomic([Item(text="new")], target)
    assert sidecar.read_text() == "existing"
    assert not target.exists()


def test_concurrent_writers_publish_exactly_one_complete_pair(tmp_path):
    target = tmp_path / "shard.jsonl"
    def write(text):
        try:
            write_jsonl_shard_atomic([Item(text=text)], target)
            return True
        except FileExistsError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, ["first", "second"]))
    assert sorted(results) == [False, True]
    assert verify_shard(target)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["shard.jsonl", "shard.jsonl.sha256"]


def test_checksum_publication_failure_rolls_back_owned_shard(tmp_path, monkeypatch):
    publish = shard_io._publish
    def fail_checksum(source, target):
        if str(target).endswith(".sha256"):
            raise OSError("simulated disk failure")
        publish(source, target)
    monkeypatch.setattr(shard_io, "_publish", fail_checksum)
    with pytest.raises(OSError, match="simulated disk failure"):
        write_jsonl_shard_atomic([Item(text="partial")], tmp_path / "shard.jsonl")
    assert list(tmp_path.iterdir()) == []
