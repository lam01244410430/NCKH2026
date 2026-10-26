"""Immutable UTF-8 JSONL shards with atomic publication and SHA-256 sidecars.

A verified sidecar is the completion marker. Two directory entries cannot be
published in one filesystem transaction; readers must call verify_shard before
consuming a shard. A killed process may leave an incomplete pair or lock, which
requires inspection rather than automatic overwrite on resume.
"""

import hashlib
import hmac
import os
import re
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel


@contextmanager
def _exclusive_lock(path: Path):
    handle = path.open("x", encoding="ascii")
    try:
        yield
    finally:
        handle.close()
        path.unlink()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _publish(source: Path, target: Path) -> None:
    """Atomically publish without clobbering an existing destination."""
    if os.name == "nt":
        # Windows rename refuses an existing destination (unlike POSIX rename).
        os.rename(source, target)
    else:
        # A hard link is the POSIX atomic no-replace equivalent on this sibling FS.
        os.link(source, target)
        source.unlink()


def write_jsonl_shard_atomic(records: Iterable[BaseModel], target: Path) -> Path:
    """Write a new shard; return ``<target>.sha256``; never replace either file.

    The parent directory must already exist. An exclusive sibling lock protects
    cooperating writers, and no-replace publication also protects existing files.
    Temporary files and this invocation's published data are removed on errors.
    """
    target = Path(target)
    sidecar = Path(str(target) + ".sha256")
    lock = Path(str(target) + ".lock")
    temporary: list[Path] = []
    published: list[Path] = []
    with _exclusive_lock(lock):
        try:
            if target.exists() or sidecar.exists():
                raise FileExistsError(f"immutable shard or checksum already exists: {target}")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             prefix=f".{target.name}.", suffix=".tmp",
                                             dir=target.parent, delete=False) as stream:
                data_temp = Path(stream.name)
                temporary.append(data_temp)
                for record in records:
                    stream.write(record.model_dump_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            digest = _sha256(data_temp)
            with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", newline="\n",
                                             prefix=f".{sidecar.name}.", suffix=".tmp",
                                             dir=target.parent, delete=False) as stream:
                checksum_temp = Path(stream.name)
                temporary.append(checksum_temp)
                stream.write(digest + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _publish(data_temp, target)
            published.append(target)
            _publish(checksum_temp, sidecar)
            published.append(sidecar)
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            raise
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)
    return sidecar


def verify_shard(target: Path) -> bool:
    """Return false for absent, unreadable, malformed, or mismatched pairs."""
    target = Path(target)
    try:
        expected = Path(str(target) + ".sha256").read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            return False
        return hmac.compare_digest(expected, _sha256(target))
    except (OSError, UnicodeError):
        return False
