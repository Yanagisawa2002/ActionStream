"""Read byte-identical compact records split to respect Git's file-size limit."""

import hashlib
import io
import json
from pathlib import Path
import tarfile


def archive_bytes(path):
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    manifest = json.loads(path.with_name("record_parts.json").read_text())[path.name]
    blocks = []
    for row in manifest["parts"]:
        assert Path(row["name"]).name == row["name"]
        data = path.with_name(row["name"]).read_bytes()
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        blocks.append(data)
    data = b"".join(blocks)
    assert len(data) == manifest["bytes"]
    assert hashlib.sha256(data).hexdigest() == manifest["sha256"]
    return data


def open_archive(path):
    return tarfile.open(fileobj=io.BytesIO(archive_bytes(path)))
