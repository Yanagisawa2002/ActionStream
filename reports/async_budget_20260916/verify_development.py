"""Check consumed-seed provenance and independently recount profiled work misses."""

import hashlib
import json
from pathlib import Path
from record_io import archive_bytes, open_archive

HERE = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((HERE / "development_manifest.json").read_text())
    backup = json.loads((HERE / "development-backup-verification.json").read_text())
    archive = HERE / "development_records.tar.gz"
    assert (
        hashlib.sha256(archive_bytes(archive)).hexdigest()
        == manifest["compact"]["sha256"]
    )
    assert backup["status"] == "PASS" and not backup["errors"]
    assert backup["verified_files"] == manifest["files"]
    assert backup["archives"] == manifest
    with open_archive(archive) as tar:
        raw = {
            m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()
        }

    def read(name):
        return json.loads(raw[name])

    files = read("development-files.json")["files"]
    assert len(files) == manifest["files"]
    for name, row in files.items():
        if name not in raw:
            assert Path(name).suffix in (".npz", ".pt")
            continue
        assert len(raw[name]) == row["bytes"]
        assert hashlib.sha256(raw[name]).hexdigest() == row["sha256"]
    source_audit = read("development-source-verification.json")
    assert source_audit["status"] == "PASS" and len(source_audit["records"]) == 9
    cases = 0
    for record in source_audit["records"]:
        prefix = record["run"] + "/"
        provenance = read(prefix + "provenance.json")
        assert provenance["purpose"] == "DEVELOPMENT_ONLY"
        assert provenance["source_sha256"] == record["source_sha256"]
        for name, expected in provenance["source_sha256"].items():
            assert (
                hashlib.sha256(raw[prefix + "source/" + name]).hexdigest() == expected
            )
        summary = read(prefix + "profile-summary.json")
        assert [row["seed"] for row in summary["episodes"]] == provenance["seeds"]
        for row in summary["episodes"]:
            seed = row["seed"]
            assert 2026092100 <= seed <= 2026092109 or 2026092200 <= seed <= 2026092209
            events = [
                json.loads(line)
                for line in raw[prefix + str(seed) + "/runtime.jsonl"].splitlines()
            ]
            work = [e["work_wall_s"] for e in events if e["event"] == "dispatch"]
            assert len(work) == row["timing"]["controls"]
            assert sum(v > 0.05 for v in work) == row["timing"]["deadline_misses"]
            assert row["integrity_passed"] and not row["premature_stop"]
            cases += 1
    return dict(status="PASS", runs=len(source_audit["records"]), consumed_cases=cases)


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
