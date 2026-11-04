"""Scoring sets of the gen_*.py outputs, in the IntPhys2 layout score/run_test.py reads (plain copies, no links).

One set per output name (lambda, sigma', prefix, seed):
  <out>/sets/<name stem>/Main/metadata.csv     copy of the source IntPhys2 Main/metadata.csv
  <out>/sets/<name stem>/Main/Videos/<Impossible video>.mp4   the generated clip (Impossible slot)
                                     <Possible video>.mp4     the original Possible clip
                                     meta.json                {"outputs": {<Impossible video>.mp4: {"choice", ...}}}
so `score/run_test.py --intphys2-dir <out>/sets/<name stem>` scores it and score/run_compare.py finds the side the
generation chose next to each clip. Without a source metadata.csv (-i was a plain folder of .mp4 files) no set is written.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
from pathlib import Path


def _copy(src: Path, dst: Path) -> None:
    """Copy unless `dst` is already as new as `src` (a regenerated clip replaces the old copy)."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def export_sets(pair: dict, out_root: Path, names: list[str], metadata: Path) -> int:
    """Copy this pair's existing outputs `names` into their sets; returns how many sets hold it."""
    if not metadata.is_file():
        return 0
    meta_f = Path(pair["out"]) / "meta.json"
    outputs = json.loads(meta_f.read_text()).get("outputs", {}) if meta_f.exists() else {}
    done = 0
    for name in names:
        gen = Path(pair["out"]) / name
        if not gen.is_file():
            continue
        videos = Path(out_root) / "sets" / Path(name).stem / "Main" / "Videos"
        videos.mkdir(parents=True, exist_ok=True)
        if not (videos.parent / "metadata.csv").exists():
            _copy(metadata, videos.parent / "metadata.csv")
        bad, good = Path(pair["bad"]).name, Path(pair["good"]).name
        _copy(Path(pair["good"]), videos / good)
        _copy(gen, videos / bad)
        entry = dict(outputs.get(name, {}), cid=pair["cid"], source=f"{pair['cid']}/{name}")
        with open(videos / ".meta.lock", "w") as lock:          # several processes may share one output folder
            fcntl.flock(lock, fcntl.LOCK_EX)
            sm = videos / "meta.json"
            data = json.loads(sm.read_text()) if sm.exists() else {"outputs": {}}
            if data["outputs"].get(bad) != entry:
                data["outputs"][bad] = entry
                sm.with_name("meta.json.tmp").write_text(json.dumps(data, indent=1))
                os.replace(sm.with_name("meta.json.tmp"), sm)
        done += 1
    return done
