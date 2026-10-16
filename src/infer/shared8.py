"""Shared8: the teammate's four-model ROI ensemble, as a recipe model of infer.pipeline ("shared8").

Ported from output/edu/runtime/shared8_fourway_hook.py (notebook rsna-theo-v13-shared8-fourway-20261015), which
replaced model 88 of dataset_6 by this ensemble. The bundle sits in <weights>/shared8/, unchanged:
  runtime/  shared8_fourway_worker.py, shared8_deploy_model.py, shared8_deploy_preprocess.py
  deps/     the pinned timm 1.0.22 the worker imports (isolated from the pipeline's process)
  configs/, models/   the four models: CoAtNet2 ImageNet1k 128, CoAtNet2 12k 128, ConvNeXt-tiny 128,
                      EfficientNetV2-S 224 on native pixels
  submission_manifest.json   their equal (0.25) probability weights

Per chunk, every ok sagittal / coronal series with ROI boxes is rendered by their own `prepare_series` from its raw
pixels (8 ROI crops at 128 px "original" and 224 px "native" pixels; missing ROIs are zeroed and left out), sent to
the worker process (same GPU, separate CUDA allocator), and its study rows come back: per (study, finding) the mean
over series of each model, then their 0.25-weighted mean ("ours"). That value is the study probability of model
"shared8" in the recipe, exactly what their hook inserted under the name "88".
"""

import json
import os
import select
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

NAME = "shared8"


def _preprocess(root):
    """Their shared8_deploy_preprocess module (runtime/), imported from the bundle."""
    sys.path.insert(0, str(Path(root) / "runtime"))
    try:
        import shared8_deploy_preprocess as m
    finally:
        sys.path.pop(0)
    return m


def render_one(item):
    """(series, its box records, bundle root) -> (original 128 px crops, native 224 px crops, have) or None.
    Copied from the hook's render_one."""
    series, records, root = item
    pre = _preprocess(root)
    CLASSES, prepare_series = pre.CLASSES, pre.prepare_series
    raw = series["_shared8_raw"].numpy()
    if series["_shared8_unsigned"]:
        raw = raw.view(np.uint16)
    lookup = {r["class"]: r for r in records if r["class"] in CLASSES}
    assert len(lookup) == sum(r["class"] in CLASSES for r in records), "Duplicate ROI records"
    have = np.array([c in lookup for c in CLASSES], dtype=bool)
    if not have.any():
        return None
    # Missing ROIs are never inferred or included in a mean; placeholder geometry only satisfies the renderer.
    records = [lookup.get(c, dict(**{"class": c}, z0=0, z1=0, y0=0, y1=0, x0=0, x1=0)) for c in CLASSES]
    side = "L" if (series["side"] == "L") ^ bool(series["conv_flip"]) else "R"
    kw = dict(volume=raw, records=records, side=side, plane=series["plane"], original_n_slices=series["n_slices"])
    original = prepare_series(**kw, sizes=(128,), pixels="original")[128]
    native = prepare_series(**kw, sizes=(224,), pixels="native")[224]
    original[~have] = 0
    native[~have] = 0
    return original, native, have


class Worker:
    """The four models, resident in their own process (copied from the hook's Worker)."""

    def __init__(self, root, out_dir):
        self.root, self.out_dir = Path(root), Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.out_dir / "shared8_worker.log").open("w")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(self.root / "runtime"), str(self.root / "deps")]),
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   SHARED8_FOURWAY_BUNDLE=str(self.root.resolve()))
        self.process = subprocess.Popen([sys.executable, "-u", str(self.root / "runtime/shared8_fourway_worker.py")],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                        bufsize=1, env=env)
        self.requests = 0
        try:
            self.handshake = self.receive(600)
            assert self.handshake["kind"] == "ready", self.handshake
        except BaseException:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
            self.log.close()
            raise

    def receive(self, timeout=600):
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise RuntimeError("Shared8 worker timed out; inspect shared8_worker.log")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Shared8 worker exited ({self.process.poll()}); inspect shared8_worker.log")
        result = json.loads(line)
        if result["kind"] == "error":
            raise RuntimeError(result["error"])
        return result

    def predict(self, prepared, metadata):
        self.requests += 1
        original = np.stack([v[0] for v in prepared]) if prepared else np.empty((0, 8, 8, 128, 128), np.uint8)
        native = np.stack([v[1] for v in prepared]) if prepared else np.empty((0, 8, 8, 224, 224), np.uint8)
        have = np.stack([v[2] for v in prepared]) if prepared else np.empty((0, 8), bool)
        planes = np.array([0 if s["plane"] == "Sagittal" else 1 for s in metadata], np.int64)
        with tempfile.TemporaryDirectory(prefix="shared8_chunk_", dir=self.out_dir) as directory:
            path = Path(directory) / "crops.npz"
            np.savez(path, original=original, native=native, have=have, planes=planes)
            request = dict(kind="predict", request_id=self.requests, path=str(path.resolve()), series=metadata)
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            result = self.receive()
        assert result["kind"] == "predictions" and result["request_id"] == self.requests
        return result

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write('{"kind":"stop"}\n')
                self.process.stdin.flush()
                result = self.receive(30)
                assert result["kind"] == "stopped"
                self.process.wait(timeout=30)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.log.close()


def predict_chunk(studies, boxes, root, worker_holder, out_dir):
    """Shared8 study probabilities of one chunk -> rows (model, StudyInstanceUID, cls, p) for the recipe.
    worker_holder: a dict keeping the (lazily started) Worker across chunks."""
    import torch

    lookup = {uid: g.to_dict("records") for uid, g in boxes.groupby("SeriesInstanceUID")} if len(boxes) else {}
    eligible = [s for st in studies for s in st if s["status"] == "ok" and s["plane"] in ("Sagittal", "Coronal")
                and s["SeriesInstanceUID"] in lookup and "_shared8_raw" in s]
    with ThreadPoolExecutor(max_workers=2) as pool:
        rendered = list(pool.map(render_one, [(s, lookup[s["SeriesInstanceUID"]], root) for s in eligible]))
    pairs = [(s, v) for s, v in zip(eligible, rendered) if v is not None]
    metadata = [{k: s[k] for k in ["StudyInstanceUID", "SeriesInstanceUID", "plane"]} for s, v in pairs]
    torch.cuda.synchronize()
    torch.cuda.empty_cache()             # separate processes have separate allocators: free unused workspaces first
    if worker_holder.get("worker") is None:
        worker_holder["worker"] = Worker(root, out_dir)
    result = worker_holder["worker"].predict([v for s, v in pairs], metadata)
    for st in studies:
        for s in st:
            s.pop("_shared8_raw", None)
            s.pop("_shared8_unsigned", None)
    out = result["study_rows"]
    if not out:
        return pd.DataFrame(columns=["model", "StudyInstanceUID", "cls", "p"])
    return pd.DataFrame(out)[["StudyInstanceUID", "cls", "ours"]].rename(columns={"ours": "p"}).assign(model=NAME)
