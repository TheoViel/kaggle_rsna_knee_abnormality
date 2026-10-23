"""Apply the hidden-test robustness edits to a team submission notebook (b1_c1 cell + our pipeline + merge).

    python tools_robust_nb.py IN.ipynb OUT.ipynb

Cells are found by their content, so the script works on any notebook of the jiwei-edu-theo / best_lb family:
  - a new first code cell runs src/infer/sanitize.py (embedded: the notebook may run a pinned src without it) into a
    shadow test root and sets RSNA_TEST_ROOT; any failure there leaves the original root in use;
  - b1_c1 cell: find_root() returns RSNA_TEST_ROOT (the synovitis specialists use the same find_root); the specialist
    block skips studies without a sagittal / axial fluid series (and studies its workers reject, rerunning), gives
    them b1_c1's synovitis rank, and keeps b1_c1's synovitis on any other failure;
  - our paths cell: DATA_DIR = RSNA_TEST_ROOT;
  - pipeline cell: a failed pipeline does not raise (the merge falls back);
  - shard summary / merge cells: any failure keeps b1_c1's submission.csv for all 12 findings;
  - medial OA specialist cell (best_lb): any failure restores the submission it started from.
Asserts each edit applies exactly once.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SANITIZE = open(os.path.join(HERE, "infer", "sanitize.py")).read()
assert "'''" not in SANITIZE

SAN_CELL = r'''# Input normalisation before every component: the hidden test set may hold DICOMs our training data never had
# (compressed transfer syntaxes, multi-frame or RGB files, several echoes at one position, no PixelSpacing, blank
# Anatomical_Plane / fluid flags in the csv). Unusual series are rewritten as plain single-frame MONOCHROME2 files in a
# shadow root (the others are symlinks), and RSNA_TEST_ROOT points b1_c1, the specialists and our pipeline to it.
# The sanitizer (src/infer/sanitize.py, embedded below) runs in its own process, with codecs from the specialist
# bundle's wheels installed in a separate folder. Any failure here leaves RSNA_TEST_ROOT unset: every component then
# reads the original competition root, as before.
import os, sys, subprocess, zipfile

ON_KAGGLE = os.path.exists("/kaggle/input")
_tmp = "/tmp" if ON_KAGGLE else os.path.abspath("tmp")
_shadow, _codecs, _script = f"{_tmp}/knee_test_root", f"{_tmp}/knee_codecs", f"{_tmp}/knee_sanitize.py"
os.environ.pop("RSNA_TEST_ROOT", None)
try:
    _roots = [p for p in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
                          "/kaggle/input/rsna-knee-abnormality-detection", "input/rsna-knee-abnormality-detection")
              if os.path.isfile(os.path.join(p, "test.csv"))]
    assert _roots, "competition root not found"
    os.makedirs(_tmp, exist_ok=True)
    open(_script, "w").write(SANITIZE_PY)
    _spec = ["/kaggle/input/datasets/arc144/rsna-knee-syn-specialists-20261009",   # as in the b1_c1 cell
             "/kaggle/input/rsna-knee-syn-specialists-20261009", "input/rsna-knee-syn-specialists-20261009"]
    _deps = [f"{d}/dicom_dependencies" for d in _spec if os.path.isdir(f"{d}/dicom_dependencies")]
    if not _deps:                                # a ZIP attachment: unpack only the wheels
        for z in [f"{d}/artifacts.zip" for d in _spec if os.path.isfile(f"{d}/artifacts.zip")][:1]:
            with zipfile.ZipFile(z) as f:
                f.extractall(f"{_tmp}/knee_specialist_wheels", [n for n in f.namelist() if n.startswith("dicom_dependencies/")])
            _deps = [f"{_tmp}/knee_specialist_wheels/dicom_dependencies"]
    _env = dict(os.environ)
    if _deps:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps", "--no-compile",
                            "--disable-pip-version-check", "--target", _codecs, "--find-links", f"{_deps[0]}/wheels",
                            "-r", f"{_deps[0]}/requirements.txt"])
        if r.returncode == 0:
            _env["PYTHONPATH"] = _codecs + os.pathsep + _env.get("PYTHONPATH", "")
        print("codecs:", "installed" if r.returncode == 0 else "pip failed, the environment's decoders only")
    else:
        print("codecs: specialist wheels not found, the environment's decoders only")
    r = subprocess.run([sys.executable, _script, "--root", _roots[0], "--out", _shadow, "--workers", str(os.cpu_count())],
                       env=_env, timeout=3600)
    assert r.returncode == 0 and os.path.isfile(os.path.join(_shadow, "test_series.csv")), f"sanitize rc={r.returncode}"
    os.environ["RSNA_TEST_ROOT"] = _shadow
except Exception as e:  # noqa: BLE001
    print("!! input normalisation skipped, components read the original root:", repr(e))
print("RSNA_TEST_ROOT:", os.environ.get("RSNA_TEST_ROOT", "(original)"))
'''

SPEC_OLD = '''    _ids_path = WORK / "specialist_ids.csv"
    _baseline[["StudyInstanceUID"]].to_csv(_ids_path, index=False)
    _env = dict(os.environ, PYTHONPATH=str(_dependencies)+os.pathsep+str(_runtime)+os.pathsep+os.environ.get("PYTHONPATH", ""),
                OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    _out = WORK / "specialists"
    subprocess.run([sys.executable, str(_runtime / "synovitis_submission_runtime.py"),
        "--bundle", str(_assets), "--root", str(find_root()), "--ids", str(_ids_path),
        "--output", str(_out), "--variant", VARIANT], env=_env, check=True)
    _ours = pd.read_csv(_out / "specialist_predictions.csv", dtype={"StudyInstanceUID": str}, float_precision="round_trip")
    if not _ours.StudyInstanceUID.is_unique or set(_ours.StudyInstanceUID) != set(_ids):
        raise RuntimeError("Specialist study identities differ")
    _ours = _ours.set_index("StudyInstanceUID").loc[_ids]
    _families = ["resnet", "inception", "dino"] if VARIANT == "threeway" else ["resnet"]
    _probabilities = _ours[_families].mean(axis=1).to_numpy()
    if not np.isfinite(_probabilities).all() or (_probabilities < 0).any() or (_probabilities > 1).any():
        raise RuntimeError("Invalid specialist probabilities")
    if not np.allclose(_probabilities, _ours.ours.to_numpy(), atol=1e-14, rtol=0):
        raise RuntimeError("Within-specialist probability average differs")
    _jiwei_rank = pd.Series(_baseline.Synovitis.to_numpy()).rank(method="average", pct=True).to_numpy()
    _ours_rank = pd.Series(_probabilities).rank(method="average", pct=True).to_numpy()
'''
SPEC_NEW = '''    _ids_path = WORK / "specialist_ids.csv"
    _env = dict(os.environ, PYTHONPATH=str(_dependencies)+os.pathsep+str(_runtime)+os.pathsep+os.environ.get("PYTHONPATH", ""),
                OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    _out = WORK / "specialists"
    # The runtime stops on the first study without a usable sagittal / axial fluid series ("No usable axial/sagittal
    # fluid input"). Studies without such a series in the csv are skipped up front; a study that still fails (e.g. its
    # pixels do not decode) is read from the worker logs, skipped, and the run repeated (at most 5 runs). Skipped
    # studies take b1_c1's synovitis rank in place of the specialists' (below).
    _sc = pd.read_csv(find_root() / "test_series.csv", dtype=str)
    _fl = pd.to_numeric(_sc["Fluid_Sensitive"], errors="coerce") if "Fluid_Sensitive" in _sc else pd.Series(np.nan, index=_sc.index)
    _usable = set(_sc[_sc.Anatomical_Plane.isin(["Sagittal", "Axial"]) & (_fl.isna() | (_fl == 1))].StudyInstanceUID)
    _skip = {u for u in _ids if u not in _usable}
    for _attempt in range(5):
        _run_ids = [u for u in _ids if u not in _skip]
        pd.DataFrame({"StudyInstanceUID": _run_ids}).to_csv(_ids_path, index=False)
        shutil.rmtree(_out, ignore_errors=True)
        _rc = subprocess.run([sys.executable, str(_runtime / "synovitis_submission_runtime.py"),
            "--bundle", str(_assets), "--root", str(find_root()), "--ids", str(_ids_path),
            "--output", str(_out), "--variant", VARIANT], env=_env).returncode if _run_ids else 1
        if _rc == 0:
            break
        _logs = " ".join(p.read_text(errors="ignore") for p in _out.glob("*.log"))
        _bad = set(re.findall(r"No usable axial/sagittal fluid input for ([0-9.]+)", _logs)) & set(_run_ids)
        if not _bad:
            raise RuntimeError(f"Specialist run failed (rc={_rc}) without a skippable study")
        _skip |= _bad
    else:
        raise RuntimeError("Specialist run still failing after 5 attempts")
    print(f"specialists: {len(_run_ids)} studies, {len(_skip)} skipped (b1_c1's synovitis rank)", flush=True)
    _ours = pd.read_csv(_out / "specialist_predictions.csv", dtype={"StudyInstanceUID": str}, float_precision="round_trip")
    if not _ours.StudyInstanceUID.is_unique or set(_ours.StudyInstanceUID) != set(_run_ids):
        raise RuntimeError("Specialist study identities differ")
    _ours = _ours.set_index("StudyInstanceUID").reindex(_ids)
    _families = ["resnet", "inception", "dino"] if VARIANT == "threeway" else ["resnet"]
    _probabilities = _ours[_families].mean(axis=1).to_numpy()
    _ran = np.isin(np.asarray(_ids), _run_ids)
    if not np.isfinite(_probabilities[_ran]).all() or (_probabilities[_ran] < 0).any() or (_probabilities[_ran] > 1).any():
        raise RuntimeError("Invalid specialist probabilities")
    if not np.allclose(_probabilities[_ran], _ours.ours.to_numpy()[_ran], atol=1e-14, rtol=0):
        raise RuntimeError("Within-specialist probability average differs")
    _jiwei_rank = pd.Series(_baseline.Synovitis.to_numpy()).rank(method="average", pct=True).to_numpy()
    _ours_rank = pd.Series(_probabilities).rank(method="average", pct=True).to_numpy().copy()   # NaN for skipped studies
    if not _ran.all():                           # skipped: b1_c1's rank; the specialists' studies keep their order
        _ours_rank[_ran] = pd.Series(_probabilities[_ran]).rank(method="average", pct=True).to_numpy()
        _ours_rank[~_ran] = _jiwei_rank[~_ran]
'''


def indent(s):
    return "".join("    " + l if l.strip() else l for l in s.splitlines(keepends=True))


def once(s, old, new):
    assert s.count(old) == 1, old[:80]
    return s.replace(old, new)


def main():
    src_path, out_path = sys.argv[1], sys.argv[2]
    nb = json.load(open(src_path))
    cells = nb["cells"]
    get = lambda c: "".join(c["source"])
    find = lambda pred: [i for i, c in enumerate(cells) if c["cell_type"] == "code" and pred(get(c))]
    assert not find(lambda s: "RSNA_TEST_ROOT" in s), "notebook already patched"

    # b1_c1 cell
    [i] = find(lambda s: "def find_root() -> Path:" in s)
    s = get(cells[i])
    s = once(s, 'def find_root() -> Path:\n    candidates = [',
             'def find_root() -> Path:\n    if os.environ.get("RSNA_TEST_ROOT"):            # the sanitized shadow root (first code cell)\n'
             '        return Path(os.environ["RSNA_TEST_ROOT"])\n    candidates = [')
    start = s.index("# Synovitis specialist blend.")
    blk = once(s[start:], SPEC_OLD, SPEC_NEW)
    blk = once(blk, "    import subprocess\n", "    import re\n    import shutil\n    import subprocess\n")
    pre, body = blk.split('if __name__ == "__main__":\n')
    blk = (pre + 'if __name__ == "__main__":\n    try:\n' + indent(body).rstrip() + "\n"
           "    except Exception as _e:  # noqa: BLE001 -- any failure keeps b1_c1's synovitis\n"
           "        print(\"!! synovitis specialists failed, b1_c1's synovitis kept:\", repr(_e), flush=True)\n"
           "        if (WORK / \"baseline_submission.csv\").is_file():\n"
           "            (WORK / \"submission.csv\").write_bytes((WORK / \"baseline_submission.csv\").read_bytes())\n")
    s = s[:start] + blk
    compile(s, "b1_c1", "exec")
    cells[i]["source"] = s.splitlines(keepends=True)
    b1_cell = i

    # our paths
    [i] = find(lambda s: 'OUT_DIR = "../output/inference/"' in s)
    s = once(get(cells[i]), '    OUT_DIR = "../output/inference/"\n',
             '    OUT_DIR = "../output/inference/"\nDATA_DIR = os.environ.get("RSNA_TEST_ROOT", DATA_DIR)   # the sanitized shadow root (first code cell)\n')
    cells[i]["source"] = s.splitlines(keepends=True)

    # pipeline: never raise (the merge falls back to b1_c1)
    [i] = find(lambda s: "cmd = (" in s and "-m infer.pipeline" in s)
    s = get(cells[i])
    if "subprocess.run(shlex.split(cmd), check=True)" in s:
        s = once(s, "subprocess.run(shlex.split(cmd), check=True)",
                 "_rc = subprocess.run(shlex.split(cmd)).returncode   # a failure falls back to b1_c1 in the merge\n"
                 "if _rc:\n    print(f\"!! pipeline rc={_rc}\")")
        cells[i]["source"] = s.splitlines(keepends=True)

    # shard summary
    [i] = find(lambda s: s.startswith("shards = sorted("))
    cells[i]["source"] = ("try:\n" + indent(get(cells[i])).rstrip() + "\n"
                          "except Exception as e:  # noqa: BLE001 -- the merge cell falls back to b1_c1\n"
                          "    print(\"!! no pipeline outputs:\", repr(e))\n").splitlines(keepends=True)

    # merge
    [i] = find(lambda s: s.startswith("COLS = ["))
    cells[i]["source"] = (
        "# Any failure below (our pipeline crashed / wrote nothing, a missing column, ...) keeps b1_c1's submission.csv for\n"
        "# all 12 findings rather than stopping the notebook: the file is only rewritten by the last line.\n"
        "try:\n" + indent(get(cells[i])).rstrip() + "\n"
        "except Exception as e:  # noqa: BLE001\n"
        "    print(\"!! merge failed, b1_c1's submission kept for every finding:\", repr(e))\n"
        "    sub = pd.read_csv(\"submission.csv\")\n"
        "    assert sub.drop(columns=\"StudyInstanceUID\").notna().all().all()\n"
        "sub.head()\n").splitlines(keepends=True)

    # medial OA specialist (best_lb)
    for i in find(lambda s: "medial_oa_blend_runtime.py" in s and "baseline_submission_v31.csv" in s):
        s = get(cells[i])
        s = once(s, 'shutil.copyfile("submission.csv", "baseline_submission_v31.csv")\n',
                 'shutil.copyfile("submission.csv", "baseline_submission_v31.csv")\ntry:\n')
        head, body = s.split("try:\n")
        body, tail = body.rsplit('pd.read_csv("submission.csv").head()', 1)
        s = (head + "try:\n" + indent(body).rstrip() + "\n"
             "except Exception as e:  # noqa: BLE001 -- keep the submission this cell started from\n"
             "    print(\"!! medial OA specialist failed, submission before it kept:\", repr(e))\n"
             "    shutil.copyfile(\"baseline_submission_v31.csv\", \"submission.csv\")\n"
             'pd.read_csv("submission.csv").head()' + tail)
        compile(s.replace("%%time", ""), "moa", "exec")
        cells[i]["source"] = s.splitlines(keepends=True)

    for c in cells:
        if c["cell_type"] == "code":
            compile("\n".join(l for l in get(c).splitlines() if not l.lstrip().startswith(("%", "!"))), "cell", "exec")
    san = "SANITIZE_PY = r'''" + SANITIZE + "'''\n\n" + SAN_CELL
    compile(san, "san", "exec")
    cells.insert(b1_cell, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": san.splitlines(keepends=True)})
    json.dump(nb, open(out_path, "w"), indent=1, ensure_ascii=False)
    open(out_path, "a").write("\n")
    print("patched ->", out_path)


if __name__ == "__main__":
    main()
