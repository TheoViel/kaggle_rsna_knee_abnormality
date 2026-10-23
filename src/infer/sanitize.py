"""Input normaliser for the hidden test set, run before every inference component (b1_c1, our pipeline, specialists).

    python -m infer.sanitize --root <competition root> --out <shadow root> [--workers 4]

Every model was developed on plain DICOMs: uncompressed, one grayscale frame per file, MONOCHROME2, one file per slice
position, a filled Anatomical_Plane. The hidden test set may differ; one unusual series must not crash a component
(e.g. b1_c1's B1 unpacks `height, width = image.shape` and stops the whole notebook on an RGB or multi-frame file).
This pass reads headers only and writes a shadow root that every component reads instead:
  - test.csv copied; test_series.csv copied, with
      Anatomical_Plane normalised ("SAG", "sagittal" -> "Sagittal") and, when blank, filled from the series' majority
        ImageOrientationPatient (left blank when there is no geometry),
      a blank Fluid_Sensitive / Fat_Suppression (equal in every training series) filled from the headers: fat
        suppression in ScanOptions / SeriesDescription / ProtocolName or an inversion pulse (99.7% of 3,000 training
        series; b1_c1 calls int() on these flags);
  - a series that needs no change: a symlink to the original folder (no copy);
  - a series that does: rewritten as uncompressed single-frame MONOCHROME2 grayscale files:
      compressed transfer syntax -> decoded (codecs: pylibjpeg / openjpeg / gdcm / pillow, whatever is installed),
      multi-frame file -> one file per frame, ImagePositionPatient / ImageOrientationPatient / PixelSpacing taken
        from the per-frame or shared functional groups,
      RGB / YBR -> luminance (uint8),
      several files at one slice position (e.g. echoes) -> the one with the lowest InstanceNumber,
      no PixelSpacing -> ImagerPixelSpacing, else square pixels over a 160 mm field of view on the longer side (the
        training median in every plane; spacing only scales geometry, a missing tag must not drop the series);
  - a file that cannot be read or decoded is left out of the shadow series; a series with no readable file keeps
    its symlink (the components then fail on it as before and their per-study fallbacks apply).
Missing geometry is not invented (a guessed orientation could swap medial and lateral). Writes <out>/sanitize.json:
counts per issue and the series rewritten. Never raises on a bad series: the run itself must not stop the notebook.
"""
import argparse
import copy
import glob
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PLAIN = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"}


def _orient(iop):
    n = np.abs(np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float)))
    return ["Sagittal", "Coronal", "Axial"][int(np.argmax(n))]


def fat_suppressed(d):
    """Header -> 0 / 1, the csv's Fat_Suppression (= Fluid_Sensitive) convention."""
    g = lambda k: str(d.get(k, "") or "").upper()
    text = g("SeriesDescription") + " " + g("ProtocolName")
    if re.search(r"NON[ _-]?(FS|FAT)", text):
        return 0
    if set(re.findall(r"[A-Z0-9_]+", g("ScanOptions"))) & {"FS", "FAT_SAT", "SFS", "CHESS", "WE"}:
        return 1
    if re.search(r"(?<![A-Z])(FS|FAT|SPAIR|SPIR|STIR|FATSAT|FSAT|TIRM|DIXON|WATER|WE)(?![A-Z])|_FS|FS_|SPAIR|STIR|TIRM|"
                 r"FATSAT|_WE_", text):
        return 1
    return int(float(d.get("InversionTime", 0) or 0) > 0 or "IR" in g("ScanningSequence"))


def inspect(folder):
    """Header pass over a series -> (issues, majority plane, fat suppression guess, per-file (path, position key,
    instance))."""
    import pydicom
    files = sorted(glob.glob(os.path.join(folder, "*.dcm"))) or sorted(
        f for f in glob.glob(os.path.join(folder, "*")) if os.path.isfile(f))
    issues, planes, rows, fs = set(), [], [], []
    for f in files:
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:  # noqa: BLE001
            issues.add("unreadable_header")
            continue
        ts = str(getattr(getattr(d, "file_meta", None), "TransferSyntaxUID", "") or "")
        if ts and ts not in PLAIN:
            issues.add("compressed")
        if int(d.get("NumberOfFrames", 1) or 1) > 1:
            issues.add("multiframe")
        if "PixelSpacing" not in d:
            issues.add("no_spacing")
        if int(d.get("SamplesPerPixel", 1) or 1) != 1 or not str(d.get("PhotometricInterpretation", "MONOCHROME2")).startswith("MONOCHROME"):
            issues.add("color")
        if not fs:
            fs.append(fat_suppressed(d))
        iop, ipp = d.get("ImageOrientationPatient"), d.get("ImagePositionPatient")
        if iop is not None and len(iop) == 6:
            planes.append(_orient(iop))
        key = None
        if iop is not None and ipp is not None and len(iop) == 6:
            n = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
            key = round(float(np.dot(np.asarray(ipp, float), n)), 2)
        rows.append((f, key, int(d.get("InstanceNumber", 0) or 0)))
    keys = [k for _, k, _ in rows if k is not None]
    if len(keys) != len(set(keys)):
        issues.add("duplicate_positions")
    plane = max(set(planes), key=planes.count) if planes else ""
    return sorted(issues), plane, (fs[0] if fs else 0), rows


def _frames(d):
    """Dataset -> list of (2D grayscale array, geometry dict) for each frame."""
    a = d.pixel_array
    n = int(d.get("NumberOfFrames", 1) or 1)
    color = int(d.get("SamplesPerPixel", 1) or 1) > 1
    if n == 1:
        a = a[None]
    if color:                                  # (n, H, W, 3), RGB after pydicom's YBR conversion
        a = (a[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).round().clip(0, 255).astype(np.uint8)
    shared = d.get("SharedFunctionalGroupsSequence", [None])[0]
    per = d.get("PerFrameFunctionalGroupsSequence", None)
    out = []
    for k in range(a.shape[0]):
        geo = {}
        for name, seq, attr in (("ImageOrientationPatient", "PlaneOrientationSequence", "ImageOrientationPatient"),
                                ("ImagePositionPatient", "PlanePositionSequence", "ImagePositionPatient"),
                                ("PixelSpacing", "PixelMeasuresSequence", "PixelSpacing"),
                                ("SliceThickness", "PixelMeasuresSequence", "SliceThickness")):
            for grp in ([per[k]] if per is not None and k < len(per) else []) + ([shared] if shared is not None else []):
                if seq in grp and attr in grp[seq][0]:
                    geo[name] = grp[seq][0][attr].value
                    break
            if name not in geo and name in d:
                geo[name] = d[name].value
        out.append((a[k], geo))
    return out


def rewrite(job):
    """Rewrite one series (see the module doc) -> (series uid, files written, error or '')."""
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    src, dst, rows, dedupe = job
    try:
        os.makedirs(dst, exist_ok=True)
        if dedupe:                             # keep the lowest InstanceNumber per slice position
            best = {}
            for f, key, inst in rows:
                if key is None or key not in best or inst < best[key][1]:
                    best[key if key is not None else f] = (f, inst)
            files = sorted(f for f, _ in best.values())
        else:
            files = [f for f, _, _ in rows]
        written = 0
        for f in files:
            try:
                d = pydicom.dcmread(f)
                frames = _frames(d)
            except Exception:  # noqa: BLE001 -- an undecodable file is left out
                continue
            multi = len(frames) > 1
            for t in ("PerFrameFunctionalGroupsSequence", "SharedFunctionalGroupsSequence", "NumberOfFrames",
                      "PlanarConfiguration", "PixelData", "FloatPixelData", "DoubleFloatPixelData"):
                if t in d:
                    del d[t]
            for k, (img, geo) in enumerate(frames):
                e = copy.deepcopy(d)
                for name, v in geo.items():
                    setattr(e, name, v)
                if "PixelSpacing" not in e:
                    e.PixelSpacing = list(e.get("ImagerPixelSpacing") or [160.0 / max(img.shape[0], img.shape[1])] * 2)
                if img.dtype == np.uint8:
                    e.BitsAllocated, e.BitsStored, e.HighBit, e.PixelRepresentation = 8, 8, 7, 0
                else:
                    img = img.astype(np.int16 if img.dtype.kind == "i" else np.uint16)
                    e.BitsAllocated, e.BitsStored, e.HighBit = 16, 16, 15
                    e.PixelRepresentation = 1 if img.dtype == np.int16 else 0
                e.SamplesPerPixel, e.Rows, e.Columns = 1, img.shape[0], img.shape[1]
                if not str(e.get("PhotometricInterpretation", "MONOCHROME2")).startswith("MONOCHROME"):
                    e.PhotometricInterpretation = "MONOCHROME2"
                e.PixelData = np.ascontiguousarray(img).tobytes()
                e["PixelData"].VR = "OB" if img.dtype == np.uint8 else "OW"
                e.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                if multi:
                    e.InstanceNumber = k + 1
                    e.SOPInstanceUID = generate_uid()
                name = os.path.splitext(os.path.basename(f))[0] + (f"_f{k:04d}" if multi else "") + ".dcm"
                e.save_as(os.path.join(dst, name + ".tmp"), enforce_file_format=True)
                os.replace(os.path.join(dst, name + ".tmp"), os.path.join(dst, name))
                written += 1
        return os.path.basename(src), written, ""
    except Exception as ex:  # noqa: BLE001
        return os.path.basename(src), 0, repr(ex)[:200]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--series_dir", default="test_series")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    src_root, out_root = os.path.abspath(a.root), os.path.abspath(a.out)
    os.makedirs(os.path.join(out_root, a.series_dir), exist_ok=True)
    report = dict(series=0, rewritten=[], issues={}, errors=[], planes_filled=0, flags_filled=0)
    series = pd.read_csv(os.path.join(src_root, "test_series.csv"), dtype=str)
    for f in ("test.csv",):
        if os.path.exists(os.path.join(src_root, f)):
            pd.read_csv(os.path.join(src_root, f), dtype=str).to_csv(os.path.join(out_root, f), index=False)
    folders = [os.path.join(src_root, a.series_dir, r.StudyInstanceUID, r.SeriesInstanceUID) for r in series.itertuples()]
    with ProcessPoolExecutor(a.workers) as ex:
        info = list(ex.map(inspect, folders, chunksize=8))
    jobs = []
    blank = lambda v: not isinstance(v, str) or not v.strip()
    if "Anatomical_Plane" not in series:
        series["Anatomical_Plane"] = ""
    canon = {"S": "Sagittal", "C": "Coronal", "A": "Axial", "T": "Axial"}       # T: transverse
    series["Anatomical_Plane"] = [canon.get(v.strip()[:1].upper(), v) if not blank(v) else "" for v in series.Anatomical_Plane]
    for r, folder, (issues, plane, fs, rows) in zip(series.itertuples(), folders, info):
        report["series"] += 1
        for i in issues:
            report["issues"][i] = report["issues"].get(i, 0) + 1
        dst = os.path.join(out_root, a.series_dir, r.StudyInstanceUID, r.SeriesInstanceUID)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if blank(r.Anatomical_Plane) and plane:
            series.loc[r.Index, "Anatomical_Plane"] = plane
            report["planes_filled"] += 1
        for c in ("Fluid_Sensitive", "Fat_Suppression"):
            if c in series and blank(series.at[r.Index, c]):
                series.at[r.Index, c] = str(fs)
                report["flags_filled"] += 1
        fix = [i for i in issues if i in ("compressed", "multiframe", "color", "duplicate_positions", "no_spacing")]
        if fix and rows:
            jobs.append((folder, dst, rows, "duplicate_positions" in issues))
        elif not os.path.lexists(dst):
            os.symlink(folder, dst, target_is_directory=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for (folder, dst, _, _), (uid, written, err) in zip(jobs, ex.map(rewrite, jobs)):
            if err or written == 0:            # nothing usable written: keep the original series
                report["errors"].append(dict(series=uid, error=err or "no file written"))
                if os.path.isdir(dst) and not os.path.islink(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                if not os.path.lexists(dst):
                    os.symlink(folder, dst, target_is_directory=True)
            else:
                report["rewritten"].append(dict(series=uid, files=written))
    series.to_csv(os.path.join(out_root, "test_series.csv"), index=False)
    json.dump(report, open(os.path.join(out_root, "sanitize.json"), "w"), indent=1)
    print(f"sanitize: {report['series']} series, issues {report['issues']}, rewritten {len(report['rewritten'])}, "
          f"planes filled {report['planes_filled']}, flags filled {report['flags_filled']}, errors {len(report['errors'])}", flush=True)


if __name__ == "__main__":
    main()
