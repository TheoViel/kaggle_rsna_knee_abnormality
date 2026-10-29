"""Input normalisation for the hidden test set, the first step of `infer.pipeline`.

The models were developed on plain DICOMs: uncompressed, one grayscale frame per file, MONOCHROME2, one file per
slice position, PixelSpacing present, a filled Anatomical_Plane. The hidden test set may differ; one unusual series
must not lose its predictions or stop the run. A header pass over every series, then:
  - Anatomical_Plane normalised ("SAG", "sagittal" -> "Sagittal") and, when blank, filled from the series' majority
    ImageOrientationPatient (left blank without geometry: the pipeline then falls back on its own reading);
  - a series that needs a change is rewritten under <out_dir>/<SeriesInstanceUID>/ as uncompressed single-frame
    MONOCHROME2 grayscale files, and the series list points to the copy:
      compressed transfer syntax -> decoded (pylibjpeg / openjpeg / gdcm / pillow, whatever is installed),
      multi-frame file -> one file per frame, ImagePositionPatient / ImageOrientationPatient / PixelSpacing taken
        from the per-frame or shared functional groups,
      RGB / YBR -> luminance (uint8),
      several files at one slice position (e.g. echoes) -> the one with the lowest InstanceNumber,
      no PixelSpacing -> ImagerPixelSpacing, else square pixels over a 160 mm field of view (`dicom_io.pixel_spacing`);
  - a file that cannot be read or decoded is left out of the copy; a series with no usable file keeps its original
    folder (the pipeline then fails on it alone, and the study is predicted from its other series).
Missing geometry is not invented: a guessed orientation could swap medial and lateral. Never raises on a bad series.
"""

import copy
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dicom_io import orientation

PLAIN = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"}
CANON = {"S": "Sagittal", "C": "Coronal", "A": "Axial", "T": "Axial"}        # T: transverse


def inspect(folder):
    """Header pass over a series -> (issues, majority plane, per-file (path, position key, instance))."""
    import pydicom

    issues, planes, rows = set(), [], []
    for f in sorted(glob.glob(os.path.join(folder, "*.dcm"))):
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
        if int(d.get("SamplesPerPixel", 1) or 1) != 1 or \
                not str(d.get("PhotometricInterpretation", "MONOCHROME2")).startswith("MONOCHROME"):
            issues.add("color")
        iop, ipp = d.get("ImageOrientationPatient"), d.get("ImagePositionPatient")
        key = None
        if iop is not None and len(iop) == 6:
            planes.append(orientation(iop))
            if ipp is not None:
                key = round(float(np.dot(np.asarray(ipp, float),
                                         np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float)))), 2)
        rows.append((f, key, int(d.get("InstanceNumber", 0) or 0)))
    keys = [k for _, k, _ in rows if k is not None]
    if len(keys) != len(set(keys)):
        issues.add("duplicate_positions")
    return sorted(issues), (max(set(planes), key=planes.count) if planes else ""), rows


def _frames(d):
    """Dataset -> list of (2D grayscale array, geometry dict) for each frame."""
    a = d.pixel_array
    if int(d.get("NumberOfFrames", 1) or 1) == 1:
        a = a[None]
    if int(d.get("SamplesPerPixel", 1) or 1) > 1:       # (n, H, W, 3), RGB after pydicom's YBR conversion
        a = (a[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).round().clip(0, 255)
        a = a.astype(np.uint8)
    shared = d.get("SharedFunctionalGroupsSequence", [None])[0]
    per = d.get("PerFrameFunctionalGroupsSequence", None)
    out = []
    for k in range(a.shape[0]):
        geo = {}
        for name, seq in (("ImageOrientationPatient", "PlaneOrientationSequence"),
                          ("ImagePositionPatient", "PlanePositionSequence"),
                          ("PixelSpacing", "PixelMeasuresSequence"), ("SliceThickness", "PixelMeasuresSequence")):
            for grp in ([per[k]] if per is not None and k < len(per) else []) + ([shared] if shared is not None else []):
                if seq in grp and name in grp[seq][0]:
                    geo[name] = grp[seq][0][name].value
                    break
            if name not in geo and name in d:
                geo[name] = d[name].value
        out.append((a[k], geo))
    return out


def rewrite(job):
    """Rewrite one series (see the module doc) -> number of files written (0 on failure)."""
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from dicom_io import pixel_spacing

    dst, rows, dedupe = job
    try:
        os.makedirs(dst, exist_ok=True)
        if dedupe:                                   # the lowest InstanceNumber per slice position
            best = {}
            for f, key, inst in rows:
                k = key if key is not None else f
                if k not in best or inst < best[k][1]:
                    best[k] = (f, inst)
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
            for t in ("PerFrameFunctionalGroupsSequence", "SharedFunctionalGroupsSequence", "NumberOfFrames",
                      "PlanarConfiguration", "PixelData", "FloatPixelData", "DoubleFloatPixelData"):
                if t in d:
                    del d[t]
            for k, (img, geo) in enumerate(frames):
                e = copy.deepcopy(d)
                for name, v in geo.items():
                    setattr(e, name, v)
                e.SamplesPerPixel, e.Rows, e.Columns = 1, img.shape[0], img.shape[1]
                if "PixelSpacing" not in e:
                    e.PixelSpacing = pixel_spacing(e)
                if img.dtype == np.uint8:
                    e.BitsAllocated, e.BitsStored, e.HighBit, e.PixelRepresentation = 8, 8, 7, 0
                else:
                    img = img.astype(np.int16 if img.dtype.kind == "i" else np.uint16)
                    e.BitsAllocated, e.BitsStored, e.HighBit = 16, 16, 15
                    e.PixelRepresentation = int(img.dtype == np.int16)
                if not str(e.get("PhotometricInterpretation", "MONOCHROME2")).startswith("MONOCHROME"):
                    e.PhotometricInterpretation = "MONOCHROME2"
                e.PixelData = np.ascontiguousarray(img).tobytes()
                e["PixelData"].VR = "OB" if img.dtype == np.uint8 else "OW"
                e.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                if len(frames) > 1:
                    e.InstanceNumber = k + 1
                    e.SOPInstanceUID = generate_uid()
                name = os.path.splitext(os.path.basename(f))[0] + (f"_f{k:04d}" if len(frames) > 1 else "")
                e.save_as(os.path.join(dst, name + ".tmp"), enforce_file_format=True)
                os.replace(os.path.join(dst, name + ".tmp"), os.path.join(dst, name + ".dcm"))
                written += 1
        return written
    except Exception:  # noqa: BLE001
        return 0


def sanitize(df, out_dir, workers=4):
    """Series list (`infer.pipeline.list_series`: StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane, folder)
    -> the same list, planes normalised / filled and unusual series pointing to their rewritten copy."""
    df = df.copy()
    blank = lambda v: not isinstance(v, str) or not v.strip()  # noqa: E731
    df["Anatomical_Plane"] = ["" if blank(v) else CANON.get(v.strip()[:1].upper(), v) for v in df.Anatomical_Plane]
    with ProcessPoolExecutor(workers) as ex:
        info = list(ex.map(inspect, df.folder, chunksize=8))
    jobs, counts = {}, {}
    for i, (issues, plane, rows) in zip(df.index, info):
        for k in issues:
            counts[k] = counts.get(k, 0) + 1
        if blank(df.at[i, "Anatomical_Plane"]) and plane:
            df.at[i, "Anatomical_Plane"] = plane
        if rows and set(issues) & {"compressed", "multiframe", "color", "duplicate_positions", "no_spacing"}:
            jobs[i] = (os.path.join(out_dir, df.at[i, "SeriesInstanceUID"]), rows, "duplicate_positions" in issues)
    with ProcessPoolExecutor(workers) as ex:
        for i, written in zip(jobs, ex.map(rewrite, jobs.values())):
            if written:
                df.at[i, "folder"] = jobs[i][0]
            else:                                    # nothing usable: keep the original series
                shutil.rmtree(jobs[i][0], ignore_errors=True)
    print(f"sanitize: {len(df)} series, issues {counts}, rewritten {sum(f.startswith(out_dir) for f in df.folder)}",
          flush=True)
    return df
