"""Minimal DICOM reader for uncompressed little-endian files: the header fields the pipeline uses + the pixels.

`read(path)` -> dict(InstanceNumber, ImageOrientationPatient, ImagePositionPatient, PixelSpacing, SliceThickness,
Rows, Columns, BitsAllocated, PixelRepresentation, Laterality, pixels (np.ndarray or None)).
One `open().read()` per file; the pixels are np.frombuffer at the PixelData offset (bit-identical to pydicom's
pixel_array for these transfer syntaxes). Anything else (compressed / big endian / odd layouts) raises
`Unsupported`, and the caller falls back to pydicom.
"""

import struct

import numpy as np

EXPLICIT_LE, IMPLICIT_LE = "1.2.840.10008.1.2.1", "1.2.840.10008.1.2"
LONG_VR = {b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"SV", b"UC", b"UN", b"UR", b"UT", b"UV"}
TAGS = {
    (0x0020, 0x0013): "InstanceNumber", (0x0020, 0x0037): "ImageOrientationPatient",
    (0x0020, 0x0032): "ImagePositionPatient", (0x0028, 0x0030): "PixelSpacing", (0x0018, 0x0050): "SliceThickness",
    (0x0028, 0x0010): "Rows", (0x0028, 0x0011): "Columns", (0x0028, 0x0100): "BitsAllocated",
    (0x0028, 0x0103): "PixelRepresentation", (0x0020, 0x0060): "Laterality", (0x0028, 0x0002): "SamplesPerPixel",
    (0x0028, 0x0008): "NumberOfFrames",
}
US = {"Rows", "Columns", "BitsAllocated", "PixelRepresentation", "SamplesPerPixel"}
UNDEFINED = 0xFFFFFFFF


class Unsupported(Exception):
    pass


def _skip_undefined(buf, pos):
    """Position after an undefined-length value (sequence / encapsulated): scan to the matching delimiter."""
    depth = 1
    n = len(buf)
    while pos + 8 <= n:
        g, e, ln = struct.unpack_from("<HHI", buf, pos)
        if g == 0xFFFE:
            pos += 8
            if e == 0xE0DD:                      # sequence delimitation
                depth -= 1
                if depth == 0:
                    return pos
            elif e == 0xE000 and ln != UNDEFINED:    # item with explicit length: skip it
                pos += ln
            elif e == 0xE000:                    # item of undefined length: walk its elements
                depth += 1
                pos = _walk_item(buf, pos)
                depth -= 1
            # E00D item delimitation handled in _walk_item
        else:
            raise Unsupported("bad sequence")
    raise Unsupported("unterminated sequence")


def _walk_item(buf, pos):
    """Elements of an undefined-length item until its item delimitation (explicit VR LE)."""
    n = len(buf)
    while pos + 8 <= n:
        g, e = struct.unpack_from("<HH", buf, pos)
        if g == 0xFFFE and e == 0xE00D:
            return pos + 8
        pos = _skip_element(buf, pos, True)
    raise Unsupported("unterminated item")


def _skip_element(buf, pos, explicit):
    g, e = struct.unpack_from("<HH", buf, pos)
    vr, ln, pos = _len(buf, pos, explicit)
    if ln == UNDEFINED:
        return _skip_undefined(buf, pos)
    return pos + ln


def _len(buf, pos, explicit):
    """-> (vr, value length, value offset) of the element at pos."""
    if explicit:
        vr = buf[pos + 4:pos + 6]
        if vr in LONG_VR:
            return vr, struct.unpack_from("<I", buf, pos + 8)[0], pos + 12
        if vr[0:1].isalpha() and vr[1:2].isalpha():
            return vr, struct.unpack_from("<H", buf, pos + 6)[0], pos + 8
        raise Unsupported("bad VR")
    return None, struct.unpack_from("<I", buf, pos + 4)[0], pos + 8


def parse(buf, pixels=True):
    if buf[128:132] != b"DICM":
        raise Unsupported("no preamble")
    pos, ts = 132, None
    while True:                                  # file meta (group 2, explicit VR LE)
        g, e = struct.unpack_from("<HH", buf, pos)
        if g != 2:
            break
        vr, ln, v = _len(buf, pos, True)
        if e == 0x0010:
            ts = buf[v:v + ln].rstrip(b"\x00 ").decode()
        pos = v + ln
    if ts not in (EXPLICIT_LE, IMPLICIT_LE):
        raise Unsupported(f"transfer syntax {ts}")
    explicit = ts == EXPLICIT_LE
    out, n = {}, len(buf)
    while pos + 8 <= n:
        g, e = struct.unpack_from("<HH", buf, pos)
        vr, ln, v = _len(buf, pos, explicit)
        if (g, e) == (0x7FE0, 0x0010):
            if ln == UNDEFINED:
                raise Unsupported("encapsulated pixels")
            out["_pixel"] = (v, ln)
            break
        if ln == UNDEFINED:
            if not explicit:
                raise Unsupported("implicit undefined length")
            pos = _skip_undefined(buf, v)
            continue
        if v + ln > n:
            raise Unsupported("truncated element")
        name = TAGS.get((g, e))
        if name:
            raw = buf[v:v + ln]
            if name in US:
                out[name] = struct.unpack_from("<H", raw)[0]
            else:
                s = raw.decode("latin-1").strip("\x00 ")
                if name == "Laterality":
                    out[name] = s
                elif name in ("InstanceNumber", "NumberOfFrames"):
                    out[name] = int(float(s)) if s else 0
                elif name == "SliceThickness":
                    out[name] = float(s) if s else None
                else:
                    out[name] = [float(x) for x in s.split("\\")] if s else None
        pos = v + ln
    if "_pixel" not in out:
        raise Unsupported("no pixel data")
    if out.get("SamplesPerPixel", 1) != 1 or out.get("NumberOfFrames", 1) not in (0, 1):
        raise Unsupported("multi-sample / multi-frame")
    if pixels:
        v, ln = out["_pixel"]
        rows, cols, bits = out["Rows"], out["Columns"], out["BitsAllocated"]
        if bits not in (8, 16) or ln < rows * cols * bits // 8:
            raise Unsupported("pixel layout")
        dt = {(8, 0): np.uint8, (8, 1): np.int8, (16, 0): np.uint16, (16, 1): np.int16}[(bits, out["PixelRepresentation"])]
        out["pixels"] = np.frombuffer(buf, dtype=dt, count=rows * cols, offset=v).reshape(rows, cols)
    return out


_SIM = None


def _throttle(nbytes):
    """Benchmark only (KNEE_SIM_MBPS, KNEE_SIM_LAT_MS): emulate a slower filesystem, per process."""
    global _SIM
    import os
    import threading
    import time

    if _SIM is None:
        _SIM = (float(os.environ.get("KNEE_SIM_MBPS", 0)), float(os.environ.get("KNEE_SIM_LAT_MS", 0)) / 1e3,
                threading.Lock(), [time.time()])
    mbps, lat, lock, nxt = _SIM
    if not mbps:
        return
    with lock:                                  # bandwidth: one shared budget per process
        start = max(nxt[0], time.time())
        nxt[0] = start + nbytes / (mbps * 1e6)
    time.sleep(max(0.0, nxt[0] - time.time()) + lat)


def read(path, pixels=True):
    with open(path, "rb") as f:
        buf = f.read()
    _throttle(len(buf))
    return parse(buf, pixels)


def read_header(path, nbytes=16384):
    """Header fields only, from the first nbytes of the file (the whole file when PixelData starts later)."""
    with open(path, "rb") as f:
        buf = f.read(nbytes)
        try:
            return parse(buf, pixels=False)
        except (Unsupported, struct.error):
            return parse(buf + f.read(), pixels=False)
