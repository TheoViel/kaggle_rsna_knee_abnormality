"""
Browse a study's MRI series with the attention-derived bounding boxes on top.

Run it from the repository root:

    streamlit run segmentation_viewer.py

Adapted from the cell-tracking trajectory viewer: the same plotly view that keeps
its zoom and pan while you scrub, a PNG background so each move sends little,
and steppers that take the arrow keys (left / right: series, down / up: slice).

What is drawn comes from `src/attention_boxes.py` (output/attention_boxes/<run>/):
  - sagittal and coronal series: boxes of the findings whose attention map was
    localised in that series (diffuse attention gives no box);
  - axial series: the sagittal and coronal boxes of the study registered onto the
    axial grid and intersected (only where both views agree);
  - optionally, the attention heatmap of one class (sagittal / coronal series),
    to judge whether a box is earned.
A box is drawn on every slice of its slice range, labelled with its class and the
study-level predicted probability.

The localizer thumbnails show where the current slice cuts the other planes,
from the full DICOM geometry, so oblique stacks give tilted lines.
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib import colormaps
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import knee_registration as kr  # noqa: E402

OUTPUT_DIR = ROOT / "output"
BOXES_DIR = OUTPUT_DIR / "attention_boxes"
ROI_LABELS_DIR = OUTPUT_DIR / "roi" / "labels"
ROI_ATTENTION_DIR = OUTPUT_DIR / "roi" / "attention" / "ens8" / "maps"
ROI_CLASSES = FINDINGS[:7] + ["Baker's"]
SERIES_INDEX = OUTPUT_DIR / "series_index.csv"

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
            "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# One distinct colour per finding, readable on a grey MR slice.
CLASS_COLORS = dict(zip(FINDINGS, [
    "#ff4fd8", "#ffd92f", "#ff8c1a", "#29d8ff", "#ff5c5c", "#7cff6b",
    "#c58cff", "#4d9dff", "#ffb3e6", "#00e6b8", "#ffef7a", "#ff2e2e"]))
HEAT = colormaps["inferno"]

VIEW_PIXELS = 900
DISPLAY_SHARE = (5, 2)
LOCALIZER_PIXELS = 220


# --------------------------------------------------------------------------- data


@st.cache_data(show_spinner=False)
def series_index():
    return pd.read_csv(SERIES_INDEX)


@st.cache_data(show_spinner=False)
def study_labels():
    """Per study: radiologist labels (58 studies, NaN elsewhere) and Claude report labels."""
    gt = pd.read_csv(ROOT / "input" / "train.csv").drop(columns=["Report"]).set_index("StudyInstanceUID")
    claude = pd.read_csv(ROOT / "input" / "full_claude_context_v2_labels.csv").set_index("StudyInstanceUID")
    return gt, claude


@st.cache_data(show_spinner=False)
def box_runs():
    """Box sources: ROI labels / U-Net predictions (roi/...), attention boxes (attention/...)."""
    roi = sorted((f"roi/{p.stem}" for p in ROI_LABELS_DIR.glob("*.csv") if not p.stem.startswith("_")),
                 reverse=True)
    att = sorted((f"attention/{p.parent.name}" for p in BOXES_DIR.glob("*/boxes.csv")), reverse=True)
    return roi + att


@st.cache_data(show_spinner=False)
def load_boxes(source):
    """(boxes with a `prob` column for the label text, studies the source covers)."""
    kind, name = source.split("/", 1)
    if kind == "roi":
        boxes = pd.read_csv(ROI_LABELS_DIR / f"{name}.csv")
        if "conf" in boxes:              # U-Net predictions: one box per class, all shown
            boxes = boxes[boxes.z0.notna()].copy()
            boxes["prob"] = boxes.confidence if "confidence" in boxes else boxes.conf
        else:
            boxes["prob"] = boxes.score
        for c in ["z0", "z1", "y0", "y1", "x0", "x1"]:
            boxes[c] = boxes[c].astype(int)
        boxes["source"] = boxes.get("source", name)
        return boxes, sorted(boxes.StudyInstanceUID.unique())
    boxes = pd.read_csv(BOXES_DIR / name / "boxes.csv")
    info = pd.read_csv(BOXES_DIR / name / "series_info.csv")
    return boxes, sorted(info.StudyInstanceUID.unique())


@st.cache_data(show_spinner=False)
def attention_maps(source, series_uid):
    """{slices, xy_attn (K, F, h, w), z_attn (K, F), rows, cols, classes} or None."""
    kind, name = source.split("/", 1)
    path = (ROI_ATTENTION_DIR / f"{series_uid}.npz" if kind == "roi"
            else BOXES_DIR / name / "maps" / f"{series_uid}.npz")
    if not path.exists():
        return None
    d = np.load(path)
    if kind == "roi":                    # roi.extract format: 7 ROI classes, keys xy / z
        return {"slices": d["slices"], "xy_attn": d["xy"], "z_attn": d["z"], "rows": d["rows"],
                "cols": d["cols"], "classes": ROI_CLASSES[:d["xy"].shape[0]]}
    return {**{k: d[k] for k in d.files}, "classes": FINDINGS}


@st.cache_data(show_spinner=False)
def series_description(series_uid):
    m = pd.read_pickle(OUTPUT_DIR / "metadata_shards" / f"{series_uid}.pkl")
    return str(m.SeriesDescription.iloc[0]) if "SeriesDescription" in m else ""


@st.cache_resource(show_spinner=False)
def volume(series_uid):
    """The raw (slices, rows, cols) series, memory-mapped: slices are read on demand."""
    return np.load(OUTPUT_DIR / "series_npy" / f"{series_uid}.npy", mmap_mode="r")


@st.cache_data(show_spinner=False)
def contrast_limits(series_uid, low=1.0, high=99.5):
    """Display range from percentiles of a subsample, as the raw range is set by a few bright voxels."""
    v = volume(series_uid)
    sample = np.asarray(v[:: max(1, len(v) // 8), ::4, ::4], dtype=np.float32)
    lo, hi = np.percentile(sample, [low, high])
    return float(lo), float(hi)


@st.cache_resource(show_spinner=False)
def geometry_image(series_uid):
    """An empty sitk image carrying the series geometry, for mapping between grids."""
    import SimpleITK as sitk

    r = series_index().set_index("SeriesInstanceUID").loc[series_uid]
    g = kr.series_geometry(series_uid)
    img = sitk.Image([int(r.cols), int(r.rows), int(r.n_slices)], sitk.sitkUInt8)
    img.SetSpacing(g["spacing"])
    img.SetDirection(g["direction"])
    img.SetOrigin(g["origin"])
    return img


# ------------------------------------------------------------------------ drawing


def _png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def grey_slice(series_uid, z):
    lo, hi = contrast_limits(series_uid)
    plane = np.asarray(volume(series_uid)[z], dtype=np.float32)
    return (np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def heat_slice(maps, k, z):
    """3D attention (z weight x in-plane map) of class k on slice z, on the slice grid.

    Scaled by the series-wide maximum, so slices the class ignores stay dark.
    None when no frame is centred on that slice (the trimmed ends).
    """
    frames = np.flatnonzero(maps["slices"] == z)
    if len(frames) == 0:
        return None
    f = int(frames[0])
    xy = maps["xy_attn"][k].astype(np.float32)
    zw = maps["z_attn"][k]
    vmax = float((xy.reshape(len(xy), -1).max(1) * zw).max())
    a = xy[f] * zw[f] / max(vmax, 1e-12)
    # The maps cover the slice zero-padded to a centred square: upsample, crop.
    rows, cols = int(maps["rows"]), int(maps["cols"])
    s = max(rows, cols)
    up = np.asarray(Image.fromarray(a).resize((s, s), Image.BILINEAR))
    y0, x0 = (s - rows) // 2, (s - cols) // 2
    return np.clip(up[y0:y0 + rows, x0:x0 + cols], 0, 1)


@st.cache_data(show_spinner=False)
def slice_png(series_uid, z, run=None, heat_class=None):
    """One slice as a data URI, with the attention heatmap of one class blended in."""
    grey = grey_slice(series_uid, z)
    maps = attention_maps(run, series_uid) if heat_class else None
    heat = (heat_slice(maps, maps["classes"].index(heat_class), z)
            if maps is not None and heat_class in maps["classes"] else None)
    if heat is None:
        return _png(grey)
    rgb = grey.astype(np.float32)[..., None].repeat(3, -1)
    alpha = (0.65 * heat)[..., None]
    rgb = rgb * (1 - alpha) + HEAT(heat)[..., :3] * 255 * alpha
    return _png(rgb.round().astype(np.uint8))


def render(series_uid, z, boxes, run, heat_class, show_text):
    height, width = volume(series_uid).shape[1:]
    figure = go.Figure()
    figure.add_layout_image(dict(
        source=slice_png(series_uid, z, run, heat_class), xref="x", yref="y", x=-0.5, y=-0.5,
        sizex=width, sizey=height, xanchor="left", yanchor="top", sizing="stretch",
        layer="below"))
    on_slice = boxes[(boxes.z0 <= z) & (boxes.z1 >= z)]
    for b in on_slice.to_dict("records"):
        colour = CLASS_COLORS[b["class"]]
        figure.add_shape(type="rect", xref="x", yref="y", x0=b["x0"] - 0.5, x1=b["x1"] + 0.5,
                         y0=b["y0"] - 0.5, y1=b["y1"] + 0.5, line=dict(color=colour, width=2))
        if show_text:
            figure.add_annotation(x=b["x0"] - 0.5, y=b["y0"] - 0.5, xref="x", yref="y",
                                  text=f"{b['class']} {b['prob']:.2f}",
                                  showarrow=False, xanchor="left", yanchor="bottom",
                                  font=dict(color=colour, size=12), bgcolor="rgba(0,0,0,0.5)")
    # One invisible point per box for the hover read-out (shapes have no hover).
    if len(on_slice):
        figure.add_trace(go.Scatter(
            x=(on_slice.x0 + on_slice.x1) / 2, y=(on_slice.y0 + on_slice.y1) / 2, mode="markers",
            marker=dict(size=14, opacity=0),
            customdata=np.stack([on_slice["class"], on_slice.prob.round(2),
                                 on_slice.z0, on_slice.z1, on_slice.source], 1),
            hovertemplate="%{customdata[0]}  p=%{customdata[1]}<br>slices %{customdata[2]}-"
                          "%{customdata[3]}  (%{customdata[4]})<extra></extra>",
            showlegend=False))
    figure.update_xaxes(range=[-0.5, width - 0.5], visible=False, constrain="domain")
    figure.update_yaxes(range=[height - 0.5, -0.5], visible=False, scaleanchor="x",
                        scaleratio=1, constrain="domain")
    figure.update_layout(
        height=VIEW_PIXELS, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="black",
        plot_bgcolor="black", dragmode="pan", hovermode="closest",
        # Keeps zoom and pan while the slice moves; a new series resets it.
        uirevision=series_uid)
    return figure, on_slice


@st.cache_data(show_spinner=False)
def localizer_png(other_uid, current_uid, z):
    """Middle slice of another series, with the line where slice z of the current one cuts it."""
    other = volume(other_uid)
    k = len(other) // 2
    img = Image.fromarray(grey_slice(other_uid, k)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for k2, p0, p1 in kr.slice_lines(geometry_image(other_uid), k, geometry_image(current_uid)):
        if k2 == z:
            draw.line([p0, p1], fill=(255, 60, 60), width=max(2, img.width // 160))
    img.thumbnail((LOCALIZER_PIXELS, LOCALIZER_PIXELS))
    buffer = io.BytesIO()
    img.save(buffer, format="png")
    return buffer.getvalue()


# ----------------------------------------------------------------------- controls


def _step(key, options, delta):
    """Move a slider one option along, from the button beside it."""
    index = options.index(st.session_state[key]) + delta
    st.session_state[key] = options[int(np.clip(index, 0, len(options) - 1))]


def _stepper(label, options, default, key, back, forward, format_func=str):
    """A slider with a button either side of it, each moving it exactly one step (arrow keys too)."""
    st.session_state.setdefault(key, default)
    if st.session_state[key] not in options:
        st.session_state[key] = default
    minus, track, plus = st.columns([1, 7, 1], vertical_alignment="bottom")
    with track:
        chosen = st.select_slider(label, options=options, key=key, format_func=format_func)
    index = options.index(chosen)
    minus.button("−", key=f"{key} back", width="stretch", disabled=index == 0,
                 shortcut=back, help=f"{label} − 1   ({back} arrow)",
                 on_click=_step, args=(key, options, -1))
    plus.button("+", key=f"{key} forward", width="stretch", disabled=index == len(options) - 1,
                shortcut=forward, help=f"{label} + 1   ({forward} arrow)",
                on_click=_step, args=(key, options, 1))
    return chosen


def opening_slice(series_boxes, depth):
    """A slice worth opening on: the centre of the most confident box, or the middle."""
    if series_boxes.empty:
        return depth // 2
    b = series_boxes.sort_values("prob", ascending=False).iloc[0]
    return int((b.z0 + b.z1) // 2)


@st.fragment
def view(study_uid, series_rows, boxes, run, heat_class, show_text):
    """The slice view and the steppers that move through the study.

    A fragment, so that moving a stepper reruns this and nothing else, and the
    chart is handed new data in place rather than rebuilt.
    """
    display, controls = st.columns(DISPLAY_SHARE)
    uids = series_rows.SeriesInstanceUID.tolist()
    info = series_rows.set_index("SeriesInstanceUID")
    n_boxes = boxes.groupby("SeriesInstanceUID").size()

    def series_name(uid):
        n = int(n_boxes.get(uid, 0))
        return f"{'●' if n else '○'} {info.loc[uid].Anatomical_Plane[:3]} {series_description(uid)[:18]}"

    with controls:
        st.markdown("#### Navigate")
        default = max(uids, key=lambda u: n_boxes.get(u, 0))
        uid = _stepper("series", uids, default, f"series {study_uid}", "Left", "Right",
                       format_func=series_name)
        r = info.loc[uid]
        series_boxes = boxes[boxes.SeriesInstanceUID == uid]
        depth = int(r.n_slices)
        z = _stepper("slice", list(range(depth)), opening_slice(series_boxes, depth),
                     f"slice {uid}", "Down", "Up")
        source = ("registered from sagittal + coronal" if r.Anatomical_Plane == "Axial"
                  else run)
        st.caption(f"{r.Anatomical_Plane} · {'fluid-sensitive' if r.Fluid_Sensitive else 'not fluid'}"
                   f"{' · fat-sat' if r.Fat_Suppression else ''} · {int(r.rows)}×{int(r.cols)}×{depth}"
                   f"<br>{len(series_boxes)} boxes in this series ({source})", unsafe_allow_html=True)

    figure, on_slice = render(uid, z, series_boxes, run, heat_class, show_text)
    with display:
        st.plotly_chart(
            figure, width="stretch", key="slice", theme=None,
            config={"scrollZoom": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"]})

    with controls:
        if heat_class and r.Anatomical_Plane == "Axial":
            st.caption("No heatmap on axial series: their boxes come from registration.")
        st.markdown("#### Localizer")
        others = series_rows[series_rows.Anatomical_Plane != r.Anatomical_Plane]
        others = others.sort_values("Fluid_Sensitive", ascending=False).drop_duplicates("Anatomical_Plane")
        cols = st.columns(max(1, len(others)))
        for col, o in zip(cols, others.itertuples()):
            col.image(localizer_png(o.SeriesInstanceUID, uid, z),
                      caption=o.Anatomical_Plane, width="stretch")
        st.caption("Scroll to zoom, drag to pan, double-click to reset. "
                   "← → series, ↓ ↑ slice. Hover a box for its slice range.")


# --------------------------------------------------------------------------- page


def main():
    st.set_page_config(page_title="Knee findings viewer", layout="wide")
    # Enough top padding to clear Streamlit's header bar, which otherwise hides the first heading.
    st.markdown("<style>.block-container {padding-top: 3.2rem;}</style>", unsafe_allow_html=True)

    gt, claude = study_labels()
    with st.sidebar:
        st.header("Studies")
        runs = box_runs()
        if not runs:
            st.error(f"No boxes under {BOXES_DIR}: run src/attention_boxes.py first.")
            return
        run = st.selectbox("boxes", runs,
                           help="roi/<name>: output/roi/labels/<name>.csv -- ROI U-Net labels "
                                "(roundN) or predictions (roundN_pred); attention/<run>: "
                                "output/attention_boxes/<run>, from src/attention_boxes.py")
        boxes, studies = load_boxes(run)

        positive = st.multiselect("positive for (radiologist label)", FINDINGS)
        if positive:
            studies = [s for s in studies if s in gt.index and (gt.loc[s, positive] == 1).all()]
        if not studies:
            st.error("No study matches.")
            return
        study_uid = st.selectbox(f"study ({len(studies)})", studies,
                                 format_func=lambda s: f"..{s[-16:]}")

        study_boxes = boxes[boxes.StudyInstanceUID == study_uid]
        lab = pd.DataFrame({"Claude": claude.loc[study_uid, FINDINGS].astype(float)})
        if study_uid in gt.index and gt.loc[study_uid, FINDINGS].notna().all():
            lab["GT"] = gt.loc[study_uid, FINDINGS].astype(int)
        lab["boxes"] = study_boxes.groupby("class").size().reindex(FINDINGS).fillna(0).astype(int)
        st.dataframe(lab.style.format({"Claude": "{:.2f}", "GT": "{:d}"})
                     .background_gradient(cmap="Reds", vmin=0, vmax=1, subset=["Claude"]),
                     height=458, width="stretch",
                     # Narrow value columns, or the sidebar cuts the last one off.
                     column_config={c: st.column_config.Column(width=48)
                                    for c in ["Claude", "GT", "boxes"]})

        st.header("Display")
        present = [c for c in FINDINGS if c in set(boxes["class"])]
        shown = st.multiselect("findings", present, default=present)
        heat_class = st.selectbox("attention heatmap", [None] + FINDINGS,
                                  format_func=lambda c: "none" if c is None else c,
                                  help="The 3D attention of one class on sagittal / coronal "
                                       "series: concentrated where a box was made, diffuse where not.")
        show_text = st.toggle("box labels", value=True)
        st.markdown(" ".join(
            f"<span style='color:{CLASS_COLORS[c]};font-size:0.8rem'>■ {c}</span>" for c in shown),
            unsafe_allow_html=True)

    series_rows = series_index()
    series_rows = series_rows[series_rows.StudyInstanceUID == study_uid].copy()
    series_rows["plane_order"] = series_rows.Anatomical_Plane.map({"Sagittal": 0, "Coronal": 1, "Axial": 2})
    series_rows = series_rows.sort_values(["plane_order", "Fluid_Sensitive"], ascending=[True, False])
    view(study_uid, series_rows, study_boxes[study_boxes["class"].isin(shown)], run, heat_class,
         show_text)


if __name__ == "__main__":
    main()
