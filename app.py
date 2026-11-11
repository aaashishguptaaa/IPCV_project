"""
Face Studio Camera
----------------------------------------------------------------------
IPCV course project - Streamlit application.

Tabs
    Studio          stackable 7-stage editing pipeline, presets,
                    before/after slider, quality metrics, export
    Face Analysis   BlazeFace detection, 478-point mesh, head pose,
                    blink/mouth state, face geometry, Haar vs MediaPipe
    IPCV Lab        classical operations from the syllabus, live
    Live Camera     real-time video effects (streamlit-webrtc)
    Batch           apply the current pipeline to many photos -> ZIP
    About           architecture, models, team

Run:  streamlit run app.py
----------------------------------------------------------------------
"""

import base64
import io
import json
import os
import urllib.request
import zipfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

import evaluation
import face_effects as fx
import filters
import ipcv_lab as lab
import metrics
import pipeline as P
from download_models import ModelMissingError, model_status, MODELS_DIR

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(APP_DIR, "saved_images")
SAMPLE_DIR = os.path.join(APP_DIR, "sample_images")
os.makedirs(SAVE_DIR, exist_ok=True)
MAX_DIM = 1400

TEAM = [
    ("Aashish Gupta Rauniyar", "20233501", "Project architecture, multi-scale face detection & pipeline integration"),
    ("Bishal Ramjan Sah", "20233515", "478-point face mesh, head pose estimation & Haar vs MediaPipe benchmarking"),
    ("Amit Kumar", "20233046", "Enhancement filters, beauty filters & image quality metrics (PSNR, SSIM)"),
    ("Ashish Gupta", "20233511", "Portrait segmentation, background replacement & AR facial overlays"),
    ("Anubhav Kaushik", "20233068", "IPCV Lab demonstrations, live camera processing, UI design & testing"),
]
st.set_page_config(page_title="Face Studio Camera", page_icon="📸", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container { padding-top: 3.2rem; }
  .app-title { font-size: 2.1rem; font-weight: 800; margin: 0;
      background: linear-gradient(90deg, #7c4dff, #00c9a7);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .app-sub { color: #8b93a1; margin: -2px 0 0.8rem 0; font-size: 0.95rem; }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.75rem;
      background:#1d2230; color:#9fe8d6; border:1px solid #2d3446; margin-right:6px; }
  .explain { background:#141824; border-left:3px solid #00c9a7; padding:0.7rem 1rem;
      border-radius:6px; color:#cfd6e4; font-size:0.92rem; }
  div[data-testid="stMetricValue"] { font-size: 1.35rem; }
  .footer { color:#6b7280; font-size:0.78rem; text-align:center; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Version-tolerant display helpers
# ----------------------------------------------------------------------
def show(img_bgr, caption=None):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        st.image(rgb, caption=caption, width="stretch")
    except TypeError:
        st.image(rgb, caption=caption, use_container_width=True)


def table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def to_bytes(img_bgr, fmt="PNG", quality=95):
    buf = io.BytesIO()
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil.save(buf, format=fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    return buf.getvalue()


def decode(file_bytes):
    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB")  # honour phone rotation
    return filters.resize_max_dim(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), MAX_DIM)


def model_error(e):
    st.error(f"⚠️ {e}")
    st.info("Fix: connect to the internet once and run `python download_models.py`, then restart the app.")


# ----------------------------------------------------------------------
# Cached models & per-image analysis
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading face detector…")
def get_detector():
    from face_utils import FaceDetector
    return FaceDetector(min_confidence=0.5)


@st.cache_resource(show_spinner="Loading 478-point face mesh…")
def get_mesh():
    from face_utils import FaceMeshDetector
    return FaceMeshDetector(max_faces=5)


@st.cache_resource(show_spinner="Loading person segmenter…")
def get_segmenter():
    from face_utils import PersonSegmenter
    return PersonSegmenter()


@st.cache_resource(show_spinner=False)
def get_haar():
    from face_utils import HaarFaceDetector
    return HaarFaceDetector()


def det_cfg():
    return (st.session_state.get("det_multiscale", True), st.session_state.get("det_tile_thr", 0.6))


@st.cache_data(show_spinner=False, max_entries=16)
def _faces(img, multiscale, tile_thr):
    d = get_detector()
    d.tile_min_score = tile_thr
    return d.detect(img, multiscale=multiscale)


def cached_faces(img):
    return _faces(img, *det_cfg())


@st.cache_data(show_spinner=False, max_entries=16)
def _landmarks(img, multiscale, tile_thr):
    return get_mesh().landmarks(img, _faces(img, multiscale, tile_thr))


def cached_landmarks(img):
    return _landmarks(img, *det_cfg())


@st.cache_data(show_spinner=False, max_entries=8)
def cached_mask(img):
    return get_segmenter().get_mask(img)


class StreamlitModels:
    """Adapter so live/batch code can use the cached Streamlit models."""
    detector = property(lambda self: get_detector())
    mesh = property(lambda self: get_mesh())
    segmenter = property(lambda self: get_segmenter())


def run(img, s, bg=None):
    return P.run_pipeline(img, s, get_landmarks=lambda: cached_landmarks(img),
                          get_mask=lambda: cached_mask(img), get_faces=lambda: cached_faces(img),
                          bg_image=bg)


# ----------------------------------------------------------------------
# Session-state backed settings (so presets & reset can drive widgets)
# ----------------------------------------------------------------------
KEY = "s_"


def _write_settings(values):
    for k, v in values.items():
        st.session_state[KEY + k] = list(v) if isinstance(v, list) else v


def apply_preset(name):
    _write_settings(P.settings_from(P.PRESETS[name]))
    st.session_state["active_preset"] = name


def reset_all():
    _write_settings(P.settings_from())
    st.session_state["active_preset"] = None


if "initialised" not in st.session_state:
    reset_all()
    st.session_state["initialised"] = True


def current_settings():
    return {k: st.session_state[KEY + k] for k in P.DEFAULTS}


# ----------------------------------------------------------------------
# Before / after slider (pure HTML + CSS, no extra dependency)
# ----------------------------------------------------------------------
def compare_slider(before, after, height_hint=760):
    def b64(x):
        return base64.b64encode(to_bytes(filters.resize_max_dim(x, 1100), "JPEG", 88)).decode()
    h, w = before.shape[:2]
    height = int(min(height_hint, 900 * h / w)) + 12
    html = f"""
    <div id="wrap" style="position:relative;width:100%;max-width:{int((height - 12) * w / h)}px;margin:auto;
         aspect-ratio:{w}/{h};user-select:none;overflow:hidden;border-radius:10px">
      <img src="data:image/jpeg;base64,{b64(after)}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain">
      <div id="clip" style="position:absolute;inset:0;width:50%;overflow:hidden">
        <img src="data:image/jpeg;base64,{b64(before)}" id="bimg" style="position:absolute;top:0;left:0;height:100%;object-fit:contain">
      </div>
      <div id="bar" style="position:absolute;top:0;bottom:0;left:50%;width:3px;background:#00c9a7;cursor:ew-resize">
        <div style="position:absolute;top:50%;left:-13px;width:28px;height:28px;border-radius:50%;
             background:#00c9a7;transform:translateY(-50%);box-shadow:0 0 8px #0008"></div></div>
      <span style="position:absolute;top:8px;left:10px;background:#000a;color:#fff;font:12px sans-serif;padding:2px 8px;border-radius:4px">Before</span>
      <span style="position:absolute;top:8px;right:10px;background:#000a;color:#fff;font:12px sans-serif;padding:2px 8px;border-radius:4px">After</span>
    </div>
    <script>
      const wrap=document.getElementById('wrap'),clip=document.getElementById('clip'),
            bar=document.getElementById('bar'),bimg=document.getElementById('bimg');
      function size(){{ bimg.style.width=wrap.clientWidth+'px'; }}
      function move(x){{ const r=wrap.getBoundingClientRect();
        let p=Math.min(Math.max((x-r.left)/r.width,0),1)*100; clip.style.width=p+'%'; bar.style.left=p+'%'; }}
      let drag=false; wrap.addEventListener('mousedown',e=>{{drag=true;move(e.clientX)}});
      window.addEventListener('mouseup',()=>drag=false);
      window.addEventListener('mousemove',e=>{{if(drag)move(e.clientX)}});
      wrap.addEventListener('touchmove',e=>move(e.touches[0].clientX));
      window.addEventListener('resize',size); size();
    </script>"""
    components.html(html, height=height)


# ----------------------------------------------------------------------
# Header & image source
# ----------------------------------------------------------------------
st.markdown('<p style="margin:0"><span style="font-size:2rem">📸</span> <span class="app-title">Face Studio Camera</span></p>',
            unsafe_allow_html=True)
st.markdown('<p class="app-sub">Image Processing &amp; Computer Vision project — OpenCV + MediaPipe '
            '(deep-learning face analysis, no Haar cascades in the main pipeline)</p>', unsafe_allow_html=True)

sb = st.sidebar
sb.header("📷 Image source")
source = sb.radio("Input", ["Upload a photo", "Use camera", "Sample image"], horizontal=False,
                  label_visibility="collapsed")
raw = None
if source == "Upload a photo":
    up = sb.file_uploader("Upload", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
    raw = up.getvalue() if up else None
elif source == "Use camera":
    cam = sb.camera_input("Take a picture")
    raw = cam.getvalue() if cam else None
else:
    samples = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    pick = sb.selectbox("Sample", samples)
    with open(os.path.join(SAMPLE_DIR, pick), "rb") as fh:
        raw = fh.read()

original = decode(raw) if raw else None

# ----------------------------------------------------------------------
# Sidebar: pipeline controls
# ----------------------------------------------------------------------
sb.markdown("---")
sb.header("🎛️ Pipeline")
sb.caption("Stages run in order 1 → 7 and stack. Used by Studio and Batch.")

pc = sb.columns(2)
for i, name in enumerate(P.PRESETS):
    pc[i % 2].button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,))
sb.button("↺ Reset all", on_click=reset_all, type="secondary")
if st.session_state.get("active_preset"):
    sb.caption(f"Preset applied: **{st.session_state['active_preset']}** (tweak freely)")

with sb.expander("1 · Enhance", expanded=True):
    st.checkbox("Auto-enhance (CLAHE)", key=KEY + "auto_enhance")
    st.slider("Brightness", -100, 100, key=KEY + "brightness")
    st.slider("Contrast", -100, 100, key=KEY + "contrast")
    st.slider("Saturation", -100, 100, key=KEY + "saturation")
    st.slider("Gamma", 0.3, 3.0, step=0.05, key=KEY + "gamma")
    st.slider("Sharpness (unsharp mask)", -2.0, 3.0, step=0.1, key=KEY + "sharpness")
    st.slider("Warmth (cool ↔ warm)", -100, 100, key=KEY + "warmth")
with sb.expander("2 · Beauty (face mesh)"):
    st.slider("Skin smoothing", 0.0, 1.0, step=0.05, key=KEY + "beauty_smooth")
    st.slider("Skin brighten", 0, 30, key=KEY + "beauty_brighten")
    c1, c2 = st.columns([3, 1])
    c1.slider("Lip tint", 0.0, 1.0, step=0.05, key=KEY + "lip_strength")
    c2.color_picker("Lip", key=KEY + "lip_color", label_visibility="hidden")
    c1, c2 = st.columns([3, 1])
    c1.slider("Eye colour", 0.0, 1.0, step=0.05, key=KEY + "eye_strength")
    c2.color_picker("Eye", key=KEY + "eye_color", label_visibility="hidden")
with sb.expander("3 · Background (segmentation)"):
    st.selectbox("Mode", P.BG_MODES, key=KEY + "bg_mode")
    st.slider("Blur strength", 1, 40, key=KEY + "bg_strength")
    st.slider("Edge softness (feather)", 0, 40, key=KEY + "bg_feather")
    st.color_picker("Backdrop colour", key=KEY + "bg_color")
    bg_file = st.file_uploader("Custom background image", type=["jpg", "jpeg", "png", "webp"])
bg_image = decode(bg_file.getvalue()) if bg_file else None
with sb.expander("4 · Artistic style"):
    st.selectbox("Style", list(filters.STYLES), key=KEY + "style")
with sb.expander("5 · Finishing"):
    st.slider("Vignette", 0.0, 3.0, step=0.1, key=KEY + "vignette")
    st.slider("Film grain", 0, 40, key=KEY + "grain")
with sb.expander("6 · AR overlays (face mesh)"):
    st.multiselect("Overlays", list(fx.AR_OVERLAYS), key=KEY + "ar_overlays")
with sb.expander("7 · Privacy"):
    st.radio("Anonymise faces", P.PRIVACY_MODES, key=KEY + "privacy", horizontal=True)
with sb.expander("⚙️ Face detection settings"):
    st.checkbox("Multi-scale detection (finds small faces in group photos)", True, key="det_multiscale")
    st.slider("Tile confidence threshold", 0.5, 0.95, 0.6, step=0.05, key="det_tile_thr",
              help="Higher = fewer false boxes, but more small faces missed.")

settings = current_settings()

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tabs = st.tabs(["🎛️ Studio", "🧠 Face Analysis", "🔬 IPCV Lab", "🔍 Image Forgery", "🎥 Live Camera", "📦 Batch", "ℹ️ About"])


def need_image():
    st.info("👈 Choose **Upload**, **Camera** or **Sample image** in the sidebar to begin.")


# ============================== STUDIO ==============================
with tabs[0]:
    if original is None:
        need_image()
    else:
        try:
            with st.spinner("Processing…"):
                result = run(original, settings, bg_image)
        except ModelMissingError as e:
            model_error(e)
        except Exception as e:
            st.error(f"Face Analysis error: {e}")
        except Exception as e:
            st.error(f'Error during face analysis: {e}')
            result = None
        if result is not None:
            out = result.image
            for n in result.notes:
                st.warning(n)
            view = st.radio("View", ["Before / after slider", "Side by side", "Result only"],
                            horizontal=True, label_visibility="collapsed")
            if view == "Before / after slider":
                compare_slider(original, out)
            elif view == "Side by side":
                a, b = st.columns(2)
                with a:
                    show(original, "Original")
                with b:
                    show(out, "Result")
            else:
                show(out)

            active = [n for n, _ in result.timings]
            st.markdown(" ".join(f'<span class="pill">{n}</span>' for n in active) or
                        '<span class="pill">No stages active</span>', unsafe_allow_html=True)

            d1, d2, d3, d4 = st.columns([1, 1.3, 1.6, 2])
            fmt = d1.selectbox("Format", ["PNG", "JPEG"], label_visibility="collapsed")
            ext = "png" if fmt == "PNG" else "jpg"
            d2.download_button("⬇️ Download", to_bytes(out, fmt), f"face_studio_{datetime.now():%Y%m%d_%H%M%S}.{ext}",
                               mime=f"image/{'png' if fmt == 'PNG' else 'jpeg'}")
            if d3.button("💾 Save to saved_images/"):
                path = os.path.join(SAVE_DIR, f"edited_{datetime.now():%Y%m%d_%H%M%S}.png")
                cv2.imwrite(path, out)
                st.success(f"Saved `{os.path.relpath(path, APP_DIR)}`")

            with st.expander("📊 Results & metrics", expanded=False):
                cmp = metrics.compare(original, out)
                m1, m2, m3 = st.columns(3)
                m1.metric("PSNR", "∞ (identical)" if np.isinf(cmp["PSNR (dB)"]) else f"{cmp['PSNR (dB)']:.2f} dB")
                m2.metric("SSIM", f"{cmp['SSIM']:.3f}")
                m3.metric("Total processing", f"{result.total_ms:.0f} ms")
                so, sr = metrics.image_stats(original), metrics.image_stats(out)
                table(pd.DataFrame({"Measure": list(so), "Original": [round(v, 2) for v in so.values()],
                                    "Result": [round(v, 2) for v in sr.values()],
                                    "Change": [f"{sr[k] - so[k]:+.2f}" for k in so]}))
                ho = {k: v[1:255] for k, v in metrics.histograms(original).items()}
                hr = {k: v[1:255] for k, v in metrics.histograms(out).items()}
                hc1, hc2 = st.columns(2)
                with hc1:
                    st.caption("Original histogram (B, G, R; bins 1–254)")
                    st.line_chart(pd.DataFrame(ho), color=["#4f7cff", "#3ccf6e", "#ff5a5a"], height=200)
                with hc2:
                    st.caption("Result histogram (B, G, R; bins 1–254)")
                    st.line_chart(pd.DataFrame(hr), color=["#4f7cff", "#3ccf6e", "#ff5a5a"], height=200)
                if result.timings:
                    st.caption("Per-stage timing")
                    table(pd.DataFrame([{"Stage": n, "Time (ms)": round(t, 1)} for n, t in result.timings]))
                st.caption("PSNR/SSIM compare the result with the original: high = small change. "
                           "Use them to show how strongly each filter alters the image.")

# =========================== FACE ANALYSIS ===========================
with tabs[1]:
    if original is None:
        need_image()
    else:
        mode = st.radio("Analysis", ["Detection (BlazeFace)", "478-point face mesh", "Head pose · eyes · mouth",
                                     "Face geometry", "Haar vs MediaPipe"], horizontal=True)
        try:
            if mode == "Detection (BlazeFace)":
                faces = cached_faces(original)
                c1, c2 = st.columns([3, 2])
                with c1:
                    show(get_detector().draw_detections(original, faces))
                with c2:
                    st.metric("Faces detected", len(faces))
                    if faces:
                        table(pd.DataFrame([{"Face": i + 1, "Confidence": f"{f['score']:.2%}",
                                             "Box (x, y, w, h)": str(f["bbox"])} for i, f in enumerate(faces)]))
                    st.markdown('<div class="explain">BlazeFace is a lightweight single-shot CNN detector. '
                                'It returns a box, a confidence score and 6 keypoints (eyes, nose tip, mouth, '
                                'ear tragions).</div>', unsafe_allow_html=True)
            elif mode == "478-point face mesh":
                style = st.selectbox("Draw", ["tesselation", "contours", "irises", "points", "all"])
                c1, c2 = st.columns([3, 2])
                with c1:
                    show(get_mesh().draw_mesh(original, style, cached_landmarks(original)))
                with c2:
                    lms = cached_landmarks(original)
                    st.metric("Faces with mesh", len(lms))
                    st.markdown('<div class="explain">FaceLandmarker regresses <b>478 3-D points</b> per face '
                                '(468 surface points + 10 iris points). These points drive every face effect: '
                                'skin mask, lips, eyes, AR overlays and head pose.</div>', unsafe_allow_html=True)
            elif mode == "Head pose · eyes · mouth":
                lms = cached_landmarks(original)
                out, info = original.copy(), []
                for i, lm in enumerate(lms, 1):
                    pose, eyes, mar = fx.head_pose(lm), fx.eye_state(lm), fx.mouth_aspect_ratio(lm)
                    fx.draw_head_pose(out, lm, pose)
                    info.append((i, pose, eyes, mar))
                fx.draw_face_info(out, info)
                c1, c2 = st.columns([3, 2])
                with c1:
                    show(out)
                with c2:
                    if not lms:
                        st.warning("No face found.")
                    for i, pose, eyes, mar in info:
                        st.subheader(f"Face {i}: {fx.describe_pose(pose)}")
                        a, b, c = st.columns(3)
                        a.metric("Yaw", f"{pose['yaw']:+.1f}°")
                        b.metric("Pitch", f"{pose['pitch']:+.1f}°")
                        c.metric("Roll", f"{pose['roll']:+.1f}°")
                        a, b = st.columns(2)
                        a.metric("Eyes", "Closed" if eyes["closed"] else "Open", f"EAR {eyes['ear']:.3f}",
                                 delta_color="off")
                        b.metric("Mouth", "Open" if mar > 0.35 else "Closed", f"MAR {mar:.3f}", delta_color="off")
                    st.markdown('<div class="explain"><b>Head pose</b>: an orthonormal head frame is built from 3-D '
                                'landmarks (eye corners 33→263, forehead 10→chin 152) by Gram–Schmidt; yaw, pitch and '
                                'roll are read from the rotation matrix.<br><b>EAR</b> = (|p2−p6| + |p3−p5|) / '
                                '(2|p1−p4|); eyes are "closed" below 0.20.</div>', unsafe_allow_html=True)
            elif mode == "Face geometry":
                lms = cached_landmarks(original)
                if not lms:
                    st.warning("No face found.")
                for i, lm in enumerate(lms, 1):
                    g = fx.face_geometry(lm)
                    st.subheader(f"Face {i}")
                    a, b, c, d = st.columns(4)
                    a.metric("Face width", f"{g['face_width_px']:.0f} px")
                    b.metric("Face height", f"{g['face_height_px']:.0f} px")
                    c.metric("Height / width", f"{g['height_width_ratio']:.2f}")
                    d.metric("Symmetry", f"{g['symmetry_pct']:.1f}%")
                st.markdown('<div class="explain">Symmetry compares distances of five mirrored landmark pairs '
                            '(eye corners, mouth corners, cheeks, brows, jaw) from the forehead–chin mid-line.'
                            '</div>', unsafe_allow_html=True)
            else:
                c1, c2 = st.columns(2)
                get_detector().tile_min_score = det_cfg()[1]
                mp_f, mp_ms = evaluation.run_detector(get_detector(), original)
                sp_f, sp_ms = evaluation.run_detector(evaluation.SinglePass(get_detector()), original)
                hr_f, hr_ms = evaluation.run_detector(get_haar(), original)
                with c1:
                    show(get_detector().draw_detections(original, mp_f, label_keypoints=False),
                         f"MediaPipe BlazeFace multi-scale — {len(mp_f)} face(s), {mp_ms:.1f} ms "
                         f"(single pass: {len(sp_f)} face(s), {sp_ms:.1f} ms)")
                with c2:
                    show(get_haar().draw_detections(original, hr_f), f"Haar cascade — {len(hr_f)} face(s), {hr_ms:.1f} ms")
                st.markdown('<div class="explain">BlazeFace <i>short-range</i> resizes the whole frame to 128×128, '
                            'so it is excellent for selfies but misses small faces in group photos. The app therefore '
                            'also runs it on overlapping 2×2, 3×3 and 4×4 tiles and merges boxes with non-maximum '
                            'suppression (multi-scale).</div>', unsafe_allow_html=True)
                st.markdown("#### Robustness test on this image")
                exp = st.number_input("Faces actually in the photo", 1, 20, max(1, len(mp_f)))
                if st.button("▶ Run robustness test (12 conditions)"):
                    with st.spinner("Running 3 detectors on 12 degraded versions…"):
                        rows = evaluation.robustness_table(original, get_detector(), get_haar(), exp, repeats=1)
                    s = evaluation.summarise(rows, exp)
                    a, b, c = st.columns(3)
                    a.metric(f"BlazeFace multi-scale · {s['MediaPipe mean ms']} ms", s["MediaPipe detection rate"])
                    b.metric(f"BlazeFace single pass · {s['Single-pass mean ms']} ms", s["Single-pass detection rate"])
                    c.metric(f"Haar cascade · {s['Haar mean ms']} ms", s["Haar detection rate"])
                    df = pd.DataFrame(rows)
                    for col in ("MediaPipe OK", "Single-pass OK", "Haar OK"):
                        df[col] = df[col].map({True: "✅", False: "❌"})
                    df.columns = [c.replace("MediaPipe", "Multi-scale") for c in df.columns]
                    table(df)
                    st.download_button("⬇️ Download results CSV", df.to_csv(index=False).encode(), "robustness.csv")
        except ModelMissingError as e:
            model_error(e)

# ============================== IPCV LAB ==============================
with tabs[2]:
    if original is None:
        need_image()
    else:
        concept = st.selectbox("Concept", ["Colour spaces", "Histogram equalisation", "Noise & denoising",
                                           "Edge detection", "Thresholding", "Morphology",
                                           "Frequency domain (FFT)", "Geometric transforms"])
        img = filters.resize_max_dim(original, 900)
        ctrl, view = st.columns([1, 3])
        with ctrl:
            if concept == "Colour spaces":
                out, txt = lab.color_space(img, st.radio("Space", ["RGB", "HSV", "LAB", "YCrCb"]))
                pair = False
            elif concept == "Histogram equalisation":
                m = st.radio("Method", ["Global", "CLAHE"])
                clip = st.slider("CLAHE clip limit", 1.0, 8.0, 2.0) if m == "CLAHE" else 2.0
                tile = st.slider("Tile grid", 2, 16, 8) if m == "CLAHE" else 8
                out, txt = lab.histogram_equalization(img, m, clip, tile)
                base, pair = lab._bgr(lab._gray(img)), True
            elif concept == "Noise & denoising":
                kind = st.radio("Noise", ["Gaussian", "Salt & Pepper"])
                amt = st.slider("Noise amount", 0, 80, 25)
                meth = st.selectbox("Denoiser", ["Median", "Mean (box)", "Gaussian", "Bilateral", "Non-local Means"])
                k = st.slider("Kernel size", 3, 15, 5, step=2)
                noisy = lab.add_noise(img, kind, amt)
                out, txt = lab.denoise(noisy, meth, k)
                base, pair = noisy, True
                txt += f"  PSNR noisy → {metrics.psnr(img, noisy):.1f} dB, denoised → {metrics.psnr(img, out):.1f} dB."
            elif concept == "Edge detection":
                m = st.radio("Operator", ["Sobel", "Prewitt", "Laplacian", "Canny"])
                lo = st.slider("Canny low", 0, 255, 80) if m == "Canny" else 80
                hi = st.slider("Canny high", 0, 255, 160) if m == "Canny" else 160
                out, txt = lab.edges(img, m, lo, hi)
                base, pair = img, True
            elif concept == "Thresholding":
                m = st.radio("Method", ["Otsu", "Adaptive (Gaussian)", "Manual"])
                v = st.slider("T", 0, 255, 127) if m == "Manual" else 127
                out, txt = lab.threshold(img, m, v)
                base, pair = lab._bgr(lab._gray(img)), True
            elif concept == "Morphology":
                op = st.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top-hat"])
                k = st.slider("Structuring element", 3, 21, 5, step=2)
                it = st.slider("Iterations", 1, 5, 1)
                out, txt = lab.morphology(img, op, k, it)
                pair = False
            elif concept == "Frequency domain (FFT)":
                kind = st.radio("Filter", ["Low-pass", "High-pass"])
                r = st.slider("Cut-off radius D0", 5, 150, 30)
                n = st.slider("Butterworth order", 1, 6, 2)
                out, txt = lab.fft_filter(img, kind, r, n)
                pair = False
            else:
                ang = st.slider("Rotation (deg)", -180, 180, 15)
                sc = st.slider("Scale", 0.3, 2.0, 1.0)
                fl = st.radio("Flip", ["None", "Horizontal", "Vertical"])
                out, txt = lab.geometric(img, ang, sc, fl)
                base, pair = img, True
        with view:
            if pair:
                a, b = st.columns(2)
                with a:
                    show(base, "Input")
                with b:
                    show(out, "Output")
            else:
                show(out)
            st.markdown(f'<div class="explain">{txt}</div>', unsafe_allow_html=True)

# ======================== IMAGE FORGERY DETECTION ========================
with tabs[3]:
    st.subheader("🔍 Image Forgery & Tamper Detection")
    st.markdown(
        "Directly analyzes the active image from the sidebar using the **Digital Forensics Engine** "
        "(ELA, SIFT Copy-Move, Deep Learning CNN, PRNU Sensor Noise, JPEG Ghost & FFT Frequency Analysis)."
    )

    if original is None:
        need_image()
    else:
        top_c1, top_c2 = st.columns([3, 1])
        with top_c1:
            st.caption(f"Active Image from Sidebar: {original.shape[1]}×{original.shape[0]} px (Unified image source — no re-upload required)")
        with top_c2:
            st.markdown(
                '<a href="https://image-tamper-detection.onrender.com" target="_blank" style="text-decoration:none; font-size:0.85rem;">🌐 Standalone App ↗</a>',
                unsafe_allow_html=True,
            )

        run_btn = st.button("🚀 Run Full Forensic Analysis on Active Image", type="primary", use_container_width=True)

        img_key = f"forgery_{original.shape}_{int(original.mean() * 1000)}"
        if run_btn:
            st.session_state["forgery_active_key"] = img_key

        should_analyze = (st.session_state.get("forgery_active_key") == img_key) or run_btn

        if not should_analyze:
            st.info("👆 Click **Run Full Forensic Analysis on Active Image** above to evaluate this photo for tampering.")
            show(original, caption="Active Image (Awaiting Analysis)")
        else:
            with st.spinner("Analyzing image across 7 forensic techniques via live detection backend…"):
                encoded_bytes = to_bytes(original, fmt="JPEG", quality=88)
                boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
                body = (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="image"; filename="active_input.jpg"\r\n'
                    f"Content-Type: image/jpeg\r\n\r\n"
                ).encode("utf-8") + encoded_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

                req = urllib.request.Request(
                    "https://image-tamper-detection.onrender.com/analyze",
                    data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "FaceStudioCamera/1.0"},
                )

                res = None
                try:
                    with urllib.request.urlopen(req, timeout=35) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                except Exception as api_err:
                    st.warning(f"Live cloud backend connection delay ({api_err}). Running local ELA & SIFT analysis...")

            if res and "verdict" in res:
                verdict = res.get("verdict", "UNKNOWN")
                risk = res.get("risk", {})
                risk_label = risk.get("label", "Medium")
                risk_level = risk.get("level", "MEDIUM")
                conf = res.get("confidence", 0)

                v_col1, v_col2 = st.columns([1.5, 3])
                with v_col1:
                    if verdict == "REAL":
                        st.success(f"### ✅ Verdict: REAL (Authentic)\n**Risk:** {risk_label} ({risk_level})")
                    else:
                        st.error(f"### 🚨 Verdict: FAKE (Manipulated)\n**Risk:** {risk_label} ({risk_level})")
                with v_col2:
                    st.metric("Integrity Confidence", f"{conf:.1f}%")
                    st.progress(min(1.0, max(0.0, conf / 100.0)))

                ai_exp = res.get("ai_explanation", {})
                if isinstance(ai_exp, dict) and ai_exp.get("summary"):
                    st.markdown(f'<div class="explain"><b>Forensic Summary:</b> {ai_exp["summary"]}</div>', unsafe_allow_html=True)
                flags = res.get("all_flags", [])
                if flags:
                    pills = " ".join([f'<span class="pill" style="border-color:#ff7b72; color:#ffa198;">⚠️ {f}</span>' for f in flags])
                    st.markdown(f"<div style='margin: 0.6rem 0;'>{pills}</div>", unsafe_allow_html=True)

                st.markdown("---")

                st.subheader("🖼️ Forensic Evidence & Localization Heatmaps")
                v_tabs = st.tabs(["🔥 Tamper Heatmap", "🧪 ELA Map", "🎯 SIFT Matches", "〰️ Noise Variance", "👻 JPEG Ghost", "📊 FFT Spectrum"])

                def show_b64(b64_str, caption=None):
                    if not b64_str:
                        st.info(f"No visual map generated for {caption or 'this technique'}.")
                        return
                    try:
                        clean_b64 = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
                        raw = base64.b64decode(clean_b64)
                        pil_im = Image.open(io.BytesIO(raw))
                        try:
                            st.image(pil_im, caption=caption, width="stretch")
                        except TypeError:
                            st.image(pil_im, caption=caption, use_container_width=True)
                    except Exception as err:
                        st.warning(f"Could not render {caption or 'evidence image'}: {err}")

                with v_tabs[0]:
                    h_b64 = res.get("images", {}).get("heatmap") or res.get("ela", {}).get("image")
                    show_b64(h_b64, "Localized Suspect Regions (Red/Yellow = Tampered)")
                with v_tabs[1]:
                    ela_b64 = res.get("ela", {}).get("image")
                    ela_pct = res.get("ela", {}).get("percent", 0)
                    show_b64(ela_b64, f"Error Level Analysis Map (Discrepancy: {ela_pct}%)")
                with v_tabs[2]:
                    sift_b64 = res.get("sift", {}).get("image")
                    sift_match = res.get("sift", {}).get("match_count", 0)
                    is_clone = res.get("sift", {}).get("is_copy_move", False)
                    show_b64(sift_b64, f"SIFT Keypoint Clones (Matches: {sift_match}, Copy-Move: {is_clone})")
                with v_tabs[3]:
                    noise_b64 = res.get("noise", {}).get("noise_map")
                    noise_pct = res.get("noise", {}).get("percent", 0)
                    show_b64(noise_b64, f"PRNU Sensor Noise Map (Inconsistency: {noise_pct}%)")
                with v_tabs[4]:
                    ghost_b64 = res.get("ghost", {}).get("image")
                    ghost_pct = res.get("ghost", {}).get("percent", 0)
                    show_b64(ghost_b64, f"JPEG Ghost Secondary Compression Map ({ghost_pct}%)")
                with v_tabs[5]:
                    fft_b64 = res.get("frequency", {}).get("image")
                    fft_pct = res.get("frequency", {}).get("percent", 0)
                    show_b64(fft_b64, f"FFT 2D Frequency Spectrum ({fft_pct}%)")

                st.subheader("📊 7-Technique Scorecard")
                scores = [
                    {"Technique": "Error Level Analysis (ELA)", "Score": f"{res.get('ela', {}).get('percent', 0)}%", "Status": "Analyzed"},
                    {"Technique": "SIFT Copy-Move Detection", "Score": f"{res.get('sift', {}).get('match_count', 0)} matches", "Status": "Detected" if res.get('sift', {}).get('is_copy_move') else "Clean"},
                    {"Technique": "Deep Learning (CNN)", "Score": f"{res.get('cnn', {}).get('percent', 0)}%", "Status": "Inference Complete"},
                    {"Technique": "PRNU Sensor Noise Inconsistency", "Score": f"{res.get('noise', {}).get('percent', 0)}%", "Status": "Mapped"},
                    {"Technique": "JPEG Ghost Analysis", "Score": f"{res.get('ghost', {}).get('percent', 0)}%", "Status": "Checked"},
                    {"Technique": "FFT Frequency Spectrum", "Score": f"{res.get('frequency', {}).get('percent', 0)}%", "Status": "Transformed"},
                    {"Technique": "EXIF Metadata Inspection", "Score": f"{res.get('metadata', {}).get('percent', 0)}%", "Status": "Inspected"},
                ]
                table(pd.DataFrame(scores))

            else:
                st.subheader("🔬 Local ELA Inspection")
                from PIL import ImageChops, ImageEnhance
                f_c1, f_c2 = st.columns(2)
                with f_c1:
                    show(original, "Original Active Image")
                with f_c2:
                    pil_orig = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                    buf = io.BytesIO()
                    pil_orig.save(buf, "JPEG", quality=90)
                    buf.seek(0)
                    resaved = Image.open(buf)
                    diff = ImageChops.difference(pil_orig, resaved)
                    diff_enhanced = ImageEnhance.Brightness(diff).enhance(15.0)
                    ela_local = cv2.cvtColor(np.array(diff_enhanced), cv2.COLOR_RGB2BGR)
                    show(ela_local, "Local ELA Error Map (15x scale)")

# ============================ LIVE CAMERA ============================
with tabs[4]:
    try:
        import av
        from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer, WebRtcMode
        import live_effects as live
        HAVE_WEBRTC = True
    except Exception:  # noqa: BLE001
        HAVE_WEBRTC = False
    if not HAVE_WEBRTC:
        st.warning("Live video in the browser needs `pip install streamlit-webrtc`.")
        st.info("Alternatively run the desktop live window: `python live_camera.py`")
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            live_mode = st.selectbox("Live effect", live.LIVE_MODES)
            mirror = st.checkbox("Mirror (selfie view)", True)
            st.caption("Click **START** and allow camera access.")
            st.caption("Tip: the desktop window `python live_camera.py` is faster and needs no browser.")
        with c2:
            RTC_CONFIGURATION = RTCConfiguration(
                {
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]},
                        {"urls": ["stun:global.stun.twilio.com:3478"]},
                    ]
                }
            )

            # Eagerly load model singletons on main thread for thread-safety
            det_inst = get_detector()
            mesh_inst = get_mesh()
            seg_inst = get_segmenter()

            class EagerModels:
                detector = det_inst
                mesh = mesh_inst
                segmenter = seg_inst

            class Proc(VideoProcessorBase):
                def __init__(self):
                    self.lp = live.LiveProcessor(EagerModels())

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    try:
                        img = self.lp.process(img)
                    except Exception as exc:  # noqa: BLE001
                        cv2.putText(img, f"Error: {exc}"[:80], (10, 30), 0, 0.6, (0, 0, 255), 2)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            ctx = webrtc_streamer(
                key="live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=Proc,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if ctx.video_processor:
                ctx.video_processor.lp.mode = live_mode
                ctx.video_processor.lp.mirror = mirror

            with st.expander("📸 Alternative: Snap from Webcam (100% Reliable without WebRTC)"):
                snap = st.camera_input("Take photo with effect", key="tab3_snap")
                if snap:
                    s_img = decode(snap.getvalue())
                    if mirror:
                        s_img = cv2.flip(s_img, 1)
                    proc_inst = live.LiveProcessor(EagerModels())
                    proc_inst.mode = live_mode
                    proc_inst.mirror = False
                    st.image(proc_inst.process(s_img), channels="BGR", use_container_width=True)

# =============================== BATCH ===============================
with tabs[5]:
    st.markdown("Apply the **current sidebar pipeline** to many photos at once and download a ZIP.")
    if settings == P.settings_from():
        st.info("The pipeline is empty: pick a preset or move some sliders in the sidebar first.")
    files = st.file_uploader("Photos", type=["jpg", "jpeg", "png", "webp", "bmp"], accept_multiple_files=True,
                             key="batch_files")
    if files and st.button(f"▶ Process {len(files)} photo(s)"):
        buf, rows = io.BytesIO(), []
        prog = st.progress(0.0)
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for i, f in enumerate(files):
                img = decode(f.getvalue())
                try:
                    r = run(img, settings, bg_image)
                    z.writestr(os.path.splitext(f.name)[0] + "_studio.png", to_bytes(r.image))
                    rows.append({"File": f.name, "Size": f"{img.shape[1]}×{img.shape[0]}",
                                 "Stages": len(r.timings), "Time (ms)": round(r.total_ms, 1),
                                 "SSIM vs original": round(metrics.ssim(img, r.image), 3),
                                 "Notes": "; ".join(r.notes)})
                except ModelMissingError as e:
                    model_error(e)
                    break
                prog.progress((i + 1) / len(files))
        if rows:
            table(pd.DataFrame(rows))
            st.download_button("⬇️ Download ZIP", buf.getvalue(), f"face_studio_batch_{datetime.now():%Y%m%d_%H%M%S}.zip",
                               mime="application/zip")

# =============================== ABOUT ===============================
with tabs[6]:
    a, b = st.columns([3, 2])
    with a:
        st.subheader("Processing pipeline")
        st.code("Input (upload / camera / sample)\n  │  EXIF-rotate, resize ≤ 1400 px\n"
                "  ├─► FaceLandmarker ── 478 landmarks ──┐\n"
                "  ├─► Selfie segmenter ── person mask ──┤  (computed once, cached)\n"
                "  ├─► BlazeFace ── face boxes ──────────┤\n"
                "  ▼                                     ▼\n"
                "1 Enhance → 2 Beauty → 3 Background → 4 Style → 5 Finishing → 6 AR → 7 Privacy\n"
                "  ▼\nOutput + metrics (PSNR, SSIM, histograms, timing) → PNG / JPEG / ZIP", language=None)
        st.subheader("Team (Section A · Team No. 1)")
        table(pd.DataFrame(TEAM, columns=["Member", "Roll no.", "Contribution"]))
    with b:
        st.subheader("Model files")
        stat = model_status()
        table(pd.DataFrame([{"File": k, "Status": "✅ present" if v else "❌ missing"} for k, v in stat.items()]))
        st.caption(f"Folder: `{MODELS_DIR}`")
        if not all(stat.values()):
            st.warning("Run `python download_models.py` while online so the demo works offline.")
        st.subheader("Libraries")
        import mediapipe
        st.write(f"OpenCV {cv2.__version__} · MediaPipe {mediapipe.__version__} · Streamlit {st.__version__}")

st.markdown('<p class="footer">Face Studio Camera · IPCV Project · Streamlit + OpenCV + MediaPipe</p>',
            unsafe_allow_html=True)
