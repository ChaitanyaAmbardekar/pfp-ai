"""PFP-AI web UI. Everything runs locally; photos never leave this computer."""
import csv
import datetime
import random
from pathlib import Path

import cv2
import streamlit as st

from app.advice import build_advice
from app.analyzer import LABELS, PhotoAnalysis, analyze_photo, load_weights
from app.body_pose import PoseDetector
from app.editor import LocalEditor
from app.explain import explain, explain_comparison
from app.face_detector import FaceDetector
from app.image_io import SUPPORTED, load_image_bgr
from app.pfp_simulator import to_circle
from app.ranking import close_call, rank, top_n
from app.suggestions import PRESETS, apply_preset, build_suggestions

UPLOADS = Path("outputs/uploads")
PAIR_FILE = Path("dataset/labels/pairwise_ui.csv")
STYLES = ["(none)", "Chad-lite", "Nonchalant", "Mysterious", "Clean", "Cinematic", "Casual"]

st.set_page_config(page_title="PFP-AI", layout="wide")


@st.cache_resource
def load_models():
    return FaceDetector(), FaceDetector(score_threshold=0.5), PoseDetector()


def circle_rgba(bgr_square, size):
    return cv2.cvtColor(to_circle(bgr_square, size), cv2.COLOR_BGRA2RGBA)


def save_uploads(files):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        p = UPLOADS / Path(f.name).name
        p.write_bytes(f.getbuffer())
        paths.append(p)
    return paths


def show_photo(p):
    try:
        bgr, _ = load_image_bgr(p)
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    except Exception as e:
        st.warning(f"Cannot show {Path(p).name}: {e}")


def run_analysis(paths):
    det, det_small, pose = load_models()
    weights = load_weights()
    results = []
    bar = st.progress(0.0)
    for i, p in enumerate(paths):
        try:
            results.append(analyze_photo(p, det, det_small, pose, weights))
        except Exception as e:
            results.append(PhotoAnalysis(name=p.name, path=str(p), weights=weights,
                                         problem=f"unexpected error: {e}"))
        bar.progress((i + 1) / len(paths))
    bar.empty()
    return rank(results)


def save_pair(a, b, preferred, style):
    PAIR_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not PAIR_FILE.exists()
    with open(PAIR_FILE, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["image_A", "image_B", "preferred", "style", "time", "source"])
        w.writerow([Path(a).stem, Path(b).stem, preferred,
                    "" if style == "(none)" else style,
                    datetime.datetime.now().isoformat(timespec="seconds"), "ui"])


st.title("PFP-AI: profile picture evaluator")
st.caption("Runs on this computer only. Scores rate the photograph as a profile picture, "
           "not the person. Expression and style match are not scored yet.")

tab_rank, tab_pair = st.tabs(["Rank my photos", "Teach it my taste"])

with tab_rank:
    files = st.file_uploader("Upload 1 to 20 photos", accept_multiple_files=True,
                             type=sorted(e.lstrip(".") for e in SUPPORTED))
    if files and len(files) > 20:
        st.error("Please upload 20 photos or fewer.")
    elif files and st.button("Analyze", type="primary"):
        paths = save_uploads(files)
        st.session_state.paths = paths
        with st.spinner("Analyzing locally..."):
            st.session_state.ranked = run_analysis(paths)

    ranked = st.session_state.get("ranked")
    if ranked:
        ok = [a for a in ranked if a.ok]
        for a in ranked:
            if not a.ok:
                st.warning(f"{a.name}: skipped ({a.problem})")
        if not ok:
            st.error("No photo could be scored.")
        else:
            st.subheader("Ranking")
            st.dataframe([{"#": i, "photo": a.name, "score": a.overall,
                           **{LABELS[k]: a.sub[k] for k in LABELS},
                           "framing": a.facts["framing"]}
                          for i, a in enumerate(ok, 1)], hide_index=True)
            if close_call(ranked):
                st.info("Close call: the top two are within 3 points, so treat the order as a toss-up.")

            st.subheader("Top 3")
            cols = st.columns(3)
            for col, a in zip(cols, top_n(ranked, 3)):
                bgr, _ = load_image_bgr(a.path)
                p = a.best.plan
                sq = bgr[p.y:p.y + p.side, p.x:p.x + p.side]
                col.image(circle_rgba(sq, 160), caption=f"{a.name}: {a.overall:.0f}/100")

            st.divider()
            choice = st.selectbox("Look closer at", [a.name for a in ok])
            a = next(x for x in ok if x.name == choice)
            left, right = st.columns([1, 1])

            with left:
                st.subheader("Why")
                st.code(explain(a), language=None)
                if len(ok) > 1 and a is ok[0]:
                    st.code(explain_comparison(ok[0], ok[1]), language=None)
                tips = build_advice(a)
                if tips:
                    st.subheader("Tips for a better PFP")
                    for t in tips:
                        st.write("- " + t)

            with right:
                st.subheader("Suggested edits (you choose)")
                preset = st.selectbox("Style preset", list(PRESETS),
                                      format_func=lambda k: PRESETS[k][0])
                st.caption(PRESETS[preset][1])
                base = build_suggestions(a)
                edits = apply_preset(preset, base, a)
                for e in edits:
                    e.enabled = st.checkbox(f"{e.label}: {e.reason}", value=True,
                                            key=f"{a.name}:{preset}:{e.id}")
                bgr, _ = load_image_bgr(a.path)
                editor = LocalEditor()
                before = editor.apply(bgr, [e for e in base if e.kind == "crop"])
                after = editor.apply(bgr, edits)
                c1, c2 = st.columns(2)
                c1.image(circle_rgba(before, 256), caption="Before")
                c2.image(circle_rgba(after, 256), caption="After")
                st.caption("Small-size preview (64 / 96 / 128 px):")
                small = st.columns(4)
                for col, s in zip(small, (64, 96, 128)):
                    col.image(circle_rgba(after, s))
                ok_png, buf = cv2.imencode(".png", to_circle(after, min(512, after.shape[0])))
                if ok_png:
                    st.download_button("Download this PFP", buf.tobytes(),
                                       file_name=f"{Path(a.name).stem}_pfp.png", mime="image/png")

with tab_pair:
    paths = st.session_state.get("paths", [])
    if len(paths) < 2:
        st.info("Upload and analyze at least 2 photos in the first tab, then come back here.")
    else:
        if "pair" not in st.session_state or any(p not in paths for p in st.session_state.pair):
            st.session_state.pair = random.sample(paths, 2)
        if st.button("New random pair"):
            st.session_state.pair = random.sample(paths, 2)
            st.rerun()
        pa, pb = st.session_state.pair
        st.write("**Which is better as a profile picture?** "
                 "(Judge the photo as a PFP, not as a memory.)")
        ca, cb = st.columns(2)
        with ca:
            st.caption("A")
            show_photo(pa)
        with cb:
            st.caption("B")
            show_photo(pb)
        style = st.selectbox("Optional: which style does the winner better represent?", STYLES)
        b1, b2, b3 = st.columns(3)
        pick = None
        if b1.button("A is better", use_container_width=True): pick = Path(pa).stem
        if b2.button("B is better", use_container_width=True): pick = Path(pb).stem
        if b3.button("Same", use_container_width=True): pick = "same"
        if pick:
            save_pair(pa, pb, pick, style)
            st.session_state.pair = random.sample(paths, 2)
            st.toast(f"Saved to {PAIR_FILE}")
            st.rerun()
        if PAIR_FILE.exists():
            n = max(0, sum(1 for _ in open(PAIR_FILE, encoding="utf-8")) - 1)
            st.caption(f"{n} comparisons saved so far in {PAIR_FILE}")
            