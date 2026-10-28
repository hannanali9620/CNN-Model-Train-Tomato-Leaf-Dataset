"""
Tomato Leaf Disease Dashboard
Run with:  streamlit run app.py
Required in this folder: tomato_tl2.keras, class_names.json, .streamlit/config.toml
"""
import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

IMG_SIZE = 128
GREEN, RED, AMBER = "#2F7D4F", "#C8432B", "#B7791F"

INFO = {
    "Bacterial_spot": ("Bacterial Spot", "Small, dark, water-soaked spots on the leaf, sometimes with a yellow halo."),
    "Early_blight": ("Early Blight", "Brown spots with concentric rings, usually starting on the older leaves."),
    "Late_blight": ("Late Blight", "Large, irregular, dark water-soaked patches; white mold may show underneath in humid weather."),
    "Leaf_Mold": ("Leaf Mold", "Yellow patches on the upper surface with olive-gray mold underneath."),
    "Septoria_leaf_spot": ("Septoria Leaf Spot", "Many small round spots with light centers and dark edges."),
    "Spider_mites Two-spotted_spider_mite": ("Spider Mites", "Tiny yellow speckles on the leaf, sometimes with fine webbing."),
    "Target_Spot": ("Target Spot", "Brown spots with concentric rings, similar to Early Blight."),
    "Tomato_Yellow_Leaf_Curl_Virus": ("Yellow Leaf Curl Virus", "Leaves curl upward, yellow at the edges, and stay small."),
    "Tomato_mosaic_virus": ("Mosaic Virus", "Light and dark green mottling with a distorted leaf shape."),
    "healthy": ("Healthy", "Even green color with no spots or curled areas."),
}
SIMILAR = {"Early_blight", "Target_Spot", "Septoria_leaf_spot"}


def html(s):
    return "".join(line.strip() for line in s.splitlines())


st.set_page_config(page_title="Tomato leaf check", page_icon="🍅", layout="centered")

st.markdown(html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Figtree:wght@400;500;600&display=swap');
html, body, .stApp, p, div, span, label {font-family:'Figtree', system-ui, sans-serif;}
footer {visibility:hidden;}
.block-container {padding-top:2.2rem; max-width:820px;}
.hero h1 {font-family:'Bricolage Grotesque', Georgia, serif; font-weight:800; font-size:2.5rem; line-height:1.05; margin:0 0 .5rem 0; color:#14301F; letter-spacing:-0.02em;}
.hero p {font-size:1.05rem; color:#3d5646; margin:0 0 1.2rem 0; max-width:34rem;}
[data-testid="stFileUploader"] section {background:#fff; border:2px dashed #9DB8A5; border-radius:18px; padding:1.4rem;}
[data-testid="stImage"] img {border-radius:18px;}
.verdict {background:#fff; border-radius:20px; padding:1.3rem 1.3rem 1.1rem; border-left:8px solid var(--c); box-shadow:0 1px 0 rgba(20,48,31,.06);}
.vtop {display:flex; gap:1rem; align-items:center;}
.ring {position:relative; width:92px; height:92px; flex:none; border-radius:50%; background:conic-gradient(var(--c) calc(var(--p) * 1%), #DCE6D9 0); -webkit-mask:radial-gradient(farthest-side, transparent calc(100% - 11px), #000 calc(100% - 10px)); mask:radial-gradient(farthest-side, transparent calc(100% - 11px), #000 calc(100% - 10px));}
.ringwrap {position:relative; width:92px; height:92px; flex:none;}
.ringnum {position:absolute; inset:0; display:grid; place-items:center; font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:1.35rem; color:var(--c);}
.vlabel {font-size:.9rem; color:#5b7263; margin-bottom:.1rem;}
.vname {font-family:'Bricolage Grotesque', sans-serif; font-weight:800; font-size:1.7rem; line-height:1.1; color:var(--c);}
.vdesc {margin:.9rem 0 0 0; color:#2c4435; font-size:.98rem; line-height:1.5;}
.bars {margin-top:1rem;}
.bname {display:flex; justify-content:space-between; font-size:.92rem; margin-bottom:.2rem; color:#14301F;}
.track {height:9px; background:#DCE6D9; border-radius:9px; margin-bottom:.7rem; overflow:hidden;}
.track i {display:block; height:100%; border-radius:9px; background:#8FB59B;}
.track i.top {background:var(--c);}
.note {border-radius:14px; padding:.75rem 1rem; margin-top:1rem; font-size:.93rem; line-height:1.45;}
.note.warn {background:#FBF0D9; color:#6b4a0c;}
.note.info {background:#E3ECF4; color:#26445f;}
.tips {display:grid; grid-template-columns:repeat(3, 1fr); gap:.8rem; margin-top:1.2rem;}
.tip {background:#fff; border-radius:16px; padding:.9rem 1rem; font-size:.92rem; color:#2c4435; line-height:1.4;}
.tip b {display:block; font-family:'Bricolage Grotesque', sans-serif; font-size:1rem; margin-bottom:.2rem; color:#14301F;}
.fine {font-size:.82rem; color:#5b7263; margin-top:1.4rem;}
@media (max-width:640px) {.tips {grid-template-columns:1fr;} .hero h1 {font-size:2rem;}}
</style>
"""), unsafe_allow_html=True)

st.markdown(html("""
<div class="hero">
<h1>Identify tomato leaf diseases</h1>
<p>Upload a clear photo of a leaf. The model picks the most likely of 10 conditions and shows how confident it is.</p>
</div>
"""), unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("tomato_tl2.keras")
    with open("class_names.json") as f:
        names = json.load(f)
    return model, names


model, class_names = load_assets()


def preprocess(file):
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    square = ImageOps.fit(img, (IMG_SIZE, IMG_SIZE), method=Image.BILINEAR)
    arr = np.expand_dims(np.array(square, dtype="float32"), axis=0)  # Rescaling model ke andar hai
    return img, square, arr


file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if not file:
    st.markdown(html("""
    <div class="tips">
    <div class="tip"><b>One leaf</b>Keep a single leaf in the frame, fully visible.</div>
    <div class="tip"><b>Good light</b>Shoot in daylight. Avoid shadows and glare.</div>
    <div class="tip"><b>Close up</b>Let the leaf fill most of the frame, in focus.</div>
    </div>
    """), unsafe_allow_html=True)
else:
    original, square, arr = preprocess(file)
    probs = model.predict(arr, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    conf = float(probs[top]) * 100
    gap = (float(probs[top]) - float(probs[second])) * 100
    raw = class_names[top]
    name, desc = INFO.get(raw, (raw, ""))

    if conf < 70:
        color, label = AMBER, "Not sure. Closest guess"
    elif raw == "healthy":
        color, label = GREEN, "The leaf looks healthy"
    else:
        color, label = RED, "Possible disease"

    col1, col2 = st.columns([5, 6], gap="large")
    with col1:
        st.image(original, use_container_width=True)

    with col2:
        st.markdown(html(f"""
        <div class="verdict" style="--c:{color}">
        <div class="vtop">
        <div class="ringwrap"><div class="ring" style="--p:{conf:.0f}; --c:{color}"></div><div class="ringnum" style="--c:{color}">{conf:.0f}%</div></div>
        <div><div class="vlabel">{label}</div><div class="vname">{name}</div></div>
        </div>
        <p class="vdesc">{desc}</p>
        </div>
        """), unsafe_allow_html=True)

    notes = ""
    if conf < 70:
        notes += '<div class="note warn">Confidence is low. The photo may not be a tomato leaf, or the light or focus may be off. Do not rely on this result.</div>'
    elif gap < 20:
        a = INFO.get(raw, (raw,))[0]
        b = INFO.get(class_names[second], (class_names[second],))[0]
        notes += f'<div class="note info">The model is torn between <b>{a}</b> and <b>{b}</b>. Keep both in mind.</div>'
    if raw in SIMILAR:
        notes += '<div class="note info">Early Blight, Target Spot and Septoria look alike. In testing the model sometimes mixed them up: only 61% of Early Blight leaves were identified correctly.</div>'

    bars = ""
    for rank, k in enumerate(order[:3]):
        n = INFO.get(class_names[k], (class_names[k],))[0]
        p = probs[k] * 100
        cls = "top" if rank == 0 else ""
        bars += f'<div class="bname"><span>{n}</span><span>{p:.1f}%</span></div><div class="track"><i class="{cls}" style="width:{p:.1f}%"></i></div>'

    st.markdown(html(f'<div class="bars" style="--c:{color}">{bars}</div>{notes}'), unsafe_allow_html=True)

    with st.expander("Probability for all 10 conditions"):
        chart = pd.DataFrame({
            "Condition": [INFO.get(n, (n,))[0] for n in class_names],
            "Probability %": probs * 100,
        }).set_index("Condition")
        st.bar_chart(chart, color=GREEN)

    with st.expander("What the model saw"):
        st.image(square.resize((384, 384), Image.NEAREST), caption="Center crop, 128x128")

with st.expander("About this model"):
    st.markdown(
        "- Trained on **tomato** leaves from the PlantVillage dataset (MobileNetV2, transfer learning).\n"
        "- Accuracy on the held-out test set was **93.3%**. That is measured on clean lab-style images.\n"
        "- On real field photos (cluttered background, shadows, dust) accuracy can be lower.\n"
        "- For a photo of anything other than a tomato leaf it will still pick one of the 10 conditions, and that answer will be wrong."
    )

st.markdown('<div class="fine">This is an estimate, not a diagnosis. Confirm with an agricultural expert before making crop decisions.</div>',
            unsafe_allow_html=True)