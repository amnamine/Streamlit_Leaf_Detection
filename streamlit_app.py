import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍅 Tomato Leaf Disease Classifier",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f2027 0%, #1a3a2a 50%, #0f2027 100%); }
    .main-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }
    .hero-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #56ab2f, #a8e063);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .section-label {
        color: #a8e063;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stRadio"] > label { color: rgba(255,255,255,0.85) !important; }
    div[data-testid="stRadio"] div[role="radio"] { border-color: rgba(255,255,255,0.2) !important; }
    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(90deg, #56ab2f, #a8e063);
        color: #0f2027;
        font-weight: 800;
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 12px;
        cursor: pointer;
        letter-spacing: 0.05em;
        transition: all 0.2s;
        margin-top: 0.5rem;
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(86,171,47,0.4);
    }
    .result-box {
        background: rgba(86,171,47,0.12);
        border: 1px solid rgba(86,171,47,0.35);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-disease { font-size: 1.6rem; font-weight: 800; color: #a8e063; margin-bottom: 0.3rem; }
    .result-conf { font-size: 1rem; color: rgba(255,255,255,0.65); }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #56ab2f, #a8e063); }
    section[data-testid="stFileUploadDropzone"] {
        background: rgba(255,255,255,0.04) !important;
        border: 2px dashed rgba(168,224,99,0.4) !important;
        border-radius: 14px !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: white !important;
    }
    p, li, span { color: rgba(255,255,255,0.85); }
    .uploaded-img { border-radius: 14px; overflow: hidden; }
    .model-badge {
        display: inline-block;
        background: rgba(168,224,99,0.15);
        border: 1px solid rgba(168,224,99,0.3);
        color: #a8e063;
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .error-box {
        background: rgba(255,80,80,0.1);
        border: 1px solid rgba(255,80,80,0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #ff9999;
    }
    hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ─── Class Names ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Miner",
    "Tomato___Mosaic_virus",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Model Definitions ────────────────────────────────────────────────────────

def build_densenet121(num_classes):
    from torchvision.models import densenet121
    model = densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model


def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ─── Fix: Cross-version H5 loader ────────────────────────────────────────────
#
# The .h5 was saved with a newer Keras that adds 'quantization_config' to Dense
# layer configs. Older Keras versions don't recognise this kwarg and crash.
# Solution: subclass Dense to silently ignore unknown kwargs, then load with
# that as a custom object.

def _make_compat_dense():
    """
    Returns a Dense subclass that drops unknown kwargs (like quantization_config)
    so an H5 saved by a newer Keras loads cleanly on an older Keras install.
    """
    import tensorflow as tf

    class CompatDense(tf.keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            # Drop any kwarg that Dense doesn't accept in this Keras version
            _KNOWN = {
                "units", "activation", "use_bias",
                "kernel_initializer", "bias_initializer",
                "kernel_regularizer", "bias_regularizer",
                "activity_regularizer", "kernel_constraint",
                "bias_constraint", "name", "trainable", "dtype",
            }
            filtered = {k: v for k, v in kwargs.items() if k in _KNOWN}
            super().__init__(*args, **filtered)

        @classmethod
        def from_config(cls, config):
            # Same filtering at deserialization time
            _KNOWN = {
                "units", "activation", "use_bias",
                "kernel_initializer", "bias_initializer",
                "kernel_regularizer", "bias_regularizer",
                "activity_regularizer", "kernel_constraint",
                "bias_constraint", "name", "trainable", "dtype",
            }
            filtered = {k: v for k, v in config.items() if k in _KNOWN}
            return cls(**filtered)

    return CompatDense


@st.cache_resource(show_spinner=False)
def load_efficientnet(path: str):
    """
    Load the EfficientNetB3 .h5 in a way that is robust to Keras version
    mismatches (specifically the quantization_config key added by Keras 3).
    """
    import tensorflow as tf

    CompatDense = _make_compat_dense()

    # Strategy 1 – custom_objects shim (works for most version gaps)
    try:
        model = tf.keras.models.load_model(
            path,
            custom_objects={"Dense": CompatDense},
            compile=False,
        )
        return model
    except Exception:
        pass

    # Strategy 2 – legacy H5 format via TF's internal loader
    try:
        model = tf.keras.models.load_model(
            path,
            custom_objects={"Dense": CompatDense},
            compile=False,
            safe_mode=False,
        )
        return model
    except Exception:
        pass

    # Strategy 3 – rebuild architecture and load weights only
    # (fallback if config deserialization keeps failing)
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
    )
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.load_weights(path)
    return model


# ─── PyTorch loaders ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_densenet(path: str):
    model = build_densenet121(NUM_CLASSES)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_resnet(path: str):
    model = build_resnet50(NUM_CLASSES)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ─── Transforms & Preprocessing ──────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

torch_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_for_torch(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    return torch_transform(img).unsqueeze(0)


def preprocess_for_tf(img: Image.Image) -> "np.ndarray":
    """
    EfficientNetB3 has built-in rescaling — pass raw 0-255 float32 values.
    """
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


# ─── Prediction Helpers ───────────────────────────────────────────────────────

def predict_torch(model, img: Image.Image):
    tensor = preprocess_for_torch(img)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    return top_idx, top_conf, probs


def predict_tf(model, img: Image.Image):
    arr   = preprocess_for_tf(img)
    probs = model.predict(arr, verbose=0)[0]
    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    return top_idx, top_conf, probs


# ─── UI ───────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🍃 Tomato Leaf Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a leaf image · choose a model · get instant diagnosis</div>', unsafe_allow_html=True)

# ── Step 1 – Upload ──────────────────────────────────────────────────────────
st.markdown('<p class="section-label">① Upload Leaf Image</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

col_img, col_info = st.columns([1, 1], gap="medium")

pil_image = None
if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    with col_img:
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)
    with col_info:
        st.markdown(f"""
        <div class="main-card" style="padding:1.2rem">
            <p class="section-label">Image Info</p>
            <p>📐 Size: {pil_image.width} × {pil_image.height} px</p>
            <p>🗂 File: {uploaded_file.name}</p>
            <p>📦 Type: {uploaded_file.type}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Step 2 – Choose Model ────────────────────────────────────────────────────
st.markdown('<p class="section-label">② Select Model</p>', unsafe_allow_html=True)

MODEL_OPTIONS = {
    "🧠 DenseNet121  (PyTorch  •  leaf_densenet121.pth)":  "densenet",
    "⚡ EfficientNetB3  (TensorFlow  •  leaf_efficientnet.h5)": "efficientnet",
    "🏆 ResNet50  (PyTorch  •  leaf_resnet50.pth)":        "resnet",
}

model_choice_label = st.radio(
    label="",
    options=list(MODEL_OPTIONS.keys()),
    label_visibility="collapsed",
)
model_choice = MODEL_OPTIONS[model_choice_label]

MODEL_DESCRIPTIONS = {
    "densenet":     "DenseNet121 — Dense connections, excellent feature reuse. Fast and lightweight.",
    "efficientnet": "EfficientNetB3 — Compound scaling, state-of-the-art accuracy-to-param ratio.",
    "resnet":       "ResNet50 — Deep residual network with custom head, aggressive augmentation, mixed precision.",
}
st.caption(f"ℹ️ {MODEL_DESCRIPTIONS[model_choice]}")

st.markdown("---")

# ── Step 3 – Model File Path ──────────────────────────────────────────────────
st.markdown('<p class="section-label">③ Model File Path</p>', unsafe_allow_html=True)

DEFAULT_PATHS = {
    "densenet":     "leaf_densenet121.pth",
    "efficientnet": "leaf_efficientnet.h5",
    "resnet":       "leaf_resnet50.pth",
}

model_path = st.text_input(
    label="Path to model weights file",
    value=DEFAULT_PATHS[model_choice],
    placeholder="e.g. /path/to/leaf_efficientnet.h5",
)

st.markdown("---")

# ── Step 4 – Predict ─────────────────────────────────────────────────────────
predict_btn = st.button("🔍  Predict Disease", use_container_width=True)

if predict_btn:
    if pil_image is None:
        st.markdown('<div class="error-box">⚠️ Please upload a leaf image first.</div>', unsafe_allow_html=True)
    elif not model_path.strip():
        st.markdown('<div class="error-box">⚠️ Please provide the model file path.</div>', unsafe_allow_html=True)
    elif not os.path.exists(model_path.strip()):
        st.markdown(
            f'<div class="error-box">⚠️ Model file not found: <code>{model_path}</code><br>'
            f'Place the .pth / .h5 file in the same directory as this script, or provide the full path.</div>',
            unsafe_allow_html=True,
        )
    else:
        path = model_path.strip()
        with st.spinner("Loading model & running inference…"):
            try:
                if model_choice == "densenet":
                    model = load_densenet(path)
                    top_idx, top_conf, probs = predict_torch(model, pil_image)

                elif model_choice == "efficientnet":
                    model = load_efficientnet(path)
                    top_idx, top_conf, probs = predict_tf(model, pil_image)

                else:  # resnet
                    model = load_resnet(path)
                    top_idx, top_conf, probs = predict_torch(model, pil_image)

                predicted_label = CLASS_NAMES[top_idx]
                is_healthy      = "healthy" in predicted_label.lower()
                display_name    = predicted_label.replace("Tomato___", "").replace("_", " ")
                emoji           = "✅" if is_healthy else "🔴"

                st.markdown(f"""
                <div class="result-box">
                    <div class="result-disease">{emoji} {display_name}</div>
                    <div class="result-conf">Confidence: <strong>{top_conf * 100:.1f}%</strong></div>
                    <div class="model-badge">via {model_choice_label.split('(')[0].strip()}</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("#### 📊 Top-5 Predictions")
                top5_idx = np.argsort(probs)[::-1][:5]

                for rank, idx in enumerate(top5_idx):
                    label    = CLASS_NAMES[idx].replace("Tomato___", "").replace("_", " ")
                    conf_val = float(probs[idx])

                    col_lbl, col_bar = st.columns([2, 3])
                    with col_lbl:
                        st.markdown(
                            f"<p style='color:rgba(255,255,255,0.85); margin:0; font-size:0.88rem; padding-top:4px'>"
                            f"{'🥇' if rank==0 else f'{rank+1}.'} {label}</p>",
                            unsafe_allow_html=True,
                        )
                    with col_bar:
                        st.progress(conf_val, text=f"{conf_val*100:.1f}%")

            except Exception as e:
                st.markdown(
                    f'<div class="error-box">❌ Prediction failed:<br><code>{e}</code></div>',
                    unsafe_allow_html=True,
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.78rem'>"
    "Tomato Leaf Disease Classifier · DenseNet121 · EfficientNetB3 · ResNet50"
    "</p>",
    unsafe_allow_html=True,
)