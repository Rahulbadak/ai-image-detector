import numpy as np
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = "detector_v2_effnetb0.keras"
IMG_SIZE = (224, 224)
UNSURE_BELOW = 65.0     # isse kam confidence pe "Uncertain" dikhayega
# Class order: FAKE=0, REAL=1, isliye model ka output = P(REAL)

st.set_page_config(page_title="AI Image Detector", page_icon="🕵️", layout="centered")


@st.cache_resource
def load_model():
    import keras
    return keras.saving.load_model(MODEL_PATH, compile=False)


def preprocess(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)    # training jaisa hi resize
    return np.asarray(img, dtype="float32")[None, ...]   # EfficientNet khud rescale karta hai


st.title("🕵️ AI Image Detector")
st.caption("Upload an image to check whether it is real or AI-generated.")

file = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png", "webp"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        model = load_model()
        p_real = float(model.predict(preprocess(img), verbose=0)[0][0])

    is_real = p_real >= 0.5
    conf = (p_real if is_real else 1 - p_real) * 100

    if conf < UNSURE_BELOW:
        st.warning(f"Prediction: 🟡 UNCERTAIN (leaning {'REAL' if is_real else 'AI-GENERATED'})\n\n"
                   f"Confidence: {conf:.1f}%")
    elif is_real:
        st.success(f"Prediction: 🟢 REAL IMAGE\n\nConfidence: {conf:.1f}%")
    else:
        st.error(f"Prediction: 🔴 AI-GENERATED IMAGE\n\nConfidence: {conf:.1f}%")

    st.progress(min(max(p_real, 0.0), 1.0), text=f"P(real) = {p_real*100:.1f}%")

st.divider()
st.caption("Model: EfficientNet-B0 fine-tuned on mixed real/AI images (224x224).")
with st.expander("Limitations"):
    st.write(
        "Test accuracy is about 82-84% on mixed images, lower on new or unseen generators "
        "and on heavily edited photos. Treat the output as an estimate, not proof."
    )