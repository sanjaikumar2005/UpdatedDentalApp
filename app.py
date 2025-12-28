import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import speech_recognition as sr
import numpy as np
import requests
import gdown
import os

# =============================
# CONFIG
# =============================

MODEL_URL = "https://drive.google.com/uc?id=1jFsvVVLK_VBtGiRcHj-Hv0cBOs-FjBCu"
MODEL_PATH = "model.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model (one-time)..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

CLASS_NAMES = ["Calculus", "Gingivitis"]

HF_API_KEY = "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"
API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# =============================
# AI FUNCTION (FREE + REALTIME)
# =============================
def ai_answer(question, language, mode):
    prompt = f"""
You are a dental doctor AI.

Answer clearly.
Language: {language}
Explanation Mode: {mode}

Question:
{question}

If language is Tamil, answer fully in Tamil.
If Patient mode, explain very simply.
If Doctor mode, use medical terms.
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        return "⚠️ AI is busy. Please try again."

    result = response.json()

    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    else:
        return "⚠️ AI response error. Try again."

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

model = load_model()

# =============================
# IMAGE TRANSFORM
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="Dental AI Assistant", layout="centered")
st.title("🦷 Smart Dental Diagnosis App")

# =============================
# IMAGE INPUT
# =============================
st.subheader("📷 Upload Image or Use Camera")

img = st.camera_input("Camera") or st.file_uploader(
    "Upload Image", type=["jpg", "png", "jpeg"]
)

if img:
    image = Image.open(img).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred = np.argmax(probs)

    confidence = probs[pred] * 100
    disease = CLASS_NAMES[pred]

    st.success(f"🧠 Prediction: **{disease}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    if confidence < 60:
        disease = "Healthy"
        st.warning("✅ Teeth look **Healthy**")

    st.subheader("💊 Patient Care Advice")

    if disease == "Calculus":
        st.write("""
• Professional scaling needed  
• Brush twice daily  
• Anti-plaque mouthwash  
• Avoid tobacco  
""")
    elif disease == "Gingivitis":
        st.write("""
• Improve oral hygiene  
• Medicated mouthwash  
• Avoid sugary food  
• Visit dentist if bleeding continues  
""")
    else:
        st.write("""
• Teeth are healthy  
• Continue brushing twice daily  
• Regular dental checkups  
""")

# =============================
# QUESTION SECTION
# =============================
st.divider()
st.subheader("💬 Ask Dental Questions (Voice or Text)")

voice_text = ""

if st.button("🎙️ Speak"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)

    try:
        voice_text = r.recognize_google(audio)
        st.success(f"You said: {voice_text}")
    except:
        st.error("❌ Could not understand voice")

text_question = st.text_input("Or type your question")

question = voice_text if voice_text else text_question

# =============================
# OPTIONS
# =============================
col1, col2 = st.columns(2)
with col1:
    language = st.selectbox("Language", ["English", "Tamil"])
with col2:
    mode = st.selectbox("Explanation Mode", ["Patient", "Doctor"])

# =============================
# AI RESPONSE
# =============================
if question:
    with st.spinner("🤖 AI is answering..."):
        answer = ai_answer(question, language, mode)
        st.info(answer)
