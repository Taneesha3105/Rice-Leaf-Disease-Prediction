import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# ===============================
# 🌿 CONFIGURATION
# ===============================

st.set_page_config(
    page_title="🌾Rice Leaf Disease Detector",
    layout="wide",
    page_icon="🌱"
)

# --- Hide Streamlit default menu, footer, and toolbar ---
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ===============================
# 💅 CUSTOM CSS
# ===============================

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    * {
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: #f4fff8;
    }
    .stApp {
        background-image: linear-gradient(120deg, #ffffff 0%, #e8f5e9 100%);
    }
    .big-font {
        font-size: 42px !important;
        font-weight: 700;
        color: #2e7d32;
        text-align: center;
        margin: 20px 0;
    }
    .medium-font {
        font-size: 20px !important;
        color: #388e3c;
        text-align: center;
    }
    .info-box {
        background-color: #fff;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #66bb6a;
    }
    .disease-box {
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        text-align: center;
        font-size: 18px;
    }
    .detected {
        background-color: rgba(102, 187, 106, 0.15);
        border: 2px solid #388e3c;
    }
    .healthy {
        background-color: rgba(129, 199, 132, 0.15);
        border: 2px solid #81c784;
    }
    .stButton>button {
        background-color: #43a047;
        color: white;
        border-radius: 15px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2e7d32;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 🧠 MODEL & DATA LOADING
# ===============================

# Class labels
CLASS_MAPPINGS = {
    0: "bacterial_leaf_blight",
    1: "healthy",
    2: "leaf_scald",
    3: "tungro"
}

# Load fertilizer data (Excel)
@st.cache_data
def load_fertilizer_data(excel_path="fertilizer_dataset_Brief.xlsx"):
    if not os.path.exists(excel_path):
        st.warning(f"⚠️ Fertilizer data file not found: {excel_path}")
        return {}
    df = pd.read_excel(excel_path, engine="openpyxl")

    # Restrict to only 4 diseases
    allowed_diseases = ["Healthy", "Leaf Scald", "Brown Spot", "Tungro"]
    df = df[df["Disease"].isin(allowed_diseases)]

    fertilizer_data = {}
    for _, row in df.iterrows():
        disease_name = row['Disease']
        fertilizer_data[disease_name] = {
            "Nitrogen Fertilizer": f"{row['Nitrogen Fertilizer']} {row['Fertilizer Quantity(kg/acre)']}",
            "Phosphorus Fertilizer": f"{row['Phosphorus Fertilizer']} {row['Fertilizer Quantity(kg/acre)']}",
            "Potassium Fertilizer": f"{row['Potassium Fertilizer']} {row['Fertilizer Quantity(kg/acre)']}",
            "Zinc Fertilizer": f"{row['Zinc Fertilizer']} {row['Fertilizer Quantity(kg/acre)']}"
        }
    return fertilizer_data

fertilizer_recommendations = load_fertilizer_data()

# Load trained ResNet50 model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model_path = "supervised_resnet50_model.pth"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ===============================
# 📷 IMAGE PREDICTION
# ===============================

def predict_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        probs = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
        predicted_class = CLASS_MAPPINGS[predicted.item()]
    return predicted_class, probs

# ===============================
# 🌾 STREAMLIT UI
# ===============================

st.markdown('<div class="big-font">🌾Rice Leaf Disease Detector🌾</div>', unsafe_allow_html=True)
st.markdown('<div class="medium-font">AI-powered system for detecting rice leaf diseases and suggesting fertilizers</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🔍Disease Detection", "🌱Fertilizer Recommendations"])

# -------------------
# 🧠 TAB 1: Detection
# -------------------
with tab1:
    st.markdown('<div class="info-box">Upload a clear image of a rice leaf to detect the disease.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Rice Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="📷 Uploaded Rice Leaf", use_container_width=True)

            with col2:
                with st.spinner("🔍 Analyzing image..."):
                    disease_label, confidence = predict_image(image)

                if disease_label == "healthy":
                    st.markdown(f"""
                    <div class="disease-box healthy">
                        <h3>✅ Predicted Disease: {disease_label.capitalize()}</h3>
                        <p>Confidence: {confidence * 100:.2f}%</p>
                        <p>Your crop appears healthy. No treatment needed!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="disease-box detected">
                        <h3>⚠️ Predicted Disease: {disease_label.replace('_', ' ').title()}</h3>
                        <p>Confidence: {confidence * 100:.2f}%</p>
                        <p>This leaf shows signs of <b>{disease_label.replace('_', ' ').title()}</b>.</p>
                        <p>Check the 'Fertilizer Recommendations' tab for treatment suggestions.</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error processing image: {str(e)}")

# -------------------
# 🌱 TAB 2: Fertilizer
# -------------------
with tab2:
    st.markdown('<div class="info-box">Select a disease to view recommended fertilizers and nutrient details.</div>', unsafe_allow_html=True)
    selected_disease = st.selectbox("Select Disease", options=list(fertilizer_recommendations.keys()))

    if selected_disease:
        fert_info = fertilizer_recommendations[selected_disease]
        st.markdown(f"""
        <div class="info-box">
            <h3>Fertilizer Recommendations for {selected_disease}</h3>
            <ul>
                <li><b>Nitrogen:</b> {fert_info['Nitrogen Fertilizer']}</li>
                <li><b>Phosphorus:</b> {fert_info['Phosphorus Fertilizer']}</li>
                <li><b>Potassium:</b> {fert_info['Potassium Fertilizer']}</li>
                <li><b>Zinc:</b> {fert_info['Zinc Fertilizer']}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
