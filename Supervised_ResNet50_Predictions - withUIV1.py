import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Rice Leaf Disease Detector", layout="wide")

st.title("🌾 Rice Leaf Disease Detector")
st.write("AI-powered system for detecting rice leaf diseases and suggesting fertilizers")

# -------------------------------
# CLASS LABELS
# -------------------------------
class_mappings = {
    0: "bacterial_leaf_blight",
    1: "healthy",
    2: "leaf_scald",
    3: "tungro"
}

# -------------------------------
# LOAD FILES SAFELY
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "supervised_resnet50_model.pth")
CSV_PATH = os.path.join(BASE_DIR, "fertilizer_dataset_Brief.csv")

# -------------------------------
# LOAD FERTILIZER DATA
# -------------------------------
@st.cache_data
def load_fertilizer_data():
    try:
        df = pd.read_csv(CSV_PATH)
        fertilizer_data = {}

        for _, row in df.iterrows():
            disease_name = row['Disease']
            fertilizer_data[disease_name] = {
                "Nitrogen": f"{row['Nitrogen Fertilizer']} ({row['Fertilizer Quantity(kg/acre)']} kg/acre)",
                "Phosphorus": f"{row['Phosphorus Fertilizer']} ({row['Fertilizer Quantity(kg/acre)']} kg/acre)",
                "Potassium": f"{row['Potassium Fertilizer']} ({row['Fertilizer Quantity(kg/acre)']} kg/acre)",
                "Zinc": f"{row['Zinc Fertilizer']} ({row['Fertilizer Quantity(kg/acre)']} kg/acre)"
            }

        return fertilizer_data
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return {}

fertilizer_data = load_fertilizer_data()

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 4)

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        return model, device

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

model, device = load_model()

# -------------------------------
# IMAGE TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload Rice Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        if model is None:
            st.error("Model not loaded. Please check deployment.")
        else:
            # Predict
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                label = class_mappings[predicted.item()]

            # Show prediction
            st.success(f"Prediction: {label}")

            # Show fertilizer
            st.subheader("🌱 Fertilizer Recommendations")

            if label in fertilizer_data:
                for key, value in fertilizer_data[label].items():
                    st.write(f"✅ {key}: {value}")
            else:
                st.warning("No recommendations found")

    except Exception as e:
        st.error(f"Error processing image: {e}")
``
