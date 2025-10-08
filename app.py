import streamlit as st
from PIL import Image
from model_loader import load_model, predict_image


# 1️ Page Config

st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector 🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2️ Sidebar Info

st.sidebar.title("Pneumonia Detection App")
st.sidebar.markdown("""
**How to use:**
1. Upload a Chest X-ray image (jpg, jpeg, png).
2. The model will predict whether the lungs are **Normal** or show **Pneumonia**.
""")




# 3️ Model Loading with Cache
@st.cache_resource
def get_model():
    model, device = load_model()  
    return model, device

model, device = get_model()


# 4️ Main App Interface
st.title("🩺 Chest X-Ray Pneumonia Detector")
st.write("Upload a chest X-ray image and the model will predict **Normal** or **Pneumonia**.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        with st.spinner("Analyzing X-ray..."):
            pred_class, confidence = predict_image(model, device, image)

        # Display result
        if pred_class == "Pneumonia":
            st.error(f" Pneumonia Detected!")
        else:
            st.success(f" Normal Lungs")

        # Show confidence
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
        st.progress(int(confidence*100))

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")



