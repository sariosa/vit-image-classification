pip install streamlit

%%writefile app.py
import streamlit as st
from PIL import Image
import requests
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

@st.cache_resource
def load_model_and_processor():
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    return processor, model

processor, model = load_model_and_processor()

st.title("Image Classification with ViT")

uploaded_file = st.file_uploader("Give me an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded Image', use_column_width=True)

if st.button("Classify"):
    with st.spinner('Processing...'):
        try:
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]

            st.success(f"Predicted class: {predicted_label}")
        except Exception as e:
            st.error(f"Error processing image: {e}")

!streamlit run app.py --server.port 8511 & npx localtunnel --port 8511
