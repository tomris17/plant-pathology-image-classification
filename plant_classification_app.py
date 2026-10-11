import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Plant Pathology Classifier",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_model():
    # Eğitilen modelin dizinde bulunduğunu varsayıyoruz
    model = tf.keras.models.load_model("best_plant_pathology_model.h5", compile=False)
    return model

model = load_model()

# Yarışmadaki temel sınıflar (örnek olarak yaygın etiketler)
classes = [
    'healthy', 'scab', 'frog_eye_leaf_spot', 'complex', 
    'rust', 'powdery_mildew', 'scab frog_eye_leaf_spot complex', 'scab frog_eye_leaf_spot'
]

st.title("🌿 Plant Pathology - Leaf Disease Classification")
st.write("Bu uygulama, bitki yaprağı fotoğraflarını analiz ederek olası hastalıkları (EfficientNet tabanlı) tespit eder.")

uploaded_file = st.file_uploader("Bir Bitki Yaprağı Fotoğrafı Yükleyin (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Yaprak Fotoğrafı", use_column_width=True)
    
    if st.button("Hastalığı Teşhis Et"):
        # Görüntüyü modele uygun boyuta getirme (128x128)
        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Tahmin
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0])) * 100
        
        st.markdown("---")
        st.subheader("📊 Teşhis Sonucu:")
        
        # Sınıf eşleşme kontrolü
        if pred_idx < len(classes):
            detected_label = classes[pred_idx]
        else:
            detected_label = f"Sınıf ID: {pred_idx}"
            
        if "healthy" in detected_label.lower():
            st.success(f"🟢 **SAĞLIKLI BİTKİ** (Teşhis: {detected_label} - Güven: %{confidence:.2f})")
        else:
            st.error(f"🔴 **HASTALIK TESPİT EDİLDİ** (Teşhis: {detected_label} - Güven: %{confidence:.2f})")