import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import platform

# 1. Configuración de la página
st.set_page_config(
    page_title="Control de Acceso - Juana",
    page_icon="🔒",
    layout="centered"
)

# Estilo CSS para ajustar márgenes
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Carga del modelo y etiquetas
@st.cache_resource
def load_keras_model():
    return load_model('keras_model.h5')

@st.cache_data
def load_labels():
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip().split(' ', 1)[-1] for line in f.readlines()]
    return class_names

try:
    model = load_keras_model()
    class_names = load_labels()
except Exception as e:
    st.error(f"Error al cargar archivos del modelo (`keras_model.h5` o `labels.txt`): {e}")

# 3. Barra Lateral (Sidebar)
with st.sidebar:
    st.image('OIG5.jpg', use_column_width=True)
    st.title("Sistema de Reconocimiento")
    st.info("Modelo de visión artificial entrenado en **Teachable Machine** para el control de acceso.")
    
    with st.expander("Información del sistema"):
        st.write(f"**Versión de Python:** {platform.python_version()}")
        st.write(f"**Clases detectables:** {', '.join(class_names) if 'class_names' in locals() else 'No cargadas'}")

# 4. Contenido Principal
st.title("🔒 Control de Acceso Facial")
st.write("Por favor, ubícate frente a la cámara y toma una foto para iniciar sesión.")

img_file_buffer = st.camera_input("Captura de rostro")

if img_file_buffer is not None:
    # Procesamiento de la imagen
    img = Image.open(img_file_buffer).convert('RGB')
    
    # Redimensionar al tamaño que requiere Keras (224, 224)
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    
    # Convertir a arreglo NumPy y normalizar
    img_array = np.asarray(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1.0
    
    # Arreglo final con dimensiones (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predicción
    with st.spinner("Procesando imagen..."):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_detected = class_names[index]
        confidence_score = prediction[0][index]

    st.markdown("---")
    
    # Verificar la detección según la etiqueta del modelo
    # Cambia 'Juana' si en tu labels.txt el nombre exacto está escrito diferente
    ES_JUANA = "Juana" in class_detected and confidence_score > 0.70

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(img, caption="Rostro capturado", width=180)

    with col2:
        if ES_JUANA:
            st.success("### ¡Juana detectada!")
            st.markdown("**Estado:** Bienvenida a tu sesión.")
            st.metric(label="Confianza de detección", value=f"{confidence_score * 100:.1f}%")
            st.balloons()
        else:
            st.error("### Juana no detectada")
            st.markdown("**Estado:** Inicio de sesión declinado.")
            st.metric(label="Confianza de detección", value=f"{confidence_score * 100:.1f}%")
