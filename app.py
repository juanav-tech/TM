import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# 1. Configuración de la página
st.set_page_config(
    page_title="Control de Acceso - Juana",
    page_icon="🔒",
    layout="centered"
)

# Carga del modelo original
model = load_model('keras_model.h5')

# 2. Barra Lateral (Sidebar)
with st.sidebar:
    try:
        image = Image.open('OIG5.jpg')
        st.image(image, width=350)
    except:
        pass
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar")
    
    with st.expander("Información del sistema"):
        st.write("Versión de Python:", platform.python_version())

# 3. Interfaz Principal
st.title("🔒 Reconocimiento de Imágenes")
st.write("Por favor, ubícate frente a la cámara y toma una foto para iniciar sesión.")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # --- TU CÓDIGO DE PROCESAMIENTO INICIAL ---
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalización inicial (127.0)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Predicción inicial
    prediction = model.predict(data)
    print(prediction)

    st.markdown("---")

    # --- INDICAR ÍNDICE DE JUANA ---
    # Cambia el 0 por la posición exacta donde tu modelo detecta a Juana (0, 1 o 2)
    INDICE_JUANA = 0 
    
    probabilidad_juana = prediction[0][INDICE_JUANA]

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(img, caption="Foto capturada", width=180)

    with col2:
        # Condición inicial (> 0.5)
        if probabilidad_juana > 0.5:
            st.success("### Juana detectada, bienvenida a tu sesión")
            st.metric(label="Probabilidad", value=f"{probabilidad_juana:.2f}")
            st.balloons()
        else:
            st.error("### Juana no detectada, inicio de sesión declinado")
            st.metric(label="Probabilidad", value=f"{probabilidad_juana:.2f}")
            
    # Muestra los valores de predicción crudos para inspección
    with st.expander("Ver vector de predicción (print)"):
        st.write(prediction)
