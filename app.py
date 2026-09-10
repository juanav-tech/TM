import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# 1. Configuración de la página
st.set_page_config(
    page_title="Control de Acceso - Juana",
    page_icon="🔒",
    layout="centered"
)

# 2. Carga de modelo y etiquetas con caché
@st.cache_resource
def load_keras_model():
    # Carga el modelo exportado de Teachable Machine
    return load_model('keras_model.h5', compile=False)

@st.cache_data
def load_labels():
    with open('labels.txt', 'r', encoding='utf-8') as f:
        # Limpia las líneas del archivo labels.txt (ej: "0 Juana" -> "Juana")
        class_names = [line.strip().split(' ', 1)[-1] for line in f.readlines()]
    return class_names

try:
    model = load_keras_model()
    class_names = load_labels()
except Exception as e:
    st.error(f"Error cargando los archivos `keras_model.h5` o `labels.txt`: {e}")

# 3. Barra Lateral
with st.sidebar:
    try:
        st.image('OIG5.jpg', use_container_width=True)
    except:
        pass
    st.title("Sistema de Reconocimiento")
    st.info("Modelo entrenado en **Teachable Machine**")
    
    with st.expander("Detalles del sistema"):
        st.write(f"**Python:** {platform.python_version()}")
        if 'class_names' in locals():
            st.write("**Clases cargadas:**")
            for idx, name in enumerate(class_names):
                st.write(f"- Clase {idx}: `{name}`")

# 4. Interfaz Principal
st.title("🔒 Control de Acceso Facial")
st.write("Ubícate frente a la cámara y toma una foto para validar tu ingreso.")

img_file_buffer = st.camera_input("Captura de rostro")

if img_file_buffer is not None:
    # --- PROCESAMIENTO EXACTO DE TEACHABLE MACHINE ---
    
    # 1. Cargar imagen como RGB
    img = Image.open(img_file_buffer).convert('RGB')
    
    # 2. Redimensionado simple a 224x224 (igual que Teachable Machine)
    img_resized = img.resize((224, 224))
    
    # 3. Convertir a arreglo NumPy
    img_array = np.asarray(img_resized, dtype=np.float32)
    
    # 4. Normalización estándar de Teachable Machine: (x / 127.5) - 1
    normalized_image_array = (img_array / 127.5) - 1.0
    
    # 5. Formato de entrada para Keras (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # --- PREDICCIÓN ---
    with st.spinner("Procesando imagen..."):
        prediction = model.predict(data)
        
        # Obtener el índice con mayor probabilidad
        index_detected = np.argmax(prediction[0])
        label_detected = class_names[index_detected]
        confidence = float(prediction[0][index_detected])

    st.markdown("---")
    
    # Comprobar si la clase detectada con mayor probabilidad es "Juana"
    # (Asegúrate de que 'Juana' coincida exactamente con la etiqueta en labels.txt)
    ES_JUANA = "juana" in label_detected.lower()

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(img_resized, caption="Imagen procesada", width=180)

    with col2:
        if ES_JUANA:
            st.success("### ¡Juana detectada!")
            st.markdown("**Estado:** Bienvenida a tu sesión.")
            st.metric(label="Confianza", value=f"{confidence * 100:.1f}%")
            st.balloons()
        else:
            st.error("### Juana no detectada")
            st.markdown("**Estado:** Inicio de sesión declinado.")
            st.metric(label="Clase detectada", value=f"{label_detected} ({confidence * 100:.1f}%)")
            
    # Depuración rápida: desactiva o quita esto si no lo necesitas
    with st.expander("Ver vector de probabilidades en vivo"):
        for name, prob in zip(class_names, prediction[0]):
            st.write(f"**{name}:** {prob * 100:.2f}%")
