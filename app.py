import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
# from mtcnn import MTCNN  # Opcional, si quieres mantenerlo

st.set_page_config(page_title="Detector de Fatiga", layout="centered")
st.title("🧠 Detector de Fatiga en Conductores")

@st.cache_resource
def cargar_modelos():
    modelo_ojos = load_model("modelo_ojos.keras")
    modelo_rostro = load_model("modelo_rostro.keras")
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return modelo_ojos, modelo_rostro, detector

modelo_ojos, modelo_rostro, detector = cargar_modelos()

class FatigaDetector(VideoTransformerBase):
    def __init__(self):
        self.cerrado_inicio = None
        self.TIEMPO_UMBRAL = 5
        self.estado_fatiga = "Normal"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))  # Reducir para mejorar rendimiento
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect_faces = detector.detect_faces(frame_rgb)  # Si usas MTCNN
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            color = (0, 255, 0)
            rostro = img[y:y+h, x:x+w]
            if rostro.shape[0] < 128 or rostro.shape[1] < 128:
                continue

            rostro_redim = cv2.resize(rostro, (128, 128)) / 255.0
            rostro_redim = rostro_redim.reshape(1, 128, 128, 3)

            pred_rostro = modelo_rostro.predict(rostro_redim, verbose=0)
            clase_rostro = np.argmax(pred_rostro)

            ojos_cerrados = 0
            # Por simplicidad, marcamos 2 ojos cerrados si la predicción de rostro detecta fatiga
            if clase_rostro == 0:  # o tu clase de "cerrado"
                ojos_cerrados = 2

            if ojos_cerrados == 2:
                if self.cerrado_inicio is None:
                    self.cerrado_inicio = time.time()
                elif time.time() - self.cerrado_inicio > self.TIEMPO_UMBRAL:
                    self.estado_fatiga = "FATIGA DETECTADA"
                    color = (0, 0, 255)
            else:
                self.cerrado_inicio = None
                self.estado_fatiga = "Normal"
                color = (0, 255, 0)

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, self.estado_fatiga, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(img, f"Estado: {self.estado_fatiga}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        return img

if st.checkbox("Iniciar detección en tiempo real"):
    webrtc_streamer(
        key="fatiga_stream",
        video_transformer_factory=FatigaDetector,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
