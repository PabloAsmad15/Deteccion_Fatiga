import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

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
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            color = (0, 255, 0)
            rostro = img[y:y+h, x:x+w]
            if rostro.shape[0] < 128 or rostro.shape[1] < 128:
                continue

            # opcional: evaluar el rostro completo, no necesario para los ojos
            #rostro_redim = cv2.resize(rostro, (128, 128)) / 255.0
            #rostro_redim = rostro_redim.reshape(1, 128, 128, 3)
            #pred_rostro = modelo_rostro.predict(rostro_redim, verbose=0)

            ojos_cerrados = 0

            # Cortar dos regiones aproximadas para los ojos (simplificación)
            ojo_izq = img[y+int(h*0.2):y+int(h*0.5), x:x+int(w/2)]
            ojo_der = img[y+int(h*0.2):y+int(h*0.5), x+int(w/2):x+w]

            for ojo in [ojo_izq, ojo_der]:
                if ojo.size == 0:
                    continue
                ojo_gray = cv2.cvtColor(ojo, cv2.COLOR_BGR2GRAY)
                ojo_resized = cv2.resize(ojo_gray, (64, 64)) / 255.0
                ojo_resized = ojo_resized.reshape(1, 64, 64, 1)

                pred_ojo = modelo_ojos.predict(ojo_resized, verbose=0)
                clase_ojo = np.argmax(pred_ojo)

                if clase_ojo == 0:  # Cerrado
                    ojos_cerrados += 1

            # Si ambos ojos cerrados
            if ojos_cerrados == 2:
                if self.cerrado_inicio is None:
                    self.cerrado_inicio = time.time()
                elif time.time() - self.cerrado_inicio >= self.TIEMPO_UMBRAL:
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
