import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

st.set_page_config(page_title="Detector de Fatiga", layout="centered")
st.title("🧠 Detector de Fatiga en Conductores")

def cargar_modelos():
    modelo_ojos = load_model("modelo_ojos.keras")
    modelo_rostro = load_model("modelo_rostro.keras")
    detector = MTCNN()
    return modelo_ojos, modelo_rostro, detector

modelo_ojos, modelo_rostro, detector = cargar_modelos()

class FatigaDetector(VideoTransformerBase):
    def __init__(self):
        self.cerrado_inicio = None
        self.TIEMPO_UMBRAL = 5
        self.estado_fatiga = "Normal"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detecciones = detector.detect_faces(frame_rgb)

        for deteccion in detecciones:
            x, y, w, h = deteccion['box']
            x, y = max(0, x), max(0, y)
            color = (0, 255, 0)

            rostro = img[y:y+h, x:x+w]
            if rostro.shape[0] < 128 or rostro.shape[1] < 128:
                continue

            rostro_redim = cv2.resize(rostro, (128, 128)) / 255.0
            rostro_redim = rostro_redim.reshape(1, 128, 128, 3)

            pred_rostro = modelo_rostro.predict(rostro_redim, verbose=0)
            clase_rostro = np.argmax(pred_rostro)

            keypoints = deteccion['keypoints']
            ojos_cerrados = 0

            for nombre_ojo in ['left_eye', 'right_eye']:
                ex, ey = keypoints[nombre_ojo]
                ojo = img[ey-20:ey+20, ex-20:ex+20]
                if ojo.shape[0] != 40 or ojo.shape[1] != 40:
                    continue

                ojo_gray = cv2.cvtColor(ojo, cv2.COLOR_BGR2GRAY)
                ojo_redim = cv2.resize(ojo_gray, (64, 64)) / 255.0
                ojo_redim = ojo_redim.reshape(1, 64, 64, 1)

                pred_ojo = modelo_ojos.predict(ojo_redim, verbose=0)
                clase_ojo = np.argmax(pred_ojo)

                if clase_ojo == 0:
                    ojos_cerrados += 1
                    cv2.circle(img, (ex, ey), 10, (0, 0, 255), 2)
                else:
                    cv2.circle(img, (ex, ey), 10, (0, 255, 0), 2)

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

#  LLAMADA DENTRO DE CONTROL INTERACTIVO DE STREAMLIT
if st.checkbox("Iniciar detección en tiempo real"):
    webrtc_streamer(
        key="fatiga_stream",
        video_transformer_factory=FatigaDetector,
        media_stream_constraints={"video": True, "audio": False}
    )
