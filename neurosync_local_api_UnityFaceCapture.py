
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

from flask import request, jsonify
import numpy as np
import torch
import flask
import socket
import time


from utils.generate_face_shapes import generate_facial_data_from_bytes
from utils.model.model import load_model
from utils.config import config

from utils.FaceSample_pb2 import FaceSample

app = flask.Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Activated device:", device)

model_path = 'utils/model/model.pth'
blendshape_model = load_model(model_path, config, device)


# ───────────────────────── UDP 설정 ────────────────────────────
UNITY_IP   = "127.0.0.1"      # 필요 시 Unity PC IP
UNITY_PORT = 6000             # Connections 창에서 지정한 포트
sock       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_to_unity(coeffs):
    """
    coeffs: list[float] length 52, values 0-1
            OR list[list[float]] length (T,52)
    """
    # ── ① 2차원인 경우 첫 프레임만 선택 ──────────────
    if len(coeffs) and isinstance(coeffs[0], (list, np.ndarray)):
        coeffs = coeffs[-1]                 # 가장 마지막 프레임

    # ── ② FaceSample 직렬화 ─────────────────────
    sample = FaceSample()
    sample.version    = 1
    sample.timestamp  = time.time()
    sample.hasTracked = True
    sample.blendShapeWeights.extend(float(x) for x in coeffs)
    payload = sample.SerializeToString()

    print(f"[SEND] {len(payload)} bytes → {UNITY_IP}:{UNITY_PORT}")
    sock.sendto(payload, (UNITY_IP, UNITY_PORT))
    



@app.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    audio_bytes = request.data
    coeffs = generate_facial_data_from_bytes(
        audio_bytes, blendshape_model, device, config
    )
    coeffs_list = coeffs.tolist() if isinstance(coeffs, np.ndarray) else coeffs

    send_to_unity(coeffs_list)        

    return jsonify({'blendshapes': coeffs_list})





if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
