#!/usr/bin/env python3
# Run this on beacon
from flask import Flask, request, jsonify
import easyocr, base64, numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
reader = easyocr.Reader(['en'], gpu=True)

@app.route('/ocr', methods=['POST'])
def ocr():
    data   = request.json
    img    = Image.open(BytesIO(base64.b64decode(data['image'])))
    result = reader.readtext(np.array(img), detail=0, paragraph=True)
    return jsonify({"text": " ".join(result).strip()})

@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "ocr"})

app.run(host='0.0.0.0', port=5001)
