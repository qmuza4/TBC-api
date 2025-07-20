import os
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import joblib
import pandas as pd
import json
from helpers.image_segmentation import preparation
from helpers.supabase_storage import uploadToStorage, isAdmin, createUser
import io
import sqlite3
from keras.models import load_model

# init app dll

app = Flask(__name__)

CORS(app, origins=["http://localhost:5173", "https://tbscreen.ai"])

load_dotenv()

# create koneksi ke database
conn = sqlite3.connect(os.path.join('db', 'database.db'), check_same_thread=False)
cursor = conn.cursor()

# load model segmentasi
DU_model = load_model(os.path.join("models", "16_100_double_UNET.hdf5"))
SU_model = load_model(os.path.join("models", "32_100_single_unet_030525.hdf5"))

# Dapatkan informasi semua model
@app.route('/', methods=['GET'])
def index():
    return jsonify("Hello, world!")

@app.route('/models', methods=['GET'])
def getallmodels():
    cursor.execute("SELECT * FROM tbcmodels")
    data = cursor.fetchall()

    res = []
    for rows in data:
        res.append({"id": rows[0], "filename": rows[1], "description": rows[2]})

    return jsonify(res)

# Dapatkan informasi 1 model
@app.route('/models/<id>', methods=['GET'])
def getmodels(id):
    cursor.execute("SELECT * FROM tbcmodels WHERE id = ?", (id, ))
    data = cursor.fetchone()

    if(data):
        res = {"filename": data[1], "description": data[2]}
    else:
        res = "model not found"

    return jsonify(res)

@app.route('/createuser', methods=['POST'])
def register():
    data = json.loads(request.data)
    if not data.get('admin_uuid'):
        return jsonify({'result': 'Error', 'message': 'Unauthorized'})
    if not isAdmin(data.get('admin_uuid')):
        return jsonify({'result': 'Error', 'message': 'Unauthorized'})
    if not data.get('email') or not data.get('password'):
        return jsonify({'result': 'Error', 'message': 'Bad Request'})
    
    res= createUser(data.get('email'), data.get('password'), data.get('role', 'user'))

    if res.get("error"):
        return jsonify({'result': 'Error', 'message': res['error']['message']})
    return jsonify({'user': res['user']})

# Route prediksi, input berupa form-data dengan file: file x-ray dan model_id: integer, id dari model
@app.route('/predict', methods=['POST'])
def prediction():
    try:
        model_id = json.loads(request.form['model_id'])
    except:
        return jsonify({'result': 'Error', 'message': 'Model not selected'})
    
    model_fp = getmodels(model_id).json['filename']
    modelpath = os.path.join('models', model_fp)

    try:
        with open(modelpath, 'rb') as f:
            model = joblib.load(f)
    except:
        return jsonify({'result': 'Error', 'message': 'Error loading model'})

    f = request.files['file']
    if not f:
        return jsonify({'result': 'Error', 'message': 'Input not found'})

    imagepath = os.path.join('usercontent', f.filename)
    f.save(imagepath)

    # id 1-4 merupakan model single UNET, sedangkan 5-8 merupakan model double UNET. dapat berubah tergantung databasenya.
    input_model = DU_model if model_id > 4 else SU_model
    seg_image, areas_label, area_lung, label_location, success = preparation(input_model, imagepath)
    if not success:
        return jsonify({'result': 'Error', 'message': 'Error segmenting image'})
    
    labels = [
        'background', 
        'luas blue', 
        'luas pengganti putih', 
        'luas brown',
        'luas yellow',
        'luas purple',
        'luas darktail', 
        'posisi blue', 
        'posisi pengganti putih', 
        'posisi brown',
        'posisi yellow',
        'posisi purple',
        'posisi darktail'
    ]

    ratios = {}
    for i in range(1, 7):
        ratios[labels[i]] = areas_label[i] / area_lung
    
    ratios_res = ratios.copy()

    if "posisi" in model_fp:
        for i in range(7, 13):
            ratios[labels[i]] = label_location[i-6]
    
    ratios = pd.DataFrame([ratios])

    result = model.predict_proba(ratios)[0]

    # Upload ke image storage
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_url = uploadToStorage(seg_image, "ai-analysis-" + str(timestamp))

    # hapus supaya tidak menumpuk di server 
    if os.path.exists(imagepath):
        os.remove(imagepath)

    #jadikan satu response
    return jsonify({"file": img_url, "areas_label": ratios_res, "pred_result": result.tolist()})

# Route prediksi, input berupa JSON dengan file: citra X-ray (Base64-encoded string) dan model_id: integer, id dari model
@app.route("/predictB64", methods=["POST"])
def pred64():
    data = json.loads(request.data)
    if not data["model_id"] or not data["file"]:
        return jsonify({'result': 'Error', 'message': 'Input not found'})
    
    # Revert base64 string to bytes
    import base64
    image_bytes = base64.b64decode(data["file"])
    image_stream = io.BytesIO(image_bytes)

    import imghdr
    image_format = imghdr.what(None, h=image_bytes)

    # Create FileStorage object
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    input_FS = FileStorage(
        stream=image_stream,
        filename=f"{timestamp}.{image_format}" if image_format else f"{timestamp}.png",
        content_type=f"image/{image_format}" if image_format else "application/octet-stream"
    )

    files = {'file': input_FS, 'model_id':data["model_id"]}

    # Forward created file to original route
    with app.test_client() as client:
        res = client.post('/predict', data=files, content_type='multipart/form-data')
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('APP_PORT', 5000))
    