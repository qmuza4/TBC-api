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
from helpers.supabase_storage import uploadToStorage, isAdmin, createUser, updateUser, deleteUser
import io
import sqlite3
from supabase import create_client
from keras.models import load_model

# init app dll

app = Flask(__name__)

CORS(app, origins=["http://localhost:5173", "https://tbscreen.ai"])

load_dotenv()

# create koneksi ke database sqlite lokal
conn = sqlite3.connect(os.path.join('db', 'database.db'), check_same_thread=False)
cursor = conn.cursor()

# load model segmentasi yang berukuran besar
DU_model = load_model(os.path.join("models", "16_100_double_UNET.hdf5"))
SU_model = load_model(os.path.join("models", "32_100_single_unet_030525.hdf5"))

# create client supabase
service_role_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# ***~~~___ Routes ___~~~***

# Section: Server-side auth. data user mayoritas dihandle di react app (frontend) dengan supabase db dan supabase auth, namun ada dungsi-fungsi spesifik
# pada supabase.auth.admin yang memerlukan service role key khusus (Do not expose in frontend client) sehingga harus dilakukan di sini.

# Buat user baru pada supabase.auth
@app.route('/createuser', methods=['POST'])
def registeruser():
    data = json.loads(request.data)
    if not data.get('admin_uuid'):
        return jsonify({'result': 'Error', 'message': 'Unauthorized'}), 401
    if not isAdmin(service_role_client, data.get('admin_uuid')):
        return jsonify({'result': 'Error', 'message': 'Admin-only feature'}), 403
    if not data.get('email') or not data.get('password'):
        return jsonify({'result': 'Error', 'message': 'Bad Request'}), 400
    
    res = createUser(service_role_client, data.get('email'), data.get('password'), data.get('role', 'user'))

    if res.get("error"):
        return jsonify({'result': 'Error', 'message': res['error']['message']}), 500
    return jsonify({'user': res['user']})

# Edit data user pada pada supabase.auth
@app.route('/updateuser/<user_uuid>', methods=['PATCH'])
def edituser(user_uuid):
    data = json.loads(request.data)
    if not data.get('admin_uuid'):
        return jsonify({'result': 'Error', 'message': 'Unauthorized'}), 401
    if not isAdmin(service_role_client, data.get('admin_uuid')):
        return jsonify({'result': 'Error', 'message': 'Admin-only feature'}), 403
    
    changable_fields = {"email", "password", "phone", "user_metadata"}
    updated_fields = {k: v for k, v in data.items() if k in changable_fields}
    
    res = updateUser(service_role_client, user_uuid, updated_fields)

    if res.get("error"):
        return jsonify({'result': 'Error', 'message': res['error']['message']}), 500
    return jsonify({'user': res['user']})

# Hapus data user pada pada supabase.auth
@app.route('/deleteuser/<user_uuid>', methods=['DELETE'])
def removeuser(user_uuid):
    data = json.loads(request.data)
    if not data.get('admin_uuid'):
        return jsonify({'result': 'Error', 'message': 'Unauthorized'}), 401
    if not isAdmin(service_role_client, data.get('admin_uuid')):
        return jsonify({'result': 'Error', 'message': 'Admin-only feature'}), 403
    
    res = deleteUser(service_role_client, user_uuid)

    if res.get("error"):
        return jsonify({'result': 'Error', 'message': res['error']['message']}), 500
    return jsonify({'user': 'User deleted successfully'})

# Section: data model. karena database model tidak mengalami banyak perubahan (tergantung update pihak developer, dan kemungkinan akan konstan most of the time)
# dan jumlahnya kecil maka dilakukan pada sqlite lokal server, terpisah dari database user dll pada supabase

@app.route('/', methods=['GET'])
def index():
    return jsonify("Hello, world!")

# Dapatkan informasi semua model
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

# Section: Machine learning models. tujuan utama dari penggunaan server backend dengan python. route-route dibawah berhubungan dengan model yang dibuat oleh team AI,
# meng-load model dan melakukan segmentasi serta klasifikasi dari file citra yang dimasukkan

# Route prediksi, input berupa form-data dengan file: file x-ray dan model_id: integer, id dari model
@app.route('/predict', methods=['POST'])
def prediction():
    try:
        model_id = json.loads(request.form['model_id'])
    except:
        return jsonify({'result': 'Error', 'message': 'Model not selected'}), 400
    
    model_fp = getmodels(model_id).json['filename']
    modelpath = os.path.join('models', model_fp)

    try:
        with open(modelpath, 'rb') as f:
            model = joblib.load(f)
    except:
        return jsonify({'result': 'Error', 'message': 'Error loading model'}), 500

    f = request.files['file']
    if not f:
        return jsonify({'result': 'Error', 'message': 'Input not found'}), 400

    imagepath = os.path.join('usercontent', f.filename)
    f.save(imagepath)

    # id 1-4 merupakan model single UNET, sedangkan 5-8 merupakan model double UNET. dapat berubah tergantung databasenya.
    input_model = DU_model if model_id > 4 else SU_model
    seg_image, areas_label, area_lung, label_location, success = preparation(input_model, imagepath)
    if not success:
        return jsonify({'result': 'Error', 'message': 'Error segmenting image'}), 500
    
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
    img_url = uploadToStorage(service_role_client, os.getenv("SUPABASE_URL"), seg_image, "ai-analysis-" + str(timestamp))
    if img_url is None:
        return jsonify({'result': 'Error', 'message': 'Error uploading to storage bucket'}), 500

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
        return jsonify({'result': 'Error', 'message': 'Input not found'}), 400
    
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
    