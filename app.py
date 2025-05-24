import os
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import mariadb
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import joblib
import pandas as pd
import json
from helpers.image_segmentation import preparation
import io

# init app dll
app = Flask(__name__)

CORS(app, origins=["http://localhost:5173"])

load_dotenv()
# app.config['MYSQL_DATABASE_USER'] = os.getenv("DB_USER", "root")
# app.config['MYSQL_DATABASE_PASSWORD'] = os.getenv("DB_PASS", "")
# app.config['MYSQL_DATABASE_DB'] = os.getenv("DB_TABLES", "tbc")
# app.config['MYSQL_DATABASE_HOST'] = os.getenv("DB_HOST", "localhost")
# app.config['MYSQL_DATABASE_PORT'] = int(os.getenv("DB_PORT", 3306))


# db = MySQL()
# db.init_app(app)

conn = mariadb.connect(
      host=os.getenv("DB_HOST", "localhost"),
      port=int(os.getenv("DB_PORT", 330)),
      ssl_verify_cert=True,
      user=os.getenv("DB_USER", "root"),
      password=os.getenv("DB_PASS", ""),
      database=os.getenv("DB_TABLES", "tbc")
)
cursor = conn.cursor()

# Dapatkan informasi semua model
@app.route('/models', methods=['GET'])
def getallmodels():
    cursor.execute("SELECT * FROM tbcmodels")
    data = cursor.fetchall()

    res = []
    for rows in data:
        res.append({"filename": rows[1], "description": rows[2]})

    return jsonify(res)

@app.route('/models/<id>', methods=['GET'])
def getmodels(id):
    cursor.execute("SELECT * FROM tbcmodels WHERE id = %(modelid)s", {'modelid': id})
    data = cursor.fetchone()

    if(data):
        res = {"filename": data[1], "description": data[2]}
    else:
        res = "model not found"

    return jsonify(res)

'''
old routes

@app.route('/prepare', methods=['POST'])
def preparation():

    f = request.files.get('file')
    if not f:
        return jsonify({'result': 'Error', 'message': 'Input not found'})

    imagepath = os.path.join('usercontent', 'prep' + secure_filename(f.filename))
    f.save(imagepath)

    # model lung segmentation
    modellung_path = os.path.join("legacymodels", "Preparation1.hdf5")
    try:
        model_lung = load_model(modellung_path)
    except:
        return jsonify({'result': 'Error', 'message': 'Error loading model'})
    
    # model label segmentation
    modellabel_path = os.path.join("legacymodels", "Preparation2.hdf5")
    try:
        model_label = load_model(modellabel_path)
    except:
        return jsonify({'result': 'Error', 'message': 'Error loading model'})

    SIZE = 256

    # Colormap awal: mapping dari (R, G, B) ke indeks kelas
    colormap = {
        (0, 0, 0): 0,         # Background (hitam)
        (9, 2, 221): 1,       # Blue (Kelas 1)
        (230, 150, 170): 2,   # Pengganti Putih (Kelas 2)
        (179, 124, 94): 3,    # Brown (Kelas 3)
        (250, 237, 15): 4,    # Yellow (Kelas 4)
        (154, 9, 236): 5,     # Purple (Kelas 5)
        (64, 128, 128): 6,    # Darktail (Kelas 6)
    }
    # Balik mapping sehingga menjadi: {kelas: (R, G, B)}
    colors = {v: k for k, v in colormap.items()}

    location_labels = []

    # Mulai processing input
    col_image, gray_image = load_image(imagepath)
    gray_image = Image.fromarray(gray_image)
    gray_image = np.array(gray_image.resize((SIZE,SIZE)))

    # Normalize dan penyesuaian input model
    test_img = normalize(gray_image, axis=1)
    # print("Shape image_data:", test_img.shape)  # Contoh: (N, SIZE, SIZE, 1)
    test_input = np.expand_dims(test_img,0)
    test_input = np.expand_dims(test_input,3)

    # Call helper function untuk membangun fitur
    output_image, areas_label, area_lung, label_location = build_feature(test_input, model_lung, model_label, colors, location_labels)

    # blend image dengan input
    input_image_PIL = Image.fromarray(col_image)
    output_image_PIL = Image.fromarray(output_image).resize(input_image_PIL.size)
    blended_image = Image.blend(input_image_PIL, output_image_PIL, 0.5)

    # sebelum file nya kita return
    output_io = io.BytesIO()
    blended_image.save(output_io, format="JPEG")
    output_io.seek(0)

    # hapus supaya tidak menumpuk di server 
    os.remove(imagepath)

    #jadikan satu response
    res = make_response(send_file(output_io, mimetype='image/jpeg'))
    res.headers['areas-label'] = areas_label
    res.headers['area-lung'] = area_lung
    res.headers['label-location'] = label_location
    return res

@app.route('/predict', methods=['POST'])
def prediction():
    model_id = json.loads(request.form['model_id'])
    if not model_id:
        return jsonify({'result': 'Error', 'message': 'Model not selected'})
    ""
    model_fp = getmodels(model_id).json['filename']
    modelpath = os.path.join('legacymodels', model_fp)

    try:
        with open(modelpath, 'rb') as f:
            model = joblib.load(f)
    except:
        return jsonify({'result': 'Error', 'message': 'Error loading model'})

    f = request.files['file']
    if not f:
        return jsonify({'result': 'Error', 'message': 'Input not found'})

    imagepath = os.path.join('usercontent', secure_filename(f.filename))
    f.save(imagepath)

    input_FS = FileStorage(
        stream=open(imagepath, "rb"),
        filename=os.path.split(imagepath)[-1],
        content_type=f.mimetype
    )

    files = {'file': input_FS}
    with app.test_client() as client:
        prep_res = client.post('/prepare', data=files, content_type='multipart/form-data')

    # convert balik data hasil segmentasi yang terconvert ke string di header
    prep_headers = dict(prep_res.headers)
    areas_label = literal_eval(prep_headers["areas-label"])
    area_lung = literal_eval(prep_headers["area-lung"])
    label_location = literal_eval(prep_headers["label-location"])[0] #berbeda dengan areas-label yang berupa dict, pada label-location
    # dict nya berada di dalam suatu single-element array

    labels = [
        'background', 
        'rasio_luas_blue', 
        'rasio_luas_pengganti_putih', 
        'rasio_luas_brown',
        'rasio_luas_yellow',
        'rasio_luas_purple',
        'rasio_luas_darktail', 
        'posisi_blue', 
        'posisi_pengganti_putih', 
        'posisi_brown',
        'posisi_yellow',
        'posisi_purple',
        'posisi_darktail'
    ]

    ratios = {}
    for i in range(1, 7):
        ratios[labels[i]] = areas_label[i] / area_lung
    ratios_res = ratios

    if "model2.pkl" in model_fp:
        for i in range(8, 13):
            ratios[labels[i]] = label_location[i-6]
    
    ratios = pd.DataFrame([ratios])

    result = model.predict_proba(ratios)

    # hapus supaya tidak menumpuk di server 
    os.remove(imagepath)

    #jadikan satu response
    res = make_response(send_file(io.BytesIO(prep_res.data), mimetype='image/jpeg'))
    res.headers['areas-label'] = ratios_res
    res.headers['prediction-result'] = result[0][1]
    return res

@app.route('/prepare', methods=['POST'])
def preparation():
    try:
        model_id = json.loads(request.form.get('model_id'))
    except:
        return jsonify({'result': 'Error', 'message': 'Model not selected'})
    # id 1-4 = model single unet
    # id 5-8 = model double unet
    # dapat diganti jika databasenya berubah

    f = request.files.get('file')
    if not f:
        return jsonify({'result': 'Error', 'message': 'Input not found'})

    imagepath = os.path.join('usercontent', 'prep' + secure_filename(f.filename))
    f.save(imagepath)

    # model lung segmentation
    model_path = os.path.join("models", "16_100_double_UNET.hdf5" if model_id > 4 else "32_100_single_unet_030525.hdf5")
    try:
        model = load_model(model_path)
    except:
        return jsonify({'result': 'Error', 'message': 'Error loading model'})

    SIZE = 256

    # Colormap awal: mapping dari (R, G, B) ke indeks kelas
    colormap = {
        (0, 0, 0): 0,         # Background (hitam)
        (9, 2, 221): 1,       # Blue (Kelas 1)
        (230, 150, 170): 2,   # Pengganti Putih (Kelas 2)
        (179, 124, 94): 3,    # Brown (Kelas 3)
        (250, 237, 15): 4,    # Yellow (Kelas 4)
        (154, 9, 236): 5,     # Purple (Kelas 5)
        (64, 128, 128): 6,    # Darktail (Kelas 6)
    }
    # Balik mapping sehingga menjadi: {kelas: (R, G, B)}
    colors = {v: k for k, v in colormap.items()}

    location_labels = []

    # Mulai processing input
    col_image, gray_image = load_image(imagepath)
    gray_image = Image.fromarray(gray_image)
    gray_image = np.array(gray_image.resize((SIZE,SIZE)))

    # Normalize dan penyesuaian input model
    test_img = normalize(gray_image, axis=1)
    # print("Shape image_data:", test_img.shape)  # Contoh: (N, SIZE, SIZE, 1)
    test_input = np.expand_dims(test_img,0)
    test_input = np.expand_dims(test_input,3)

    # Call helper function untuk membangun fitur
    output_image, areas_label, area_lung, label_location = build_feature(test_input, model, colors, location_labels)

    # blend image dengan input
    blended_image = blend_image_arr(col_image, output_image, col_image.shape)

    # sebelum file nya kita return
    output_io = io.BytesIO()
    blended_image.save(output_io, format="JPEG")
    output_io.seek(0)

    # hapus supaya tidak menumpuk di server 
    os.remove(imagepath)

    #jadikan satu response
    res = make_response(send_file(output_io, mimetype='image/jpeg'))
    res.headers['areas-label'] = areas_label
    res.headers['area-lung'] = area_lung
    res.headers['label-location'] = label_location
    return res
'''

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

    imagepath = os.path.join('usercontent', secure_filename(f.filename))
    f.save(imagepath)

    '''
    input_FS = FileStorage(
        stream=open(imagepath, "rb"),
        filename=os.path.split(imagepath)[-1],
        content_type=f.mimetype
    )

    files = {'file': input_FS, 'model_id': model_id}
    with app.test_client() as client:
        prep_res = client.post('/prepare', data=files, content_type='multipart/form-data')

    # convert balik data hasil segmentasi yang terconvert ke string di header
    try:
        prep_headers = dict(prep_res.headers)
        areas_label = literal_eval(prep_headers["areas-label"])
        area_lung = literal_eval(prep_headers["area-lung"])
        label_location = literal_eval(prep_headers["label-location"])[0] 
        # berbeda dengan areas-label yang berupa dict, pada label-location
        # dict nya berada di dalam suatu single-element array
    except:
        return jsonify(prep_res)
    '''
    seg_image, areas_label, area_lung, label_location, success = preparation(model_id, imagepath)
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

    result = model.predict_proba(ratios)

    # hapus supaya tidak menumpuk di server 
    os.remove(imagepath)

    #jadikan satu response
    res = make_response(send_file(seg_image, mimetype='image/jpeg'))
    res.headers['areas-label'] = ratios_res
    res.headers['prediction-result'] = result[0]
    return res

@app.route("/predictB64", methods=["POST"])
def pred64():
    data = json.loads(request.data)
    if not data["model_id"] or not data["file"]:
        return jsonify({'result': 'Error', 'message': 'Input not found'})
    
    import base64
    image_bytes = base64.b64decode(data["file"])
    image_stream = io.BytesIO(image_bytes)

    import imghdr
    image_format = imghdr.what(None, h=image_bytes)

    # Create FileStorage object
    from datetime import datetime

    input_FS = FileStorage(
        stream=image_stream,
        filename=f"{str(datetime.now())}.{image_format}" if image_format else f"{str(datetime.now())}.jpeg",
        content_type=f"image/{image_format}" if image_format else "application/octet-stream"
    )

    files = {'file': input_FS, 'model_id':data["model_id"]}

    with app.test_client() as client:
        res = client.post('/predict', data=files, content_type='multipart/form-data')
    try:
        res_headers = dict(res.headers)
        res_B64 = base64.b64encode(res.data).decode("utf-8")
        return jsonify({"file": res_B64, "areas_label": res_headers["areas-label"], "pred_result": res_headers["prediction-result"]})
    except:
        return res.data




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('APP_PORT', 5000))
    