import os
import io
import numpy as np
from keras.models import load_model
from keras.utils import normalize
from PIL import Image
from helpers.feature_prep import load_image, build_feature
from helpers.image_processing import blend_image_arr

def preparation(model, imagepath):

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
    blended_image = blend_image_arr(col_image, output_image, col_image.shape[:2]) # X dan y tanpa 3 channle RGB

    # sebelum file nya kita return
    output_io = io.BytesIO()
    blended_image.save(output_io, format="JPEG")
    output_io.seek(0)

    return output_io, areas_label, area_lung, label_location[0], True