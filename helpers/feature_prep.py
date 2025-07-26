import cv2
import numpy as np

def load_image(image_path):
  image = cv2.imread(image_path)
  if image is None:
    print('Error: Could not load image.')
    return None, None
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return image, gray

def label_to_rgb(label, colors):
    """
    Mengonversi array label 2D menjadi citra RGB.
    """
    height, width = label.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for cls, color in colors.items():
        rgb[label == cls] = color
    return rgb

def calculate_areas(mask, colors):
    """
    Menghitung luas area (jumlah piksel) untuk masing-masing kelas pada mask.
    """
    areas = {}
    for cls in sorted(colors.keys()):
        areas[cls] = int(np.count_nonzero(mask == cls))
    return areas


def find_postition(lung, mask, colors):
  # ambil member dari contours
  contours, _ = cv2.findContours(lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if len(contours) == 0:
    print("Warning: No contours found in the lung mask.")
    return []  # or some other appropriate default value

  all_points = np.vstack(contours)

  # Dapatkan bounding box (x, y, w, h)
  x, y, w, h = cv2.boundingRect(all_points)

  # batas posisi
  top_line = y + h // 3         # Pembatas horizontal atas
  mid_line = y + (2 * h) // 3   # Pembatas horizontal bawah
  center_line = x + w // 2      # Pembatas vertical tengah

  # Area Koordinat mask
  locations_maps = {
      0: mask[0:top_line,         center_line:w],   # T(U,R)
      1: mask[0:top_line,         0:center_line],   # T(U,L)
      2: mask[top_line:mid_line,  0:center_line],   # T(M,L)
      3: mask[mid_line:h,         0:center_line],   # T(L,L)
      4: mask[center_line:w,      mid_line:h],      # T(L,R)
      5: mask[top_line:mid_line,  center_line:w],   # T(M,R)
  }

  # mencari max
  location = {}
  for cls in sorted(colors.keys()):
    max_values = [0] * 6 # Initialize max_values here
    if int(np.count_nonzero(mask == cls))>0 and cls>0:
      max_values[0] = int(np.count_nonzero(locations_maps[0] == cls)) # cek area T(U,R)
      max_values[1] = int(np.count_nonzero(locations_maps[1] == cls)) # cek area T(U,L)
      max_values[2] = int(np.count_nonzero(locations_maps[2] == cls)) # cek area T(M,L)
      max_values[3] = int(np.count_nonzero(locations_maps[3] == cls)) # cek area T(L,L)
      max_values[4] = int(np.count_nonzero(locations_maps[4] == cls)) # cek area T(L,R)
      max_values[5] = int(np.count_nonzero(locations_maps[5] == cls)) # cek area T(M,R)

      location[cls] = np.argmax(np.array(max_values)) # Use max_values for argmax
    else:
      location[cls] = -1
  return location

def build_feature(input_image_area, input_image_label, segmentation_model, label_model, colors):
    """
        membuat featur dan output dari data baru
        feature:
            "luas blue", "luas pengganti putih", "luas brown", "luas yellow", "luas purple", "luas darktail", "luas paru-paru", 
            "posisi blue", "posisi pengganti putih", "posisi brown", "posisi yellow", "posisi purple", "posisi darktail",
        return:
            image prediction, feature
    """
    location_labels = []

    # Prediction process
    ## Lung Prediction
    y_pred_lung = segmentation_model.predict(input_image_area)
    pred_lung = (y_pred_lung[0,:,:,0]> 0.5).astype(np.uint8)
    save_pred_lung = pred_lung.copy()
    save_pred_lung[save_pred_lung == 1] = 255
    save_pred_lung = np.array(save_pred_lung)
    area_lung = np.sum(pred_lung)
    
    ## Label Prediction
    y_pred_label = label_model.predict(input_image_label)[0]
    ### Ambil label per piksel
    pred_mask = np.argmax(y_pred_label, axis=-1) # Bentuk: (SIZE, SIZE)  
    ### ---- MODIFIKASI: Cetak kelas-kelas yang terprediksi sebelum mapping warna ----
    predicted_classes = np.unique(pred_mask)
    
    ### Konversi mask ke citra RGB
    rgb_image_label_prediction = label_to_rgb(pred_mask, colors)

    ### Hitung luas area untuk tiap kelas
    areas = calculate_areas(pred_mask, colors)

    # posisi label
    if np.sum(predicted_classes)>0:
        location_labels.append(find_postition(pred_lung, pred_mask, colors))
        # break
    else:
        location_labels.append({0: -1, 1: -1, 2:-1, 3: -1, 4: -1, 5: -1, 6: -1})
    
    return rgb_image_label_prediction, areas, area_lung, location_labels