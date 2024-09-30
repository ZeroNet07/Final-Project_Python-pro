import numpy as np
import cv2
import json

# Daftar kelas sampah
classes = ["Baterai9V", "BateraiA2", "Botol", "Bungkus_Mie", "Bungkus_makanan", "Bungkus_minuman"]

# Daftar sampah organik dan nonorganik
sampah_organik = ["Bungkus_makanan", "Bungkus_minuman"]  # Contoh sampah organik
sampah_nonorganik = ["Baterai9V", "BateraiA2", "Botol", "Bungkus_Mie"]  # Contoh sampah nonorganik

# Inisialisasi kamera dan model ONNX
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromONNX(r"C:\Users\USER\Desktop\Python\.(Project) Menditeksi dan Pengenalan Sampah\Data set\Deteksi-Objek-dengan-YOLOV5-main\Deteksi-Objek-dengan-YOLOV5-main\best.onnx")

# Pengecekan kamera dan model
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

if net.empty():
    print("Error: Model ONNX tidak ditemukan.")
    exit()

while True:
    # Membaca frame dari kamera
    _, img = cap.read()

    # Preprocessing input untuk YOLO
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=[640, 640], mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]

    # Inisialisasi daftar untuk kelas, kepercayaan, dan bounding boxes
    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    # Dapatkan ukuran gambar
    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width / 640
    y_scale = img_height / 640

    # Deteksi objek dari hasil YOLO
    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.2:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.2:
                # Simpan informasi deteksi
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx - w / 2) * x_scale)
                y1 = int((cy - h / 2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1, y1, width, height])
                boxes.append(box)

    # Non-Max Suppression untuk menghindari multiple detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)

    # Inisialisasi variabel penghitung jumlah sampah organik dan nonorganik
    total_organik = 0
    total_nonorganik = 0

    # Periksa apakah ada deteksi setelah NMS
    if len(indices) > 0:
        # Looping untuk setiap objek terdeteksi
        for i in indices.flatten():
            x1, y1, w, h = boxes[i]
            label = classes[classes_ids[i]]
            conf = confidences[i]
            text = label + " {:.2f}".format(conf)

            # Gambar kotak deteksi dan label
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

            # Tentukan apakah sampah organik atau nonorganik
            if label in sampah_organik:
                total_organik += 1
            elif label in sampah_nonorganik:
                total_nonorganik += 1

    # Tampilkan jumlah total sampah organik dan nonorganik yang terdeteksi pada frame
    cv2.putText(img, f"Sampah Organik: {total_organik}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Sampah Non-Organik: {total_nonorganik}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame dengan deteksi
    cv2.imshow("Deteksi Objek", img)

    # Keluar dari loop jika tombol 'Esc' ditekan
    if cv2.waitKey(1) & 0xff == 27:
        break

# Hentikan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()