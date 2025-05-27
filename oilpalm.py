import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import base64
from io import BytesIO
from ultralytics import YOLO
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections
from datetime import datetime

# --------------------- Konfigurasi Halaman ---------------------
st.set_page_config(page_title="Deteksi Buah Sawit", layout="centered")

# --------------------- CSS Kustom untuk Background Hitam ---------------------
st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: white;
        }
        .stApp {
            background-color: #000000;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------- Logo Saraswanti dari GitHub ---------------------
st.image("https://raw.githubusercontent.com/shafiraamanda/ObjectDetectionYOLOV8/main/Saraswanti-Logo.png", width=150, caption="Saraswanti Group")

# --------------------- Bar Atas: Profil & Status ---------------------
col1, col2 = st.columns([1, 5])

with col1:
    st.image("/mnt/data/74f139dd-1bff-4b06-b5d2-e1f092c7cc4d.png", width=80, caption="Profil")

with col2:
    now = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"""
        <div style='text-align: right; font-size:16px; color: lime;'>
            🔄 <b>RUNNING...</b> &nbsp;&nbsp; 🕒 {now}
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------- Load Model ---------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Ganti path model sesuai punyamu

# --------------------- Fungsi Prediksi ---------------------
def predict_image(model, image):
    image = np.array(image.convert("RGB"))
    results = model(image)
    return results

# --------------------- Warna Label ---------------------
label_to_color = {
    "Masak": Color.RED,
    "Mengkal": Color.YELLOW,
    "Mentah": Color.BLACK
}
label_annotator = LabelAnnotator()

# --------------------- Fungsi Gambar Hasil Deteksi ---------------------
def draw_results(image, results):
    img = np.array(image.convert("RGB"))
    class_counts = Counter()

    for result in results:
        boxes = result.boxes
        names = result.names

        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(xyxy, class_ids, confidences):
            class_name = names[class_id]
            label = f"{class_name}: {conf:.2f}"
            color = label_to_color.get(class_name, Color.WHITE)

            class_counts[class_name] += 1

            box_annotator = BoxAnnotator(color=color)
            detection = Detections(
                xyxy=np.array([box]),
                confidence=np.array([conf]),
                class_id=np.array([class_id])
            )

            img = box_annotator.annotate(scene=img, detections=detection)
            img = label_annotator.annotate(scene=img, detections=detection, labels=[label])

    return img, class_counts

# --------------------- Inisialisasi Session ---------------------
if "camera_image" not in st.session_state:
    st.session_state["camera_image"] = ""

# --------------------- UI Utama ---------------------
st.title("📷 Deteksi dan Klasifikasi Kematangan Buah Sawit")
st.markdown("Pilih metode input gambar:")

option = st.radio("", ["Upload Gambar", "Gunakan Kamera"])
image = None

# --------------------- Upload Gambar ---------------------
if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

# --------------------- Kamera Langsung ---------------------
elif option == "Gunakan Kamera":
    st.markdown("### Kamera Belakang (Environment)")

    camera_html = """
    <div style="text-align:center;">
        <video id="video" autoplay playsinline style="width:100%; border:1px solid gray;"></video>
        <br/>
        <button onclick="takePhoto()" style="margin-top:10px; padding:10px 20px;">📸 Ambil Gambar</button>
        <canvas id="canvas" style="display:none;"></canvas>
    </div>

    <script>
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: "environment" } },
                    audio: false
                });
                const video = document.getElementById('video');
                video.srcObject = stream;
            } catch (err) {
                alert("Gagal mengakses kamera: " + err.message);
            }
        }

        function takePhoto() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/png');

            const input = window.parent.document.querySelector('input[data-testid="stTextInput"]');
            if (input) {
                input.value = dataURL;
                input.dispatchEvent(new Event("input", { bubbles: true }));
            }
        }

        document.addEventListener("DOMContentLoaded", startCamera);
    </script>
    """

    st.components.v1.html(camera_html, height=600)
    base64_img = st.text_input("Gambar dari Kamera (tersembunyi)", type="default", label_visibility="collapsed")

    if base64_img.startswith("data:image"):
        st.session_state["camera_image"] = base64_img

        try:
            header, encoded = base64_img.split(",", 1)
            decoded = base64.b64decode(encoded)
            image = Image.open(BytesIO(decoded))
            st.image(image, caption="📷 Gambar dari Kamera", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memproses gambar dari kamera: {e}")

# --------------------- Proses Deteksi ---------------------
if image:
    with st.spinner("🔍 Memproses gambar..."):
        model = load_model()
        results = predict_image(model, image)
        img_with_boxes, class_counts = draw_results(image, results)

        st.image(img_with_boxes, caption="📊 Hasil Deteksi", use_container_width=True)
        st.subheader("Jumlah Objek Terdeteksi:")
        for name, count in class_counts.items():
            st.write(f"- **{name}**: {count}")

    st.success("✅ Proses selesai pada " + datetime.now().strftime("%H:%M:%S"))
