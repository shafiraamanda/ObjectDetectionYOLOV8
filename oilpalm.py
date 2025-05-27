import streamlit as st
import base64
import os
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from io import BytesIO
from ultralytics import YOLO
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections

# Fungsi untuk encode gambar lokal ke base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Fungsi untuk menampilkan logo di kiri atas
def render_logo():
    if not os.path.exists("Saraswanti-Logo.png"):
        st.warning("⚠️ Logo Saraswanti tidak ditemukan.")
    else:
        logo_base64 = get_base64_image("Saraswanti-Logo.png")
        st.markdown(
            f"""
            <div style="position: fixed; top: 10px; left: 10px; z-index: 999;">
                <img src="data:image/png;base64,{logo_base64}" alt="Logo Saraswanti" width="120" />
            </div>
            """,
            unsafe_allow_html=True
        )

# Set halaman
st.set_page_config(page_title="Deteksi Buah Sawit", layout="centered")

# Tampilkan logo
render_logo()

# Load model hanya sekali
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# Fungsi prediksi
def predict_image(model, image):
    image = np.array(image.convert("RGB"))
    results = model(image)
    return results

# Warna sesuai label
label_to_color = {
    "Masak": Color.RED,
    "Mengkal": Color.YELLOW,
    "Mentah": Color.BLACK
}

label_annotator = LabelAnnotator()

# Gambar hasil deteksi dan beri label
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

# Inisialisasi state
if "camera_image" not in st.session_state:
    st.session_state["camera_image"] = ""

# Judul Aplikasi
st.title("📷 Deteksi dan Klasifikasi Kematangan Buah Sawit")
st.markdown("Pilih metode input gambar:")
option = st.radio("", ["Upload Gambar", "Gunakan Kamera"])
image = None

# Input via upload
if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

# Input via kamera
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

# Deteksi dan visualisasi
if image:
    with st.spinner("🔍 Memproses gambar..."):
        model = load_model()
        results = predict_image(model, image)
        img_with_boxes, class_counts = draw_results(image, results)

        st.image(img_with_boxes, caption="📊 Hasil Deteksi", use_container_width=True)
        st.subheader("Jumlah Objek Terdeteksi:")
        for name, count in class_counts.items():
            st.write(f"- **{name}**: {count}")
