# pip install streamlit facenet-pytorch torch torchvision opencv-python-headless scikit-learn

import streamlit as st
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageDraw

IMG_SIZE = (160, 160)

@st.cache_resource
def load_model():
    return InceptionResnetV1(pretrained='vggface2').eval()

@st.cache_resource
def load_mtcnn():
    return MTCNN(image_size=160, margin=0, keep_all=False, post_process=True, device='cpu')

# استخراج الوجه + رسم البوكس عليه
def get_face_and_draw_box(mtcnn, img_pil):
    boxes, _ = mtcnn.detect(img_pil)
    if boxes is not None and len(boxes) > 0:
        draw = ImageDraw.Draw(img_pil)
        box = boxes[0]
        draw.rectangle(box.tolist(), outline="lime", width=3)
        face_tensor = mtcnn(img_pil)
        return face_tensor.unsqueeze(0), img_pil, box
    return None, img_pil, None

# استخراج embedding
def get_embedding(model, face_tensor):
    with torch.no_grad():
        emb = model(face_tensor)
    return emb.squeeze().numpy()

# Normalization
def normalize(emb):
    return emb / np.linalg.norm(emb)

# مقارنة cosine
def check_presence(reg_emb, cur_emb):
    reg_emb = normalize(reg_emb)
    cur_emb = normalize(cur_emb)
    similarity = cosine_similarity([reg_emb], [cur_emb])[0][0]
    return similarity >= 0.6, similarity  # نرجع النسبة كمان

# Streamlit UI
st.title("📸 Face Presence Detection (FaceNet + Cosine Similarity)")

# التسجيل
st.sidebar.header("🧑 Enroll")
name = st.sidebar.text_input("Name:")
upload_img = st.sidebar.file_uploader("Upload a clear face image", type=['jpg', 'jpeg', 'png'])

if st.sidebar.button("✅ Enroll"):
    if name and upload_img:
        img = Image.open(upload_img).convert("RGB")
        mtcnn = load_mtcnn()
        model = load_model()

        face_tensor, img_with_box, box = get_face_and_draw_box(mtcnn, img.copy())

        if face_tensor is not None:
            emb = get_embedding(model, face_tensor)
            st.session_state.registered = True
            st.session_state.name = name
            st.session_state.reg_emb = emb
            st.session_state.counter = 0
            st.sidebar.success(f"✅ Enrolled: {name}")

            st.subheader("🖼 Uploaded Image with Face Detection")
            st.image(img_with_box, caption="Face Detected", use_column_width=True)
        else:
            st.sidebar.error("❌ No face detected. Please upload a clear frontal face.")
    else:
        st.sidebar.error("⚠️ Please provide both name and image.")

# المطابقة بالكاميرا
if st.session_state.get('registered'):
    st.header("📷 Take a Picture for Matching")
    cam_img = st.camera_input("Take a picture")

    if cam_img:
        image = Image.open(cam_img).convert("RGB")
        mtcnn = load_mtcnn()
        model = load_model()

        face_tensor, img_with_box, box = get_face_and_draw_box(mtcnn, image.copy())

        if face_tensor is not None:
            emb = get_embedding(model, face_tensor)
            matched, similarity = check_presence(st.session_state.reg_emb, emb)

            st.subheader("📦 Detected Face in Captured Image")
            st.image(img_with_box, caption="Detected Face with Bounding Box", use_column_width=True)

            if matched:
                st.success(f"✅ Face Matched! (Similarity: {similarity:.3f})")
            else:
                st.error(f"❌ Face Not Matched. (Similarity: {similarity:.3f})")
        else:
            st.error("❌ No face detected in the captured image.")
