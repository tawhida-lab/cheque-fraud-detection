import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Détection de chèques falsifiés", page_icon="🏦", layout="centered")

@st.cache_resource
def charger_modele():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    model.eval()
    return model

model   = charger_modele()
CLASSES = ["Authentique", "Falsifié"]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gradients_cam   = []
activations_cam = []

def hook_bw(module, grad_input, grad_output):
    gradients_cam.append(grad_output[0])

def hook_fw(module, input, output):
    activations_cam.append(output)

model.layer4[-1].register_forward_hook(hook_fw)
model.layer4[-1].register_backward_hook(hook_bw)

def calculer_gradcam(img_pil):
    gradients_cam.clear()
    activations_cam.clear()
    tensor     = preprocess(img_pil).unsqueeze(0)
    output     = model(tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = F.softmax(output, dim=1)[0][pred_class].item()
    model.zero_grad()
    output[0, pred_class].backward()
    grads   = gradients_cam[0].squeeze()
    acts    = activations_cam[0].squeeze()
    weights = grads.mean(dim=(1, 2))
    cam     = (weights[:, None, None] * acts).sum(dim=0)
    cam     = F.relu(cam)
    cam     = cam - cam.min()
    cam     = cam / (cam.max() + 1e-8)
    cam     = cam.cpu().detach().numpy()
    w, h    = img_pil.size
    img_arr = np.array(img_pil)
    heatmap = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_arr + 0.45 * heatmap).astype(np.uint8)
    return pred_class, confidence, overlay

st.title("🏦 Détection de chèques falsifiés")
st.caption("Modèle ResNet50 — Transfer Learning · Grad-CAM · 96% accuracy")
st.divider()

uploaded = st.file_uploader("Uploade une image de chèque", type=["png", "jpg", "jpeg"])

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    st.image(img_pil, caption="Image uploadée", use_container_width=True)

    with st.spinner("Analyse en cours..."):
        pred, conf, overlay = calculer_gradcam(img_pil)

    st.divider()

    if pred == 0:
        st.success(f"✓ Chèque AUTHENTIQUE — confiance : {conf*100:.1f}%")
    else:
        st.error(f"✗ Chèque FALSIFIÉ — confiance : {conf*100:.1f}%")

    st.progress(float(conf))
    st.subheader("Zones analysées par le modèle (Grad-CAM)")
    st.caption("Rouge = zone très influente · Bleu = peu influente")
    st.image(overlay, use_container_width=True)
    st.divider()
    st.caption("Projet portfolio — Computer Vision · Détection de fraude documentaire")