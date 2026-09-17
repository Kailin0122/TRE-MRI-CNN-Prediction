"""
Run with:
    streamlit run TRE_Prediction_GUI_NEW_Only_DenseNet.py
"""
import os
import tempfile

import cv2
import numpy as np
import nibabel as nib
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Prediction of Tumor Related Epilepsy using MRI", layout="wide")

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["NonTRE", "TRE"]
POSITIVE_CLASS = "TRE"

# Final selected model — DenseNet121
MODEL_PATH = r"C:\Users\User\PycharmProjects\FYP_Retrain\DenseNet121_CLAHE_Results_SubjectLevel\DenseNet121_CLAHE_Best_Fold5.pth"

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_clahe)


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


# ─────────────────────────────────────────────
# MODEL BUILDER — DenseNet121 only
# ─────────────────────────────────────────────
def build_densenet121(nc):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, nc)
    return m


@st.cache_resource(show_spinner="Loading final DenseNet121 model...")
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    model = build_densenet121(NUM_CLASSES)
    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for DenseNet121, hooked on the final feature block (model.features).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        # Clone immediately so the later in-place ReLU (applied to this same
        # tensor just after model.features returns) can't overwrite the
        # values we use as the CAM activations.
        self.activations = out.detach().clone()
        # Plain inference (predict_subject) runs under torch.no_grad(), so
        # `out` won't require grad there — only attach the gradient hook
        # when we're actually doing a Grad-CAM backward pass, otherwise
        # register_hook() raises "tensor that doesn't require gradient".
        if out.requires_grad:
            out.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        self.gradients = None
        output = self.model(input_tensor)
        score = output[:, class_idx].sum()
        score.backward()

        gradients = self.gradients            # (1, C, H, W)
        activations = self.activations        # (1, C, H, W)
        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
        return cam


def overlay_heatmap(orig_rgb_uint8, cam, alpha=0.45):
    h, w = orig_rgb_uint8.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * orig_rgb_uint8).astype(np.uint8)
    return overlay


# ─────────────────────────────────────────────
# NIFTI HELPERS
# ─────────────────────────────────────────────
def load_nifti_volume(uploaded_file):
    """Write the uploaded NIfTI to a temp file and load it with nibabel."""
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    img = nib.load(tmp_path)
    data = img.get_fdata()
    os.unlink(tmp_path)
    return data


def normalize_slice_to_uint8(slice2d):
    s = slice2d.astype(np.float32)
    s = s - s.min()
    if s.max() > 0:
        s = s / s.max()
    return (s * 255).astype(np.uint8)


def get_axial_slices(volume):
    """
    Returns a list of axial slices as uint8 RGB numpy arrays.
    Assumes the 3rd axis (index 2) is the axial/slice axis, which is the
    standard convention for NIfTI volumes saved as (X, Y, Z).
    """
    n_slices = volume.shape[2]
    slices = []
    for i in range(n_slices):
        sl = volume[:, :, i]
        sl = np.rot90(sl)  # radiological orientation for display
        sl_uint8 = normalize_slice_to_uint8(sl)
        sl_rgb = cv2.cvtColor(sl_uint8, cv2.COLOR_GRAY2RGB)
        slices.append(sl_rgb)
    return slices


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def predict_subject(model, tfm, selected_slices_rgb):
    """Runs DenseNet121 on each selected slice and averages the softmax
    probabilities to get one subject-level prediction (same aggregation
    logic used during evaluation)."""
    pil_images = [Image.fromarray(sl) for sl in selected_slices_rgb]
    batch = torch.stack([tfm(img) for img in pil_images]).to(DEVICE)

    with torch.no_grad():
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()  # (n_slices, 2)

    mean_probs = probs.mean(axis=0)  # (2,)
    positive_idx = CLASS_NAMES.index(POSITIVE_CLASS)
    final_idx = int(np.argmax(mean_probs))

    return {
        "final_label": CLASS_NAMES[final_idx],
        "final_tre_probability": float(mean_probs[positive_idx]),
        "per_slice_tre_probability": probs[:, positive_idx].tolist(),
        "n_slices_used": len(pil_images),
        "batch_tensor": batch,   # reused for Grad-CAM
        "positive_idx": positive_idx,
    }


@st.cache_resource(show_spinner=False)
def get_gradcam_extractor(_model):
    # Leading underscore on the arg tells st.cache_resource not to hash it
    # (the model object isn't hashable/picklable in a meaningful way here).
    # Cached so the forward hook below is registered exactly once per model
    # instance, instead of piling up a new hook on every Grad-CAM run.
    return GradCAM(_model, _model.features)


def run_gradcam_on_selected(model, batch_tensor, class_idx, selected_slices_rgb):
    cam_extractor = get_gradcam_extractor(model)

    overlays = []
    for i in range(batch_tensor.shape[0]):
        single = batch_tensor[i:i + 1].clone().requires_grad_(True)
        cam = cam_extractor.generate(single, class_idx)
        overlay = overlay_heatmap(selected_slices_rgb[i], cam)
        overlays.append(overlay)
    return overlays


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title(" Model")
st.sidebar.info("**Final model: DenseNet121** (CLAHE preprocessing, subject-level, Fold 5)")

model = None
load_error = None
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    load_error = str(e)

if load_error:
    st.sidebar.error(f"Could not load model:\n{load_error}")

# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.title("Prediction of Tumor Related Epilepsy using MRI")
st.write(
    "Upload a NIfTI MRI volume (.nii or .nii.gz) for **one patient/subject**. "
    "All axial slices will be displayed below — select the slice(s) that contain "
    "the tumor, then run the prediction. Grad-CAM visualizations will show which "
    "regions of the selected slices the model focused on."
)

if model is None:
    st.warning("The model could not be loaded. Please check the model path.")
    st.stop()

uploaded_nifti = st.file_uploader("Upload NIfTI file (.nii / .nii.gz)", type=["nii", "gz"])

if uploaded_nifti is not None:
    with st.spinner("Loading NIfTI volume and extracting axial slices..."):
        try:
            volume = load_nifti_volume(uploaded_nifti)
            axial_slices = get_axial_slices(volume)
        except Exception as e:
            st.error(f"Failed to load NIfTI file: {e}")
            st.stop()

    st.success(f"Loaded volume with {len(axial_slices)} axial slices.")

    st.subheader("Step 1 — Select the slice(s) containing tumor")
    st.caption("Tick the checkbox under any slice you want to use for prediction. You can select multiple slices.")

    n_cols = 6
    selected_indices = []
    rows = (len(axial_slices) + n_cols - 1) // n_cols
    idx = 0
    for _ in range(rows):
        cols = st.columns(n_cols)
        for c in cols:
            if idx >= len(axial_slices):
                break
            with c:
                st.image(axial_slices[idx], caption=f"Slice {idx}", use_container_width=True)
                if st.checkbox("Select this slice", key=f"slice_{idx}"):
                    selected_indices.append(idx)
            idx += 1

    st.divider()
    st.subheader("Step 2 — Run prediction")

    if selected_indices:
        st.write(f"Selected slices: {selected_indices}")
    else:
        st.info("No slices selected yet — tick at least one slice above to enable prediction.")

    if st.button("Run Prediction", type="primary", disabled=(len(selected_indices) == 0)):
        selected_slices_rgb = [axial_slices[i] for i in selected_indices]
        tfm = build_transform()

        with st.spinner("Running DenseNet121 inference..."):
            result = predict_subject(model, tfm, selected_slices_rgb)

        st.subheader("Prediction Result")
        label = result["final_label"]
        score = result["final_tre_probability"]

        col1, col2 = st.columns([1, 1])
        with col1:
            if label == POSITIVE_CLASS:
                st.error(f"### Predicted: {label}")
            else:
                st.success(f"### Predicted: {label}")
            st.metric("TRE probability", f"{score * 100:.1f}%")
            st.caption(f"Slices used: {result['n_slices_used']}")

        with col2:
            st.write("**Per-slice TRE probability**")
            st.bar_chart(
                {f"Slice {selected_indices[i]}": p for i, p in enumerate(result["per_slice_tre_probability"])}
            )

        st.divider()
        st.subheader("Step 3 — Grad-CAM visualization")
        st.caption(
            "Highlighted regions show which parts of each slice most influenced the "
            "model's prediction — use this to sanity-check clinical plausibility."
        )

        with st.spinner("Generating Grad-CAM overlays..."):
            overlays = run_gradcam_on_selected(
                model,
                result["batch_tensor"],
                result["positive_idx"] if label == POSITIVE_CLASS else CLASS_NAMES.index(label),
                selected_slices_rgb,
            )

        gc_cols = st.columns(min(4, len(overlays)))
        for i, overlay in enumerate(overlays):
            gc_cols[i % len(gc_cols)].image(
                overlay, caption=f"Slice {selected_indices[i]} — Grad-CAM", use_container_width=True
            )

        with st.expander("Raw values"):
            st.json({k: v for k, v in result.items() if k != "batch_tensor"})
else:
    st.info("Waiting for NIfTI upload...")