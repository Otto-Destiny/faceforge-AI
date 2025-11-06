"""
Face recognition module using MTCNN for detection and InceptionResnetV1 for embedding.

Expectations:
- A file named 'embeddings.pt' must be present (list of (tensor, name) OR dict name->tensor).
- facenet-pytorch (MTCNN, InceptionResnetV1), torch, PIL, matplotlib must be installed.

Functions provided:
- locate_faces(image)
- determine_name_dist(cropped_image, threshold=0.9)
- label_face(name, dist, box, axis)
- add_labels_to_image(image)

"""

from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------------
# Model & resources setup
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detector (different init style/params)
_detector = MTCNN(keep_all=True, margin=10, select_largest=False, device=DEVICE, min_face_size=40)

# Embedder
_embedder = InceptionResnetV1(pretrained="vggface2").to(DEVICE).eval()

# Load embeddings (support both list-of-tuples and dict)
_EMB_PATH = Path("embeddings.pt")
if not _EMB_PATH.exists():
    raise FileNotFoundError("embeddings.pt not found — place embeddings.pt alongside this file.")

_raw_embeddings = torch.load(str(_EMB_PATH), map_location=DEVICE)

# canonicalize embeddings into a list of (tensor, name)
_known_embeddings: List[Tuple[Tensor, str]] = []
if isinstance(_raw_embeddings, dict):
    for nm, emb in _raw_embeddings.items():
        _known_embeddings.append((emb.to(DEVICE), nm))
elif isinstance(_raw_embeddings, (list, tuple)):
    # Expect list of (tensor, name) or (name, tensor)
    for item in _raw_embeddings:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            a, b = item
            # detect order
            if isinstance(a, torch.Tensor) and isinstance(b, str):
                _known_embeddings.append((a.to(DEVICE), b))
            elif isinstance(b, torch.Tensor) and isinstance(a, str):
                _known_embeddings.append((b.to(DEVICE), a))
            else:
                # fallback: try to coerce first to tensor, second to str
                emb = a if isinstance(a, torch.Tensor) else b
                name = b if isinstance(b, str) else a
                _known_embeddings.append((emb.to(DEVICE), str(name)))
        else:
            raise ValueError("Unrecognized embeddings.pt format; expected list of (tensor,name) pairs.")
else:
    raise ValueError("Unsupported embeddings.pt structure — expected dict or list.")

# -------------
# Utilities
# -------------
def _pil_to_tensor(img: Image.Image) -> Tensor:
    """
    Convert PIL Image to CHW float tensor normalized to [0,1], matching facenet-pytorch input style.
    If MTCNN already returns cropped tensors, this helper may not be needed externally.
    """
    arr = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                           .float().view(img.size[1], img.size[0], len(img.getbands()))
                           .permute(2, 0, 1).numpy()))
    # above conversion is intentionally different style; we'll primarily rely on MTCNN's outputs.
    # This function kept for completeness / possible future use.
    return arr.float().unsqueeze(0) / 255.0


# -------------------------
# Public functions
# -------------------------
def locate_faces(image: Image.Image) -> List[Tuple[List[float], float, Tensor]]:
    """
    Runs the MTCNN on a PIL image and returns a list of (box, probability, cropped_face_tensor).
    - box is [x1, y1, x2, y2] in image coordinates.
    - probability is detection confidence (0..1)
    - cropped_face_tensor is a torch.Tensor ready for the embedder (C x H x W), on DEVICE
    """
    # facenet-pytorch MTCNN returns cropped tensors and probabilities when used like below
    # we call it twice to get boxes + crops + probabilities in a robust way
    crops, probs = _detector(image, return_prob=True)
    boxes, _ = _detector.detect(image)

    results = []
    if boxes is None or crops is None:
        return results

    # crops may be a single tensor (when keep_all=False), but here keep_all=True so expect list
    for idx, (box, prob) in enumerate(zip(boxes, probs)):
        if prob is None:
            continue
        # Convert crop to expected device and shape
        crop = crops[idx] if isinstance(crops, (list, tuple)) else crops
        if isinstance(crop, torch.Tensor):
            face_tensor = crop.to(DEVICE)
        else:
            # if crop is PIL image, transform to tensor using facenet's helper (here manual)
            face_tensor = _detector.face_crop(image, box).to(DEVICE) if hasattr(_detector, "face_crop") else torch.tensor([]).to(DEVICE)
        results.append((list(map(float, box)), float(prob), face_tensor))

    return results


def determine_name_dist(cropped_image: Tensor, threshold: float = 0.9) -> Tuple[str, float]:
    """
    Given a cropped face tensor (CxHxW), produce the embedding and find the closest known embedding.
    Returns (name, distance). If distance >= threshold, name will be 'Undetected'.
    """
    if not isinstance(cropped_image, torch.Tensor):
        raise TypeError("cropped_image must be a torch.Tensor (C,H,W) as returned by MTCNN.")

    # Ensure batch dimension
    img_batch = cropped_image.unsqueeze(0).to(DEVICE)  # shape (1, C, H, W)
    with torch.no_grad():
        emb = _embedder(img_batch)  # shape (1, 512)
    emb = emb.squeeze(0)  # shape (512,)

    # compute all L2 distances
    # stack known embeddings into matrix for vectorized distance computation
    known_stack = torch.stack([k for k, _ in _known_embeddings], dim=0)  # (N, 512)
    # emb (512,) -> (N, 512)
    dists = torch.norm(known_stack - emb.unsqueeze(0), dim=1, p=2)  # (N,)

    # find best match
    best_idx = int(torch.argmin(dists))
    best_dist = float(dists[best_idx])
    best_name = _known_embeddings[best_idx][1]

    if best_dist < float(threshold):
        return best_name, best_dist
    else:
        return "Undetected", best_dist


def label_face(name: str, dist: float, box: List[float], axis):
    """
    Draws a rectangle and label on the provided matplotlib axis.
    box = [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    # color logic: known -> blue-ish, unknown -> red
    color = "tab:blue" if name != "Undetected" else "tab:red"

    # draw rectangle (slightly thicker)
    rect = Rectangle((x1, y1), w, h, linewidth=2.2, edgecolor=color, facecolor="none", zorder=2)
    axis.add_patch(rect)

    # label text: name and distance (rounded)
    text = f"{name} {dist:.2f}"
    # place background box behind text for readability
    axis.text(x1, y1 - 6, text, fontsize=12, color=color,
              bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1), zorder=3)


def add_labels_to_image(image: Image.Image) -> matplotlib.figure.Figure:
    """
    Runs detection + recognition, draws boxes & labels onto a matplotlib Figure, and returns it.
    """
    # Prepare figure sized to image pixels
    img_w, img_h = image.size
    DPI = 96
    fig = plt.figure(figsize=(img_w / DPI, img_h / DPI), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.axis("off")

    # Detect faces
    faces = locate_faces(image)

    for box, prob, crop in faces:
        # skip low-confidence detections (same 0.9 cutoff)
        if prob is None or prob < 0.9:
            continue

        # compute identity + distance
        try:
            name, distance = determine_name_dist(crop, threshold=0.9)
        except Exception:
            # in case embedder fails for any crop, mark undetected
            name, distance = "Undetected", float("inf")

        # add annotations
        label_face(name, distance, box, ax)

    # tight layout and return
    plt.tight_layout(pad=0)
    return fig


#© 2025 Destiny Otto