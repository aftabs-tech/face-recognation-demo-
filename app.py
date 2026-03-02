
# # app.py
# from fastapi import FastAPI, File, UploadFile, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from PIL import Image
# import io
# import os
# import numpy as np
# import cv2
# import torch
# from transformers import CLIPProcessor, CLIPModel
# from sklearn.metrics.pairwise import cosine_similarity

# # -------- CONFIG --------
# ANIMALS_DIR = "animals"
# DEVICE = "cpu"  # switch to "cuda" if available

# # -------- FASTAPI SETUP --------
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------- MODEL LOADING --------
# device = torch.device(DEVICE)
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# model.to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # -------- LABELS --------
# animal_labels = [
#     "a fox",
#     "a panda",
#     "a wolf",
#     "an owl",
#     "a tiger",
#     "a lion",
#     "a cat",
#     "a dog"
# ]

# # -------- HELPERS (robust) --------
# def compute_text_features(labels):
#     inputs = processor(text=labels, return_tensors="pt", padding=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         try:
#             feats = model.get_text_features(**inputs)
#             if isinstance(feats, torch.Tensor):
#                 text_feats = feats
#             else:
#                 if hasattr(feats, "text_embeds"):
#                     text_feats = feats.text_embeds
#                 elif hasattr(feats, "pooler_output"):
#                     text_feats = feats.pooler_output
#                 else:
#                     raise RuntimeError("Unknown get_text_features return")
#         except Exception:
#             out = model(**inputs)
#             if hasattr(out, "text_embeds"):
#                 text_feats = out.text_embeds
#             elif hasattr(out, "pooler_output"):
#                 text_feats = out.pooler_output
#             else:
#                 raise RuntimeError("Unable to extract text embeddings")
#     arr = text_feats.detach().cpu().numpy()
#     arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
#     return arr

# def compute_image_feature_from_pil(pil_img: Image.Image):
#     inputs = processor(images=pil_img, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         try:
#             feats = model.get_image_features(**inputs)
#             if isinstance(feats, torch.Tensor):
#                 img_feats = feats
#             else:
#                 if hasattr(feats, "image_embeds"):
#                     img_feats = feats.image_embeds
#                 elif hasattr(feats, "pooler_output"):
#                     img_feats = feats.pooler_output
#                 else:
#                     raise RuntimeError("Unknown get_image_features return")
#         except Exception:
#             out = model(**inputs)
#             if hasattr(out, "image_embeds"):
#                 img_feats = out.image_embeds
#             elif hasattr(out, "pooler_output"):
#                 img_feats = out.pooler_output
#             else:
#                 raise RuntimeError("Unable to extract image embeddings")
#     arr = img_feats.detach().cpu().numpy()
#     if arr.ndim == 2 and arr.shape[0] == 1:
#         arr = arr[0]
#     arr = arr / np.linalg.norm(arr)
#     return arr

# # -------- PRECOMPUTE TEXT FEATURES --------
# text_features = compute_text_features(animal_labels)

# # -------- SERVE IMAGES --------
# if not os.path.exists(ANIMALS_DIR):
#     os.makedirs(ANIMALS_DIR, exist_ok=True)
# app.mount("/animals", StaticFiles(directory=ANIMALS_DIR), name="animals")

# # Build representative mapping
# animal_images = {}
# for label in animal_labels:
#     name = label.replace("a ", "").replace("an ", "").strip().lower()
#     folder = os.path.join(ANIMALS_DIR, name)
#     if os.path.isdir(folder):
#         files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#         if files:
#             # store relative path; we'll build absolute URL using request.base_url
#             animal_images[name.capitalize()] = f"/animals/{name}/{files[0]}"

# # -------- FACE DETECTOR --------
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # -------- /detect (returns top 3) --------
# @app.post("/detect")
# async def detect_animal(request: Request, file: UploadFile = File(...)):
#     contents = await file.read()
#     try:
#         img = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception:
#         return {
#             "animal": "Invalid image",
#             "similarity": 0,
#             "image_url": "",
#             "top": [],
#             "face_box": None
#         }

#     open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=6,
#         minSize=(120, 120)
#     )

#     if len(faces) == 0:
#         return {
#             "animal": "No face detected",
#             "similarity": 0,
#             "image_url": "",
#             "top": [],
#             "face_box": None
#         }

#     faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
#     x, y, w, h = faces[0]

#     if w < 120 or h < 120:
#         return {
#             "animal": "No face detected",
#             "similarity": 0,
#             "image_url": "",
#             "top": [],
#             "face_box": None
#         }

#     img_w, img_h = img.size
#     x1 = max(0, x)
#     y1 = max(0, y)
#     x2 = min(img_w, x + w)
#     y2 = min(img_h, y + h)

#     face_img = img.crop((x1, y1, x2, y2))

#     try:
#         image_feature = compute_image_feature_from_pil(face_img)
#     except Exception:
#         return {
#             "animal": "Error computing feature",
#             "similarity": 0,
#             "image_url": "",
#             "top": [],
#             "face_box": None
#         }

#     sims = cosine_similarity([image_feature], text_features)[0]

#     idxs = np.argsort(-sims)
#     top_count = min(3, len(idxs))
#     top = []

#     for i in range(top_count):
#         idx = int(idxs[i])
#         sim = float(sims[idx])
#         label = animal_labels[idx]
#         name = label.replace("a ", "").replace("an ", "").strip().capitalize()
#         rep = animal_images.get(name, "")
#         image_url = ""
#         if rep:
#             base = str(request.base_url).rstrip("/")
#             image_url = base + rep

#         top.append({
#             "animal": name,
#             "similarity": round(sim, 4),
#             "image_url": image_url
#         })

#     best = top[0] if len(top) > 0 else {
#         "animal": "Unknown",
#         "similarity": 0,
#         "image_url": ""
#     }

#     return {
#         "animal": best["animal"],
#         "similarity": round(best["similarity"], 4),
#         "image_url": best.get("image_url", ""),
#         "top": top,
#         "face_box": {
#             "x": int(x),
#             "y": int(y),
#             "w": int(w),
#             "h": int(h)
#         }
#     }




























































































# app.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
import numpy as np
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")



# -------- CONFIG --------
ANIMALS_DIR = "animals"
DEVICE = "cpu"  # switch to "cuda" if available

# -------- FASTAPI SETUP --------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- MODEL LOADING --------
device = torch.device(DEVICE)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -------- LABELS --------
animal_labels = [
    "a fox",
    "a panda",
    "a wolf",
    "an owl",
    "a tiger",
    "a lion",
    "a cat",
    "a dog"
]

# -------- HELPERS (robust) --------
def compute_text_features(labels):
    inputs = processor(text=labels, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        try:
            feats = model.get_text_features(**inputs)
            if isinstance(feats, torch.Tensor):
                text_feats = feats
            else:
                if hasattr(feats, "text_embeds"):
                    text_feats = feats.text_embeds
                elif hasattr(feats, "pooler_output"):
                    text_feats = feats.pooler_output
                else:
                    raise RuntimeError("Unknown get_text_features return")
        except Exception:
            out = model(**inputs)
            if hasattr(out, "text_embeds"):
                text_feats = out.text_embeds
            elif hasattr(out, "pooler_output"):
                text_feats = out.pooler_output
            else:
                raise RuntimeError("Unable to extract text embeddings")
    arr = text_feats.detach().cpu().numpy()
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    return arr

def compute_image_feature_from_pil(pil_img: Image.Image):
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        try:
            feats = model.get_image_features(**inputs)
            if isinstance(feats, torch.Tensor):
                img_feats = feats
            else:
                if hasattr(feats, "image_embeds"):
                    img_feats = feats.image_embeds
                elif hasattr(feats, "pooler_output"):
                    img_feats = feats.pooler_output
                else:
                    raise RuntimeError("Unknown get_image_features return")
        except Exception:
            out = model(**inputs)
            if hasattr(out, "image_embeds"):
                img_feats = out.image_embeds
            elif hasattr(out, "pooler_output"):
                img_feats = out.pooler_output
            else:
                raise RuntimeError("Unable to extract image embeddings")
    arr = img_feats.detach().cpu().numpy()
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    arr = arr / np.linalg.norm(arr)
    return arr

# -------- PRECOMPUTE TEXT FEATURES --------
text_features = compute_text_features(animal_labels)

# -------- SERVE IMAGES --------
if not os.path.exists(ANIMALS_DIR):
    os.makedirs(ANIMALS_DIR, exist_ok=True)
app.mount("/animals", StaticFiles(directory=ANIMALS_DIR), name="animals")

# Build representative mapping
animal_images = {}
for label in animal_labels:
    name = label.replace("a ", "").replace("an ", "").strip().lower()
    folder = os.path.join(ANIMALS_DIR, name)
    if os.path.isdir(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            # store relative path; we'll build absolute URL using request.base_url
            animal_images[name.capitalize()] = f"/animals/{name}/{files[0]}"

# -------- FACE DETECTOR --------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# -------- /detect (returns top 3) --------
@app.post("/detect")
async def detect_animal(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {
            "animal": "Invalid image",
            "similarity": 0,
            "image_url": "",
            "top": [],
            "face_box": None
        }

    open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(120, 120)
    )

    if len(faces) == 0:
        return {
            "animal": "No face detected",
            "similarity": 0,
            "image_url": "",
            "top": [],
            "face_box": None
        }

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]

    if w < 120 or h < 120:
        return {
            "animal": "No face detected",
            "similarity": 0,
            "image_url": "",
            "top": [],
            "face_box": None
        }

    img_w, img_h = img.size
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    face_img = img.crop((x1, y1, x2, y2))

    try:
        image_feature = compute_image_feature_from_pil(face_img)
    except Exception:
        return {
            "animal": "Error computing feature",
            "similarity": 0,
            "image_url": "",
            "top": [],
            "face_box": None
        }

    sims = cosine_similarity([image_feature], text_features)[0]

    idxs = np.argsort(-sims)
    top_count = min(3, len(idxs))
    top = []

    for i in range(top_count):
        idx = int(idxs[i])
        sim = float(sims[idx])
        label = animal_labels[idx]
        name = label.replace("a ", "").replace("an ", "").strip().capitalize()
        rep = animal_images.get(name, "")
        image_url = ""
        if rep:
            base = str(request.base_url).rstrip("/")
            image_url = base + rep

        top.append({
            "animal": name,
            "similarity": round(sim, 4),
            "image_url": image_url
        })

    best = top[0] if len(top) > 0 else {
        "animal": "Unknown",
        "similarity": 0,
        "image_url": ""
    }

    return {
        "animal": best["animal"],
        "similarity": round(best["similarity"], 4),
        "image_url": best.get("image_url", ""),
        "top": top,
        "face_box": {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h)
        }
    }


























































































