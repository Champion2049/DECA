import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import torch
import numpy as np
from loguru import logger

# --- MediaPipe ---
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_detection as mp_face
    logger.info("MediaPipe Loaded")
except Exception as e:
    logger.error(f"MediaPipe import failed: {e}")
    sys.exit()

# --- DECA ---
from decalib.deca import DECA
from decalib.utils.config import cfg

# ---------------- CONFIG ----------------
MODEL_PATH = os.path.join(os.getcwd(), 'data', 'deca_model.tar')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg.model.pretrained_modelpath = MODEL_PATH
cfg.pretrained_modelpath = MODEL_PATH
cfg.model.use_tex = False   # 🔥 KEEP THIS OFF (prevents crashes)

# ---------------- INIT DECA ----------------
try:
    deca = DECA(config=cfg, device=DEVICE)
    deca.eval()
    logger.info(f"DECA loaded on {DEVICE}")
except Exception as e:
    logger.error(f"DECA init failed: {e}")
    sys.exit()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# ---------------- MAIN LOOP ----------------
logger.info("Press ESC to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if results.detections:
        for det in results.detections:
            try:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                margin = 40
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + bw + margin)
                y2 = min(h, y + bh + margin)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # --- PREPROCESS ---
                img = cv2.resize(face, (224, 224)).astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

                # --- DECA ---
                with torch.no_grad():
                    codedict = deca.encode(img)
                    opdict, visdict = deca.decode(codedict)

                rendered = None

                # ---------- METHOD 1: visdict ----------
                if visdict and 'rendered_images' in visdict:
                    r = visdict['rendered_images'][0].cpu().numpy().transpose(1, 2, 0)
                    rendered = (np.clip(r, 0, 1) * 255).astype(np.uint8)
                    rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

                # ---------- METHOD 2: shape fallback ----------
                elif 'shape_images' in opdict:
                    r = opdict['shape_images'][0].cpu().numpy().transpose(1, 2, 0)
                    rendered = (np.clip(r, 0, 1) * 255).astype(np.uint8)
                    rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

                # ---------- APPLY ----------
                if rendered is not None:
                    rendered = cv2.resize(rendered, (x2 - x1, y2 - y1))
                    frame[y1:y2, x1:x2] = cv2.addWeighted(
                        frame[y1:y2, x1:x2],
                        0.4,
                        rendered,
                        0.6,
                        0
                    )
                else:
                    # LANDMARK FALLBACK
                    lms = opdict.get('landmarks2d', opdict.get('lms', None))
                    if lms is not None:
                        lms = lms[0].cpu().numpy()
                        lms = (lms + 1) * 112

                        for p in lms:
                            px = int(p[0] * (x2 - x1) / 224 + x1)
                            py = int(p[1] * (y2 - y1) / 224 + y1)
                            cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

            except Exception as e:
                logger.warning(f"Frame error: {e}")
                continue

    cv2.imshow("DECA Stable", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



