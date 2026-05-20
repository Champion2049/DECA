import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import cv2
import torch
import numpy as np
from loguru import logger

# =========================================================
# LEGACY PATCH FOR CHUMPY
# =========================================================
for name, target in {
    'bool': bool, 'int': int, 'float': float,
    'object': object, 'str': str
}.items():
    if not hasattr(np, name):
        setattr(np, name, target)

# =========================================================
# ADD DECA TO PATH
# =========================================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
import face_alignment

# =========================================================
# CONFIG
# =========================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_WEIGHTS = r"./data/deca_model.tar"

deca_cfg.defrost()
deca_cfg.device = DEVICE
deca_cfg.pretrained_modelpath = BASE_WEIGHTS
deca_cfg.model.pretrained_modelpath = BASE_WEIGHTS
deca_cfg.model.use_tex = False

# FIX: Completely removed the forced 'aa' jaw_type override 
# to let the network fall back to its native trained Euler state.

deca_cfg.rasterizer_type = 'pytorch3d' 
deca_cfg.freeze()

# =========================================================
# INIT MODELS
# =========================================================
logger.info(f"Initializing DECA on {DEVICE}...")
deca = DECA(config=deca_cfg, device=DEVICE)
deca.eval()
logger.info("DECA ready.")

logger.info("Initializing FAN landmark detector...")
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device=DEVICE,
    flip_input=False
)
logger.info("FAN ready.")

IMAGE_SIZE = 224

# =========================================================
# HELPER: CLEAN IMAGE PREPROCESSING
# =========================================================
def landmarks_to_crop_tensor(bgr_frame):
    """
    Tracks landmarks in RGB, crops the frame cleanly, and returns 
    the standard normalized float32 tensor array for inference.
    """
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    kps_list = fa.get_landmarks(rgb_frame)
    
    if kps_list is None or len(kps_list) == 0:
        return None

    kps = kps_list[0]

    left   = kps[:, 0].min()
    right  = kps[:, 0].max()
    top    = kps[:, 1].min()
    bottom = kps[:, 1].max()

    old_size = max(right - left, bottom - top)

    center = np.array([
        (left + right) / 2.0,
        (top + bottom) / 2.0 + old_size * 0.14
    ])
    size = int(old_size * 1.58)

    src_pts = np.float32([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] + size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
    ])
    dst_pts = np.float32([
        [0,          0],
        [IMAGE_SIZE, 0],
        [0,          IMAGE_SIZE],
    ])

    M = cv2.getAffineTransform(src_pts, dst_pts)
    
    crop_rgb = cv2.warpAffine(
        rgb_frame, M, (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Standard model scale conversion [0.0, 1.0]
    tensor = (
        torch.from_numpy(crop_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )
    return tensor

# =========================================================
# LIVE WEBCAM LOOP
# =========================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

logger.info("Webcam started. Press ESC to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_tensor = landmarks_to_crop_tensor(frame)

    if img_tensor is None:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("DECA Mesh Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    try:
        with torch.no_grad():
            codedict = deca.encode(img_tensor)
            
            opdict, visdict = deca.decode(
                codedict,
                rendering=True,
                vis_lmk=True,
                return_vis=True,
                use_detail=True  
            )

        # Build output image grid map matrix
        vis_image = deca.visualize(visdict)
        
        if isinstance(vis_image, torch.Tensor):
            vis_image = vis_image.cpu().numpy()
        
        if vis_image.dtype != np.uint8:
            if vis_image.max() <= 1.0:
                vis_image = (vis_image * 255).astype(np.uint8)
            else:
                vis_image = vis_image.astype(np.uint8)

        # Flip channels back to match normal display output rendering colors
        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        # Scale window parameters down cleanly to fit the screen
        h_vis, w_vis = vis_bgr.shape[:2]
        scale = min(1500 / w_vis, 450 / h_vis)
        vis_bgr = cv2.resize(
            vis_bgr,
            (int(w_vis * scale), int(h_vis * scale))
        )

        cv2.imshow("DECA Mesh Pipeline", vis_bgr)

    except Exception as e:
        logger.error(f"Pipeline Rendering Error: {e}")
        cv2.imshow("DECA Mesh Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\nClosed cleanly.")