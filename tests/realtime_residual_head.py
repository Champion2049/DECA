import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import cv2
import torch
import torch.nn as nn
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
# CALIBRATION CONTROLS (Tune these to fix the mouth tilt!)
# =========================================================
USE_AXIS_ANGLE_JAW = False  # Try switching to True if the mouth still skews!
EXP_SCALE = 1.0            # Influence multiplier for expression fine-tuning (0.0 to 1.0)
POSE_SCALE = 0.3           # Influence multiplier for jaw/pose fine-tuning (Lower this if mouth twists!)

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

# Toggle jaw calculation type based on your fine-tuning environment
if USE_AXIS_ANGLE_JAW:
    deca_cfg.model.jaw_type = 'aa'
else:
    deca_cfg.model.jaw_type = 'euler' # Default native state

if hasattr(deca_cfg.model, 'extract_tex'):
    deca_cfg.model.extract_tex = False
else:
    setattr(deca_cfg.model, 'extract_tex', False)

deca_cfg.rasterizer_type = 'pytorch3d' 
deca_cfg.freeze()

# =========================================================
# INIT MODELS
# =========================================================
logger.info(f"Initializing DECA baseline on {DEVICE}...")
deca = DECA(config=deca_cfg, device=DEVICE)
deca.eval()
logger.info("DECA baseline ready.")

class CustomModel(nn.Module):
    def __init__(self, layers_list):
        super().__init__()
        self.net = nn.Sequential(*layers_list)
    def forward(self, x):
        return self.net(x)

logger.info("Scanning for fine-tuned custom tracking model...")
CUSTOM_PT_WEIGHTS = None
data_dir = "./data"

if os.path.exists(data_dir):
    for file in os.listdir(data_dir):
        if file.endswith((".pt", ".pth", ".zip")) and file != "deca_model.tar":
            CUSTOM_PT_WEIGHTS = os.path.join(data_dir, file)
            break

custom_head = None

if CUSTOM_PT_WEIGHTS is not None:
    logger.info(f"Auto-detected fine-tuned file: {CUSTOM_PT_WEIGHTS}")
    try:
        checkpoint = torch.load(CUSTOM_PT_WEIGHTS, map_location=DEVICE)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        layers = []
        if 'net.0.weight' in state_dict:
            out_features, in_features = state_dict['net.0.weight'].shape
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        if 'net.2.weight' in state_dict:
            out_features, in_features = state_dict['net.2.weight'].shape
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        if 'net.4.weight' in state_dict:
            out_features, in_features = state_dict['net.4.weight'].shape
            layers.append(nn.Linear(in_features, out_features))
            
        custom_head = CustomModel(layers).to(DEVICE)
        custom_head.load_state_dict(state_dict)
        custom_head.eval()
        logger.info("Fine-tuned residual network successfully linked and matched!")
    except Exception as e:
        logger.error(f"Could not parse custom model archive: {e}. Defaulting to baseline tracking.")
        custom_head = None
else:
    logger.warning("No valid custom weights container found. Running on standard tracking.")

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
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    kps_list = fa.get_landmarks(rgb_frame)
    
    if kps_list is None or len(kps_list) == 0:
        return None

    kps = kps_list[0]
    left, right = kps[:, 0].min(), kps[:, 0].max()
    top, bottom = kps[:, 1].min(), kps[:, 1].max()

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
            
            if custom_head is not None:
                base_expression = codedict['exp']   
                base_pose = codedict['pose']         
                
                combined_features = torch.cat([base_expression, base_pose], dim=1)
                residual_correction = custom_head(combined_features)
                
                out_dim = residual_correction.shape[1]
                if out_dim == 50:
                    codedict['exp'] = base_expression + (residual_correction * EXP_SCALE)
                elif out_dim == 56:
                    # FIX: Apply independent tracking scales to prevent jaw-wrenching distortions
                    exp_delta = residual_correction[:, :50] * EXP_SCALE
                    pose_delta = residual_correction[:, 50:] * POSE_SCALE
                    
                    codedict['exp'] = base_expression + exp_delta
                    codedict['pose'] = base_pose + pose_delta

            opdict, visdict = deca.decode(
                codedict,
                rendering=True,
                vis_lmk=True,
                return_vis=True,
                use_detail=True  
            )

        vis_image = deca.visualize(visdict)
        
        if isinstance(vis_image, torch.Tensor):
            vis_image = vis_image.cpu().numpy()
        
        if vis_image.dtype != np.uint8:
            if vis_image.max() <= 1.0:
                vis_image = (vis_image * 255).astype(np.uint8)
            else:
                vis_image = vis_image.astype(np.uint8)

        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        h_vis, w_vis = vis_bgr.shape[:2]
        scale = min(1500 / w_vis, 450 / h_vis)
        vis_bgr = cv2.resize(vis_bgr, (int(w_vis * scale), int(h_vis * scale)))

        cv2.imshow("DECA Mesh Pipeline", vis_bgr)

    except Exception as e:
        logger.error(f"Pipeline Loop Error: {e}")
        cv2.imshow("DECA Mesh Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\nClosed cleanly.")