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
deca_cfg.model.jaw_type = 'aa'
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
# HELPER: LANDMARK CROP & TENSOR CONVERSION
# =========================================================
def landmarks_to_crop_tensor(rgb_frame):
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
    crop = cv2.warpAffine(
        rgb_frame, M, (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    tensor = (
        torch.from_numpy(crop.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )
    return tensor, M

# =========================================================
# LIVE WEBCAM LOOP
# =========================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

logger.info("Webcam started. Press ESC to exit.")
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    crop_data = landmarks_to_crop_tensor(rgb)

    if crop_data is None:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Webcam Overlay", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    img_tensor, M = crop_data

    try:
        with torch.no_grad():
            # 1. Extract structural parameters from face image
            codedict = deca.encode(img_tensor)
            
            # 2. Extract raw 3D mesh points from FLAME layer
            verts, _, _ = deca.flame(
                shape_params=codedict['shape'],
                expression_params=codedict['exp'],
                pose_params=codedict['pose']
            )
            
            # 3. EXPLICIT MATH PROJECTION (Replaces the broken deca.add_camera method)
            cam = codedict['cam'].view(-1, 1, 3) # [batch, 1, 3] -> [scale, tx, ty]
            X_trans = verts[:, :, :2] + cam[:, :, 1:]
            trans_verts_2d = cam[:, :, 0:1] * X_trans

        # -------------------------------------------------
        # WINDOW 1: LIVE 2D FACE MESH OVERLAY
        # -------------------------------------------------
        verts_2d = trans_verts_2d[0].cpu().numpy()
        
        # DECA uses an inverted internal tracking viewport for the Y axis
        verts_2d[:, 1] = -verts_2d[:, 1]

        # Map from [-1, 1] normalized camera space to [0, 224] crop space
        verts_2d = (verts_2d + 1.0) * (IMAGE_SIZE / 2.0)

        # Un-project the crop space coordinates back to original webcam viewport size
        M_inv = cv2.invertAffineTransform(M)
        ones = np.ones((verts_2d.shape[0], 1))
        verts_homogeneous = np.hstack([verts_2d, ones])
        verts_original_space = np.dot(verts_homogeneous, M_inv.T)

        # Plot structural wireframe points onto live face
        for pt in verts_original_space[::4]:  # Subsample every 4th point for high FPS performance
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)

        cv2.imshow("Webcam Overlay", frame)

        # -------------------------------------------------
        # WINDOW 2: PURE 3D GEOMETRY CANVAS (SPINNING MESH)
        # -------------------------------------------------
        canvas_3d = np.zeros((400, 400, 3), dtype=np.uint8)
        verts_3d = verts[0].cpu().numpy()
        
        # Center the 3D model geometry on its internal local origin
        verts_3d -= np.mean(verts_3d, axis=0)

        # Apply smooth horizontal rotation matrix over time
        angle = frame_idx * 0.04
        c, s = np.cos(angle), np.sin(angle)
        R_y = np.array([
            [c,  0, s],
            [0,  1, 0],
            [-s, 0, c]
        ])
        rotated_verts = np.dot(verts_3d, R_y.T)

        # Scale 3D model coordinate positions to fit window canvas
        scale_factor = 1200
        proj_3d = rotated_verts[:, :2] * scale_factor
        proj_3d[:, 0] += 200
        proj_3d[:, 1] = 200 - proj_3d[:, 1]

        # Draw the tracking structure geometry
        for pt in proj_3d[::2]:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < 400 and 0 <= y < 400:
                cv2.circle(canvas_3d, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow("3D Geometric Shape", canvas_3d)

        exp_vec = codedict['exp'].squeeze().cpu().numpy()
        print(f"\rTracking Live! Expression delta range: {float(exp_vec.max() - exp_vec.min()):.4f}", end='')

    except Exception as e:
        logger.error(f"\nPipeline Error: {e}")
        cv2.imshow("Webcam Overlay", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\nClosed cleanly.")