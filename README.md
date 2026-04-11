# DECA: Detailed Expression Capture and Animation (SIGGRAPH2021)

<p align="center"> 
<img src="TestSamples/teaser/results/teaser.gif">
</p>
<p align="center">input image, aligned reconstruction, animation with various poses & expressions<p align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YadiraF/DECA/blob/master/Detailed_Expression_Capture_and_Animation.ipynb?authuser=1)

This is the official Pytorch implementation of DECA. 

DECA reconstructs a 3D head model with detailed facial geometry from a single input image. The resulting 3D head model can be easily animated. Please refer to the [arXiv paper](https://arxiv.org/abs/2012.04012) for more details.

The main features:

* **Reconstruction:** produces head pose, shape, detailed face geometry, and lighting information from a single image.
* **Animation:** animate the face with realistic wrinkle deformations.
* **Robustness:** tested on facial images in unconstrained conditions.  Our method is robust to various poses, illuminations and occlusions. 
* **Accurate:** state-of-the-art 3D face shape reconstruction on the [NoW Challenge](https://ringnet.is.tue.mpg.de/challenge) benchmark dataset.
  
## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/YadiraF/DECA
  cd DECA
  ```  

### Requirements
* Python 3.7 (numpy, skimage, scipy, opencv)  
* PyTorch >= 1.6 (pytorch3d)  
* face-alignment (Optional for detecting face)  
  You can run 
  ```bash
  pip install -r requirements.txt
  ```
  Or use virtual environment by runing 
  ```bash
  bash install_conda.sh
  ```
  For visualization, we use our rasterizer that uses pytorch JIT Compiling Extensions. If there occurs a compiling error, you can install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) instead and set --rasterizer_type=pytorch3d when running the demos.

### Usage
1. Prepare data   
    run script: 
    ```bash
    bash fetch_data.sh
    ```
    <!-- or manually download data form [FLAME 2020 model](https://flame.is.tue.mpg.de/download.php) and [DECA trained model](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view?usp=sharing), and put them in ./data  -->  
    (Optional for Albedo)   
    follow the instructions for the [Albedo model](https://github.com/TimoBolkart/BFM_to_FLAME) to get 'FLAME_albedo_from_BFM.npz', put it into ./data

2. Run demos  
    a. **reconstruction**  
    ```bash
    python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True
    ```   
    to visualize the predicted 2D landmanks, 3D landmarks (red means non-visible points), coarse geometry, detailed geometry, and depth.   
    <p align="center">   
    <img src="Doc/images/id04657-PPHljWCZ53c-000565_inputs_inputs_vis.jpg">
    </p>  
    <p align="center">   
    <img src="Doc/images/IMG_0392_inputs_vis.jpg">
    </p>  
    You can also generate an obj file (which can be opened with Meshlab) that includes extracted texture from the input image.  

    Please run `python demos/demo_reconstruct.py --help` for more details. 

    **Where to see the 3D result**

    The reconstruction command writes a per-image folder under your save path.
    For example, with:
    ```bash
    python demos/demo_reconstruct.py -i TestSamples/examples/image02673.png -s demo_output_image02673 --iscrop false --saveObj true --saveKpt true --saveDepth true
    ```
    you will get:
    - `demo_output_image02673/image02673/image02673.obj`: coarse 3D mesh
    - `demo_output_image02673/image02673/image02673_detail.obj`: detailed 3D mesh
    - `demo_output_image02673/image02673/image02673.mtl` and `.png`: texture/material files
    - `demo_output_image02673/image02673_vis.jpg`: rendered visualization image

    Open the `.obj` file in a 3D viewer such as MeshLab or Blender to inspect/rotate the reconstructed 3D face.

    b. **expression transfer**   
    ```bash
    python demos/demo_transfer.py
    ```   
    Given an image, you can reconstruct its 3D face, then animate it by tranfering expressions from other images. 
    Using Meshlab to open the detailed mesh obj file, you can see something like that:
    <p align="center"> 
    <img src="Doc/images/soubhik.gif">
    </p>  
    (Thank Soubhik for allowing me to use his face ^_^)   
    
    Note that, you need to set '--useTex True' to get full texture.   

    c. for the [teaser gif](https://github.com/YadiraF/DECA/results/teaser.gif) (**reposing** and **animation**)
    ```bash
    python demos/demo_teaser.py 
    ``` 
    
    More demos and training code coming soon.

## Evaluation
DECA (ours) achieves 9% lower mean shape reconstruction error on the [NoW Challenge](https://ringnet.is.tue.mpg.de/challenge) dataset compared to the previous state-of-the-art method.  
The left figure compares the cumulative error of our approach and other recent methods (RingNet and Deng et al. have nearly identitical performance, so their curves overlap each other). Here we use point-to-surface distance as the error metric, following the NoW Challenge.  
<p align="left"> 
<img src="Doc/images/DECA_evaluation_github.png">
</p>

For more details of the evaluation, please check our [arXiv paper](https://arxiv.org/abs/2012.04012). 

For project-level model validation (baseline vs fine-tuned), see [VALIDATION_CHECKLIST.md](VALIDATION_CHECKLIST.md).

### Baseline vs Fine-Tuned Comparison (FaceScape)

Use the comparison script to evaluate baseline and fine-tuned checkpoints on the same validation list and export failure diagnostics:

```bash
python scripts/compare_models.py \
  --baseline ./data/deca_model.tar \
  --finetuned ./logs/facescape_compare_model_a/model.tar \
  --val_list ../val_list_wsl.txt \
  --max_samples 100 \
  --device cuda \
  --failure_threshold 2.0 \
  --top_k_failures 20 \
  --top_k_examples 20 \
  --out ./logs/model_compare_report.txt \
  --debug_dir ./logs/model_compare_debug
```

Outputs:
- `./logs/model_compare_report.txt`: aggregate comparison metrics.
- `./logs/model_compare_debug/all_samples.csv`: per-image metrics with source image paths.
- `./logs/model_compare_debug/all_failures.csv`: all failed samples (`finetuned_lmk_err > failure_threshold`).
- `./logs/model_compare_debug/all_successes.csv`: all successful samples (`finetuned_lmk_err <= failure_threshold`).
- `./logs/model_compare_debug/worst_failures.csv`: highest-error fine-tuned failures.
- `./logs/model_compare_debug/worst_failures/*.jpg`: side-by-side baseline vs fine-tuned renders for fast visual debugging.
- `./logs/model_compare_debug/sanity_examples.csv`: best successful fine-tuned examples.
- `./logs/model_compare_debug/sanity_examples/*.jpg`: side-by-side baseline vs fine-tuned good-case renders.

### Real Landmark Generation (Recommended Before Any Accuracy Claim)

If `.npy` landmarks are missing, the dataset falls back to a mean-face template and `LandmarkError*` becomes a proxy metric.
Generate real 68-point landmarks first:

```bash
python scripts/generate_landmarks_68.py \
  --list ../val_list_wsl.txt \
  --device cuda \
  --skip_existing \
  --report ./logs/landmark_generation_report_val_full.csv \
  --failed_list ./logs/landmark_generation_failed_val_full.txt
```

Outputs:
- `./logs/landmark_generation_report_val_full.csv`: per-image generation status (`generated`, `skipped_existing`, `detect_failed`).
- `./logs/landmark_generation_failed_val_full.txt`: explicit list of images where face detection failed.

Then run comparison on this same split:

```bash
python scripts/compare_models.py \
  --baseline ./data/deca_model.tar \
  --finetuned ./logs/facescape_compare_model_a_robust/model.tar \
  --val_list ../val_list_wsl.txt \
  --max_samples 400 \
  --device cuda \
  --failure_threshold 2.0 \
  --top_k_failures 30 \
  --top_k_examples 30 \
  --hybrid_metric lmk_abs_max \
  --hybrid_cam_thresholds 1.2,1.4,1.5,1.6,1.8,2.0,2.5,3.0 \
  --out ./logs/model_compare_report_real_lmk_FINAL.txt \
  --debug_dir ./logs/model_compare_debug_real_lmk_FINAL
```

The report now includes `RealLandmarkSamples` and `TemplateLandmarkSamples` so metric reliability is explicit.

### Lip-Focused Retraining + Hybrid Evaluation (WSL)

To improve lip-region behavior while keeping fallback safety, run the end-to-end pipeline:

```bash
bash scripts/wsl2_retrain_lip_hybrid.sh
```

What this pipeline does:
- generates missing real 68-point `.npy` landmarks for both train and val lists,
- retrains from baseline with stronger lip/landmark stability losses,
- evaluates baseline vs new checkpoint with hybrid threshold sweep,
- exports a fresh geometry mismatch audit.

Main outputs:
- `./logs/facescape_compare_model_a_lip_recover_v3/model.tar`
- `./logs/model_compare_report_lip_recover_v3.txt`
- `./logs/model_compare_debug_lip_recover_v3/*`
- `./logs/geometry_audit_lip_recover_v3.csv`
- `./logs/geometry_audit_lip_recover_v3/*`
- `./logs/lip_recover_promotion_decision.txt` (explicit promote/reject decision based on failure-rate + trimmed-mean)

Current recommended hybrid setting for this run family is a strict fallback threshold around `finetuned_lmk_abs_max > 0.6`, which kept failure rate at zero while preserving baseline-level stability.
Only promote this checkpoint if `model_compare_report_lip_recover_v3.txt` shows lower real-landmark failure rate and lower trimmed mean than baseline.

### Checkpoint Progression (Real-Landmark Eval)

Root-cause finding from recent debugging:
- With `K=1` and single-image mode, the original FaceScape loader sampled by grouped expression buckets, which reduced effective epoch size from ~19k images to ~400 groups.
- This caused unstable fine-tuning dynamics and made catastrophic outliers much more likely.
- The loader has now been fixed so single-image training iterates all images directly, and trainer now honors `max_steps` for controlled runs.

Recent fine-tuned checkpoints tested on `val_list_wsl.txt` (real landmarks available for almost all samples):

| Checkpoint | Report | Fine-tuned Mean | Fine-tuned Trimmed Mean | Fine-tuned Failure Rate (>2.0) | Promotion |
|---|---|---:|---:|---:|---|
| v2 | `./logs/model_compare_report_lip_recover_v2.txt` | 0.8963 | 0.1527 | 0.0226 | Not promoted (baseline still better) |
| v3 | `./logs/model_compare_report_lip_recover_v3.txt` | 9.1526 | 0.7769 | 0.0952 | Rejected |
| v4 | `./logs/model_compare_report_lip_recover_v4.txt` | 21.5664 | 2.6866 | 0.1153 | Rejected |
| v5 | `./logs/model_compare_report_lip_recover_v5.txt` | 5.2519 | 0.2267 | 0.0627 | Rejected |

Current best fine-tuned checkpoint among this sequence is **v2** (`./logs/facescape_compare_model_a_lip_recover_v2/model.tar`), but baseline still wins on real-landmark acceptance metrics.
Use hybrid fallback for deployment safety and continue improving only if the new checkpoint beats baseline on both:
- failure rate,
- trimmed mean landmark error.

Additional full-coverage experiments after loader fix:

| Checkpoint | Report | Fine-tuned Mean | Fine-tuned Trimmed Mean | Fine-tuned Failure Rate (>2.0) | Notes |
|---|---|---:|---:|---:|---|
| v7 (full20k) | `./logs/model_compare_report_lip_recover_v7_full20k.txt` | 2.9895 | 0.2131 | 0.0500 | Large reduction of catastrophic mean vs unstable runs |
| v8 (continue v7) | `./logs/model_compare_report_lip_recover_v8_full20k_continue.txt` | 8.5125 | 1.0307 | 0.1000 | Regressed |
| v9 (v2-style full20k) | `./logs/model_compare_report_lip_recover_v9_v2style_full20k.txt` | 50.4012 | 4.2067 | 0.1200 | Regressed |

Takeaway:
- The loader/max-step fixes are necessary and now in place.
- They significantly reduced the worst failure behavior in v7, but baseline still remains stronger on strict landmark metrics.

### Git Hygiene for Generated Landmarks

Generated landmark `.npy` files and DECA logs are local training artifacts and should not be pushed.
This repository now ignores them by default from the repo root `.gitignore`.

If any were already staged/tracked previously, untrack once:

```bash
git rm -r --cached facescape_224/*.npy DECA/logs
```

Then commit the `.gitignore` change.

### Visual Geometry Audit (Not Proxy-LMK Based)

To prioritize visually wrong fine-tuned outputs (even when proxy landmarks look acceptable), run:

```bash
python scripts/audit_geometry_mismatch.py \
  --baseline ./data/deca_model.tar \
  --finetuned ./logs/facescape_compare_model_a_robust/model.tar \
  --val_list ../val_list_wsl.txt \
  --max_samples 400 \
  --device cuda \
  --top_k 40 \
  --out_csv ./logs/geometry_audit_real_lmk_FINAL.csv \
  --debug_dir ./logs/geometry_audit_real_lmk_FINAL
```

Outputs:
- `./logs/geometry_audit_real_lmk_FINAL.csv`: ranked mismatch list (worst first).
- `./logs/geometry_audit_real_lmk_FINAL/*.jpg`: side-by-side baseline vs fine-tuned renders for the highest-mismatch cases.

### Hybrid Model (Completed by @Champion2049)

The hybrid model integration was completed by **@Champion2049**.

Basic idea:
- Run the fine-tuned model by default because it often gives lower proxy landmark error on template-supervised splits.
- Detect unstable predictions using a simple confidence proxy (`finetuned_lmk_abs_max`).
- If the proxy is above a chosen threshold, fallback to the base DECA prediction for that sample.

Why this helps:
- Fine-tuned DECA can improve median proxy metric on many normal inputs.
- Baseline DECA is more stable on catastrophic cases.
- The hybrid gate combines both strengths by suppressing unstable fine-tuned outliers.

Final benchmark summary:
- See `./logs/model_compare_report_FINAL.txt` for the final base-vs-hybrid comparison.
- With the selected threshold, the hybrid setup achieved zero proxy failure-rate on the benchmark split while preserving speed.
- Important: for the current `val_list_wsl.txt` split, `RealLandmarkSamples=0` and `TemplateLandmarkSamples=399`, so LandmarkError values are proxy/template-alignment numbers, not true geometry-accuracy ground truth.

Final benchmark snapshot (399 samples, from `./logs/model_compare_report_FINAL.txt`):

| Metric | Baseline | Fine-tuned | Hybrid (best threshold) |
|---|---:|---:|---:|
| Landmark mean (proxy) | 0.472537 | 80.368398 | 0.219577 |
| Landmark trimmed mean (proxy) | 0.474826 | 1.740729 | 0.208470 |
| Landmark median (proxy) | 0.481752 | 0.197433 | 0.197433 |
| Landmark p90 (proxy) | 0.559752 | 1.808978 | 0.288978 |
| Failure rate (>2.0, proxy) | 0.000000 | 0.100251 | 0.000000 |
| Runtime mean (ms) | 38.879 | 35.474 | 35.942 |

Interpretation:
- Fine-tuned can produce visually wrong outliers even when proxy landmark error looks low for some samples.
- Baseline remains the more stable geometry reference in difficult cases.
- Hybrid is the safer default because it auto-falls back to baseline for detected outliers.
- Do not claim geometry accuracy gains from this table unless real landmark files (`.npy`) exist for the evaluated split.

Hybrid testing commands:

1) Run the final benchmark (base vs fine-tuned + hybrid threshold search)

```bash
python scripts/compare_models.py \
  --baseline ./data/deca_model.tar \
  --finetuned ./logs/facescape_compare_model_a_robust/model.tar \
  --val_list ../val_list_wsl.txt \
  --max_samples 400 \
  --device cuda \
  --failure_threshold 2.0 \
  --top_k_failures 30 \
  --top_k_examples 30 \
  --hybrid_metric lmk_abs_max \
  --hybrid_cam_thresholds 1.2,1.4,1.5,1.6,1.8,2.0,2.5,3.0 \
  --out ./logs/model_compare_report_FINAL.txt \
  --debug_dir ./logs/model_compare_debug_FINAL
```

Automatic hybrid-render mode:
- The script now always exports `./logs/model_compare_debug_FINAL/hybrid_auto_failures/*.jpg`.
- Each image is a 3-panel view: Baseline | Fine-tuned | Hybrid(auto).
- For outliers, Hybrid(auto) immediately shows baseline fallback instead of the blank/unstable fine-tuned render.
- Threshold selection is automatic from `--hybrid_cam_thresholds` (best failure-rate then trimmed-mean). You can force one with `--hybrid_auto_threshold`.

2) Test with one given picture (example: image02673)

```bash
echo /mnt/c/Users/Chirayu/Documents/GitHub/2D-to-3D-image-reconstruction/TestSamples/examples/image02673.png > ../single_image_test_wsl.txt

python scripts/compare_models.py \
  --baseline ./data/deca_model.tar \
  --finetuned ./logs/facescape_compare_model_a_robust/model.tar \
  --val_list ../single_image_test_wsl.txt \
  --max_samples 1 \
  --device cuda \
  --failure_threshold 2.0 \
  --top_k_failures 1 \
  --top_k_examples 1 \
  --hybrid_metric lmk_abs_max \
  --hybrid_cam_thresholds 1.2,1.5,2.0 \
  --out ./logs/model_compare_report_single_image.txt \
  --debug_dir ./logs/model_compare_debug_single_image
```

Single-image outputs to check:
- `./logs/model_compare_report_single_image.txt`
- `./logs/model_compare_debug_single_image/all_samples.csv`
- `./logs/model_compare_debug_single_image/all_failures.csv` (all failed samples)
- `./logs/model_compare_debug_single_image/all_successes.csv` (all successful samples)
- `./logs/model_compare_debug_single_image/worst_failures/*.jpg` (baseline vs fine-tuned failure renders)
- `./logs/model_compare_debug_single_image/sanity_examples/*.jpg` (best successful renders)
- `./logs/model_compare_debug_single_image/hybrid_auto_failures/*.jpg` (baseline | fine-tuned | hybrid auto-fallback for outliers)

Where all failures and all good successes are stored:
- All failure rows: `./logs/model_compare_debug_FINAL/all_failures.csv`
- All success rows: `./logs/model_compare_debug_FINAL/all_successes.csv`
- Worst failure renders: `./logs/model_compare_debug_FINAL/worst_failures/*.jpg`
- Good success renders: `./logs/model_compare_debug_FINAL/sanity_examples/*.jpg`
- Auto-hybrid fallback renders for failure/outlier inspection: `./logs/model_compare_debug_FINAL/hybrid_auto_failures/*.jpg`

### Hard-case mining retraining loop

Use the exported `worst_failures.csv` to oversample difficult cases during the next fine-tuning run:

```bash
python main_train.py --cfg configs/release_version/deca_facescape_compare_model_a_hardcase.yml
```

This config enables:
- `dataset.hard_case_csv`: CSV from the comparison step.
- `dataset.hard_case_boost`: higher sampling weight for groups containing failed images.

After training, run `scripts/compare_models.py` again to verify failure-rate reduction.

## Training
1. Prepare Training Data

    a. Download image data  
    In DECA, we use [VGGFace2](https://arxiv.org/pdf/1710.08092.pdf), [BUPT-Balancedface](http://www.whdeng.cn/RFW/Trainingdataste.html) and [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)  

    b. Prepare label  
    [FAN](https://github.com/1adrianb/2D-and-3D-face-alignment) to predict 68 2D landmark  
    [face_segmentation](https://github.com/YuvalNirkin/face_segmentation) to get skin mask  

    c. Modify dataloader   
    Dataloaders for different datasets are in decalib/datasets, use the right path for prepared images and labels. 

2. Download face recognition trained model  
    We use the model from [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) for calculating identity loss,
    download [resnet50_ft](https://drive.google.com/file/d/1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU/view),
    and put it into ./data  

3. Start training

    Train from scratch: 
    ```bash
    python main_train.py --cfg configs/release_version/deca_pretrain.yml 
    python main_train.py --cfg configs/release_version/deca_coarse.yml 
    python main_train.py --cfg configs/release_version/deca_detail.yml 
    ```
    In the yml files, write the right path for 'output_dir' and 'pretrained_modelpath'.  
    You can also use [released model](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view) as pretrained model, then ignor the pretrain step.

## Related works:  
* for better emotion prediction: [EMOCA](https://github.com/radekd91/emoca)  
* for better skin estimation: [TRUST](https://github.com/HavenFeng/TRUST)

## Citation
If you find our work useful to your research, please consider citing:
```
@inproceedings{DECA:Siggraph2021,
  title={Learning an Animatable Detailed {3D} Face Model from In-The-Wild Images},
  author={Feng, Yao and Feng, Haiwen and Black, Michael J. and Bolkart, Timo},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH)}, 
  volume = {40}, 
  number = {8}, 
  year = {2021}, 
  url = {https://doi.org/10.1145/3450626.3459936} 
}
```

<!-- ## Notes
1. Training code will also be released in the future. -->

## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](https://github.com/YadiraF/DECA/blob/master/LICENSE) file.
By downloading and using the code and model you agree to the terms in the [LICENSE](https://github.com/YadiraF/DECA/blob/master/LICENSE). 

## Acknowledgements
For functions or scripts that are based on external sources, we acknowledge the origin individually in each file.  
Here are some great resources we benefit:  
- [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch) and [TF_FLAME](https://github.com/TimoBolkart/TF_FLAME) for the FLAME model  
- [Pytorch3D](https://pytorch3d.org/), [neural_renderer](https://github.com/daniilidis-group/neural_renderer), [SoftRas](https://github.com/ShichenLiu/SoftRas) for rendering  
- [kornia](https://github.com/kornia/kornia) for image/rotation processing  
- [face-alignment](https://github.com/1adrianb/face-alignment) for cropping   
- [FAN](https://github.com/1adrianb/2D-and-3D-face-alignment) for landmark detection
- [face_segmentation](https://github.com/YuvalNirkin/face_segmentation) for skin mask
- [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) for identity loss  

We would also like to thank other recent public 3D face reconstruction works that allow us to easily perform quantitative and qualitative comparisons :)  
[RingNet](https://github.com/soubhiksanyal/RingNet), 
[Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/renderer/rasterize_triangles.py), 
[Nonlinear_Face_3DMM](https://github.com/tranluan/Nonlinear_Face_3DMM),
[3DDFA-v2](https://github.com/cleardusk/3DDFA_V2),
[extreme_3d_faces](https://github.com/anhttran/extreme_3d_faces),
[facescape](https://github.com/zhuhao-nju/facescape)
<!-- 3DMMasSTN, DenseReg, 3dmm_cnn, vrn, pix2vertex -->
