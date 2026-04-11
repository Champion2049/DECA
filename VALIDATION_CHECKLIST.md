# DECA Fine-Tuning Validation Checklist

Use this checklist to move from "it runs" to "it performs well".

## 1) Freeze the comparison setup

- [ ] Baseline model: original DECA checkpoint and same inference settings.
- [ ] Fine-tuned model: your FaceScape-tuned checkpoint.
- [ ] Same validation split for both models.
- [ ] Same image preprocessing and crop settings.
- [ ] Same hardware/software environment.

Record:
- Baseline checkpoint path:
- Fine-tuned checkpoint path:
- Validation list path:
- Rasterizer type:
- Date:

## 2) Functional smoke test (must pass first)

- [ ] Run inference on at least 20 mixed-pose images with baseline.
- [ ] Run inference on the same 20 images with fine-tuned model.
- [ ] Confirm output files exist for every image:
  - .obj
  - _detail.obj
  - _vis.jpg
  - _kpt2d.txt and _kpt3d.txt
- [ ] Confirm no runtime errors or missing-file failures.

## 3) Quantitative checks

Track at least these metrics on the same validation split.

- [ ] 2D landmark error (NME) mean and median.
- [ ] Optional: identity similarity score (if identity loss/features are enabled).
- [ ] Optional: depth/normal consistency proxy if available in your pipeline.
- [ ] Runtime per image (mean ms) and GPU memory usage.

### Result table template

| Metric | Baseline DECA | Fine-tuned DECA | Better? |
|---|---:|---:|---|
| NME (mean, lower is better) |  |  |  |
| NME (median, lower is better) |  |  |  |
| Identity similarity (higher is better) |  |  |  |
| Time per image (ms, lower is better) |  |  |  |
| Peak VRAM (GB, lower is better) |  |  |  |

## 4) Qualitative review (blind side-by-side)

For 30-50 validation images, compare baseline vs fine-tuned outputs without labeling which is which first.

- [ ] Pose robustness (profile, yaw, pitch)
- [ ] Mouth/lip geometry
- [ ] Eye shape and eyelid detail
- [ ] Nose bridge and tip consistency
- [ ] Cheek and jawline plausibility
- [ ] Texture/material alignment (if used)
- [ ] Failure cases documented (occlusion, blur, extreme lighting)

Score each sample 1-5 for both models, then average.

## 5) Generalization check

- [ ] Evaluate on out-of-domain images (not FaceScape-like).
- [ ] Confirm no major regression vs baseline on in-the-wild photos.
- [ ] Count severe failures for each model.

## 6) Decision rule

You can claim "better than base DECA" only if all are true:

- [ ] Fine-tuned model improves the primary metric (NME) by a meaningful margin.
- [ ] No significant regression on out-of-domain samples.
- [ ] Visual quality is equal or better in blind review.
- [ ] Runtime/memory remain acceptable for your use case.

## 7) Final report snippet

Use this in your report once numbers are filled:

"On the held-out validation split, our FaceScape fine-tuned model achieved [X]% lower NME than base DECA, with [no/minor] regression on out-of-domain samples and comparable runtime ([Y] ms/image). Therefore, under our evaluation protocol, the fine-tuned model is [better/not better] than base DECA for our target domain."
