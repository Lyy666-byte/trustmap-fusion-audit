# TrustMap-Fusion-Audit

Code, trained weights and result files for the manuscript

> Y. Liu, W. Zhang, Y. Zhao, S. Yang, P. Wang. *A Lightweight Difference-Driven Trust-Map Network for Infrared and
> Visible Image Fusion and an Intervention-Based Audit of Fusion Metrics.* Submitted to *Electronics* (MDPI), 2026.
> School of Physical Science and Technology and Cuiying Honors College, Lanzhou University.

The repository has two parts that follow the two halves of the paper:

| Folder | Content | Paper |
|---|---|---|
| `ir_vis_fusion/` | the dual-branch fusion network with the difference-driven trust map (MATLAB R2023b): architecture, unsupervised loss, training, metric implementation, comparison, ablation, trust-map analysis, training seeds, efficiency, blending study; inference and timing scripts for the released baselines (Python) | Sections 3.1–3.3, 4, 5.1 |
| `metric_sanity/` | the intervention-based metric audit and the segmentation evaluation (Python + MATLAB): transformations, metrics of transformed images, rank attacks, pretrained and retrained SegFormer-B5 / Mask2Former, and the analysis plan `ANALYSIS_PLAN.md` | Sections 3.4, 5 |
| `figures/` | Figures 1–7 of the manuscript | |
| `make_tables.py`, `tables/` | regenerates Tables 1–8 from the result files | |

## Tables in one command

```bash
python make_tables.py
```

reads only the result files of this repository (Python standard library, no data or GPU needed) and writes
`tables/Table1.csv` … `tables/Table8.csv` and `tables/tables.md`; the bold entries of the paper are marked there.
The output is included and agrees cell by cell with the tables of the manuscript.

## The fusion network

```matlab
cd ir_vis_fusion
S = load('results/fusion_net_v3.mat');                       % the reported model (A3, training seed 0)
F = fuse_images(S.net, imread('ir.png'), imread('vi.png'));  % fused luminance, double in [0,1]
```

`fuse_images.m` converts both inputs to grayscale, runs the network (on the GPU if available) and min–max scales
the output. For colour display the fused luminance is recombined with the chrominance of the visible image.
The network has 43,356 learnable parameters (`build_fusion_net_v7.m`).

| File in `ir_vis_fusion/results/` | Model in the paper |
|---|---|
| `fusion_net_v3.mat` | full model A3, training seed 0 (Tables 1–5, all reported results of "ours") |
| `model_V7s1.mat` … `model_V7s4.mat` | A3, training seeds 1–4 (Table 5) |
| `model_AB0.mat`, `model_AB1.mat`, `model_AB2.mat` | ablation A0 (single stream), A1 (dual, no trust map), A2 (trust from concatenation) (Table 3) |
| `model_AB1s1.mat` … `model_AB1s4.mat` | A1, training seeds 1–4 (Table 5) |
| `fusion_net_v4_OLD_backup.mat` | single-stream predecessor with CBAM (Introduction, Section 4.2) |
| `densefuse.mat` | our MATLAB re-implementation of DenseFuse (Tables 1, 2, 4) |

Training (MSRS, first 200 training pairs, 3000 patches of 128×128, 40 epochs in chunks of 20; call until it prints
`DONE`):

```matlab
train_resume('V7','build_fusion_net_v7','fusion_loss_v5',40,200,20,20)            % A3, seed 0 (model_V7.mat = fusion_net_v3.mat)
train_resume('V7s1','build_fusion_net_v7','fusion_loss_v5',40,200,20,20,1)         % A3, seed 1 (2, 3, 4 alike)
train_resume('AB1','build_abl','fusion_loss_v5',40,200,20,'A1_dual_cat')           % A1 (A0: 'A0_single', A2: 'A2_cat_trust')
train_resume('AB1s1','build_abl','fusion_loss_v5',40,200,20,'A1_dual_cat',1)       % A1, seed 1 (2, 3, 4 alike)
```

The loss weights (λ_g = 20, λ_s = 3, λ_tv = 0.5) are set in `fusion_loss_v5.m`.

## Data (not included)

| Dataset | Source | Expected location |
|---|---|---|
| MSRS | https://github.com/Linfeng-Tang/MSRS | `ir_vis_fusion/tmp_msrs/{train,test}/{ir,vi,Segmentation_labels}` |
| TNO (the 20 pairs used by DenseFuse) | https://github.com/hli1221/imagefusion_densefuse (from the TNO Image Fusion Dataset, https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) | `ir_vis_fusion/data/TNO/{ir,vi}` |
| FMB | https://github.com/JinyuanLiu-CV/SegMiF (`train.zip`, `test.zip`) | `fmb_raw/`, extracted by `metric_sanity/fmb_prepare.py` to `ir_vis_fusion/tmp_fmb/` |

Exact file lists of every evaluation subset, with the SHA-256 of the decoded pixels, are in
`ir_vis_fusion/data_manifest/` (MSRS 30 test pairs = every twelfth of the 361 test pairs, MSRS 200 training pairs,
TNO 20 pairs; check with `python make_manifests.py --check` in `ir_vis_fusion/`). The 100-image audit subsets of
MSRS and FMB (random draw with seed 20261031) are listed in `metric_sanity/results/image_sets.json` and
`image_sets_fmb.json`.

Released baselines (code and weights from their authors, placed under `ir_vis_fusion/sota/`): SeAFusion, CDDFuse,
EMMA, DCEvo, MMDRFuse, LUT-Fuse. Their fused images were produced with `ir_vis_fusion/sota_*_infer.py` and, for
SeAFusion, with its own `test.py`; the exact calls are in `metric_sanity/retrain_queue.sh` and `fmb_queue.sh`.
Segmentation models (Hugging Face, placed under `metric_sanity/models/`): `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`
and `facebook/mask2former-swin-small-cityscapes-semantic`.

## Where each table and figure comes from

All paths below are relative to `ir_vis_fusion/` (network) or `metric_sanity/` (audit).

| Paper | Result files | Computed by |
|---|---|---|
| Tables 1, 2 | `results/metrics_sota_{MSRS,TNO}.csv` (means), `results/signif_{MSRS,TNO}.csv` (Wilcoxon), `results/peritem_{MSRS,TNO}.mat` (per image) | `eval_all.m`, `append_method.m`, `finalize_tag.m`, `refresh_ours.m`; metrics in `metric_full.m` |
| Table 3 | `results/ablation2_{MSRS,signif,trustdev,params}.csv` | `eval_ablation2.m`, `test_constant_map_predecessor.m` |
| Section 4.2, Fig. 4 | `results/diag_maps_msrs30.csv`, `results/semantic_maps_*.csv`, `results/constant_w_test.csv`, `results/constant_map_test_predecessor.csv`, `results/maps/*_maps.mat` | `diag_maps_msrs30.m`, `exp_semantic_maps.m`, `test_constant_w.m`, `test_constant_map_predecessor.m`, `export_maps.m` → `gen_interp_fig2.py` |
| Table 4 | `results/efficiency_timing.csv`, `results/efficiency_flops_mem.csv` | `measure_efficiency.m`, `count_flops.m`, `measure_sota_params.py`, `measure_sota_flops.py`, `measure_light_efficiency.py`, `sota_timing.py` |
| Table 5, Section 4.4 | `results/seeds5_metrics.csv`, `results/seed_pairs.csv`, `results/seeds5_*.csv` | `eval_seeds.m`, `eval_seeds_xdata.m`, `seedq.m`, `seeds5.m`, `seed_pairs.m`, `seeds5_pairs.m` |
| Section 5.1, Fig. 5 | `results/blend_frontier.csv`, `results/blend_matched.csv` | `blend_frontier.m`, `blend_matched.m`, `summarize_blend_matched.py` → `gen_blend_fig3.py` |
| Section 6 (8-bit vs. floating point) | `results/check_quant_ours.csv` | `check_quant_ours.m` |
| Table 6 | `results/cause6_metrics.csv` (MSRS), `results/fmb_metrics.csv` (FMB), per image in `results/cond_metrics*.csv` | `make_conditions.py`, `fmb_conditions.py`, `cond_metrics.m`, `cond_metrics_ext.m`, `cond_metrics_fmb.m`, `cause_analyze6.py`, `fmb_analyze.py` |
| Table 7, Fig. 6 | `results/rank_attack.csv`, `results/rank_summary.csv`, `results/retrain_cond_rank.csv`, `results/fmb_rank_attack.csv`, `results/fmb_rank_summary.csv` | `rank_attack.py`, `retrain_cond_analyze.py`, `fmb_analyze.py` → `gen_audit_example2.py` |
| Fig. 7 | `results/seg_holm.csv`, `results/fmb_seg_holm.csv`, `results/retrain_cond_pooled.csv`, `results/fmb_retrain_cond.csv` | `seg_eval.py`, `fmb_seg_eval.py`, `seg_retrain.py --conds`, `seg_retrain_fmb.py --conds`, `rank_attack.py`, `retrain_cond_analyze.py`, `fmb_analyze.py` → `ir_vis_fusion/gen_audit_forest.py` |
| Table 8, Section 5.3 | `results/retrain_scores.csv`, `results/retrain_vs_avg.csv`, `results/fmb_retrain_scores.csv`, `results/fmb_retrain_vs.csv`, `results/ours_retrain.csv` | `seg_retrain.py`, `seg_retrain_fmb.py`, `retrain_analyze.py`, `fmb_analyze.py`, `ours_analyze.py` |
| Figs. 2, 3 | — (images) | `export_roi.m`, `export_light.py`, `export_light2.py` → `gen_compare_fig3.py MSRS qual_MSRS_06_all.png 0.30 0.33 0.22 0.42` and `gen_compare_fig3.py TNO qual_TNO_all.png 0.26 0.44 0.20 0.38` |
| Fig. 1 | — (diagram) | `gen_arch_fig2.py` |

Per-image segmentation results (intersection, predicted and ground-truth pixel counts per image and class) are in
`metric_sanity/results/{seg,seg_fmb,seg_retrain,seg_retrain_cond,seg_retrain_fmb}/*.npz`. The order in which the
long runs were executed, with their arguments, is recorded in `metric_sanity/*_queue.sh`; the absolute paths in
these scripts refer to the machine on which they were run.

## Notes

* **Metric implementation.** `metric_full.m` computes the thirteen metrics. Two errors of widely used evaluation
  scripts are corrected: the mutual-information routine returned twice the entropy of the fused image
  (`analysis_MI.m` now uses a 256×256 joint histogram; check: MI(X, X) = H(X)), and the routine labelled MS-SSIM
  computes MEF-SSIM; the result files keep the column name `MS_SSIM`, the paper calls it MEF-SSIM.
* **8-bit scoring.** All fused images, ours included, are quantized to 8 bits before scoring.
* **Transformed images** of the audit are not included (about 10 GB); `metric_sanity/make_conditions.py` and
  `fmb_conditions.py` regenerate them exactly (the noise seed is derived from the image name).
* **Not included:** the datasets, third-party code and weights, fused images of the compared methods, and the
  weights of the retrained segmentation models (about 320 MB each).
* **RoadScene.** Some result files (for example `seeds5_xdata.csv`, `blend_frontier.csv`, `blend_matched.csv`,
  `rank_summary.csv`, `cause6_metrics.csv`) and scripts also contain RoadScene entries from an earlier version of
  the study. Those RoadScene pairs were not registered (uncropped infrared frames paired with cropped visible
  images); the entries are not used in the paper and should not be used.
* **Analysis plan.** `metric_sanity/ANALYSIS_PLAN.md` is kept as written, with its dated addenda. The human
  preference study it describes was prepared but not run; no human data exist and none are reported.

## License

MIT (see `LICENSE`). The datasets, baselines and segmentation models remain under the licenses of their authors.
