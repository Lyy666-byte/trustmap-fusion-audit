#!/bin/bash
# Addendum 2026-11-16: the proposed network through the task test. Safe to run again: finished steps are skipped.
TOP="/c/Users/admin/Desktop/matlab short paper"; FUS="$TOP/ir_vis_fusion"; MS="$TOP/metric_sanity"
export PYTHONIOENCODING=utf-8
cd "$MS" || exit 1
for pair in MSRS361:361 MSRStrain:1083 FMB:280 FMBtrain:1220 RoadScene:40; do
  s=${pair%%:*}; n=${pair##*:}
  [ "$(ls "$FUS/results/sota/Ours_$s" 2>/dev/null | wc -l)" = "$n" ] || { echo "STOP: Ours_$s incomplete"; exit 1; }
done
python render_ours.py > results/render_ours.log 2>&1 || { echo "STOP: render failed"; exit 2; }
if [ ! -f results/cond_metrics_ours.csv ]; then
  (powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\\Users\\admin\\run_matlab.ps1" -Code "cd('C:\\Users\\admin\\Desktop\\matlab short paper\\metric_sanity'); cond_metrics_ours(6)" -TimeoutSec 7200 > results/cond_metrics_ours.log 2>&1; echo "metrics ended $(date), csv $(ls results/cond_metrics_ours.csv 2>/dev/null)") &
fi
for model in segformer mask2former; do
  for try in 1 2 3; do
    [ -f results/seg/$model/Ours_native.npz ] && break
    .venv/Scripts/python seg_eval.py "$model" Ours --native >> "results/seg_${model}_ours.log" 2>&1 || sleep 300
  done
  for try in 1 2 3; do
    [ -f results/seg_fmb/$model/Ours_native.npz ] && break
    .venv/Scripts/python fmb_seg_eval.py "$model" Ours --native >> "results/seg_fmb_${model}_ours.log" 2>&1 || sleep 300
  done
  echo "$model pretrained MSRS $(ls results/seg/$model/Ours_native.npz 2>/dev/null) FMB $(ls results/seg_fmb/$model/Ours_native.npz 2>/dev/null) $(date)"
done
for seed in 0 1 2; do
  for ds in msrs fmb; do
    if [ $ds = msrs ]; then f=results/seg_retrain/Ours_s$seed.npz; py=seg_retrain.py; log=results/seg_retrain_Ours_s$seed.log
    else f=results/seg_retrain_fmb/Ours_s$seed.npz; py=seg_retrain_fmb.py; log=results/seg_retrain_fmb_Ours_s$seed.log; fi
    for try in 1 2 3 4; do
      [ -f "$f" ] && break
      .venv/Scripts/python $py Ours $seed >> "$log" 2>&1
      [ -f "$f" ] || { echo "Ours $ds s$seed try $try failed $(date)"; sleep 600; }
    done
    [ -f "$f" ] && echo "Ours $ds s$seed done $(date)" || echo "Ours $ds s$seed GAVE UP"
  done
done
wait
echo "QUEUE END $(date); retrain msrs $(ls results/seg_retrain/Ours_s*.npz 2>/dev/null | wc -l)/3 fmb $(ls results/seg_retrain_fmb/Ours_s*.npz 2>/dev/null | wc -l)/3; metrics $(ls results/cond_metrics_ours.csv 2>/dev/null | wc -l)/1"
