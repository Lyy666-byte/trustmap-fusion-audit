#!/bin/bash
# FMB, every step of ANALYSIS_PLAN.md in order. Safe to run again: finished steps and runs are skipped.
# Fused images -> conditions -> pretrained segmentation and metrics -> 27 trainings. Metrics run beside the GPU work, on the CPU.
TOP="/c/Users/admin/Desktop/matlab short paper"
FUS="$TOP/ir_vis_fusion"; MS="$TOP/metric_sanity"
export PYTHONIOENCODING=utf-8
cd "$MS" || exit 1
python fmb_prepare.py || { echo "STOP: prepare failed"; exit 1; }
cd "$FUS" || exit 1
for split in train test; do
  [ "$split" = train ] && { suf=_FMBtrain; n=1220; } || { suf=_FMB; n=280; }
  D="tmp_fmb/$split"
  for pair in CDDFuse:cddfuse EMMA:emma LUTFuse:lutfuse MMDRFuse:mmdrfuse DCEvo:dcevo; do
    m=${pair%%:*}; s=${pair##*:}; out="results/sota/${m}${suf}"
    if [ "$(ls "$out" 2>/dev/null | wc -l)" != "$n" ]; then
      python "sota_${s}_infer.py" "$D/ir" "$D/vi" "$out" > "$MS/results/fuse_fmb_${m}_${split}.log" 2>&1
      echo "$m $split exit $? files $(ls "$out" | wc -l) $(date)"
    else echo "$m $split already complete"; fi
  done
  out="$FUS/results/sota/SeAFusion${suf}"
  if [ "$(ls "$out" 2>/dev/null | wc -l)" != "$n" ]; then
    (cd sota/SeAFusion && python test.py -M ./model/Fusion/fusionmodel_final.pth -ir_dir "$FUS/$D/ir" -vi_dir "$FUS/$D/vi" -save_dir "$out" -j 0) > "$MS/results/fuse_fmb_SeAFusion_${split}.log" 2>&1
    echo "SeAFusion $split exit $? files $(ls "$out" | wc -l) $(date)"
  else echo "SeAFusion $split already complete"; fi
  for m in CDDFuse EMMA LUTFuse MMDRFuse DCEvo SeAFusion; do
    [ "$(ls "results/sota/${m}${suf}" | wc -l)" = "$n" ] || { echo "STOP: $m $split incomplete"; exit 2; }
  done
done
cd "$MS" || exit 1
if [ ! -f results/.fmb_stim_done ]; then
  python fmb_conditions.py > results/fmb_conditions.log 2>&1 || { echo "STOP: conditions failed, see results/fmb_conditions.log"; exit 3; }
  touch results/.fmb_stim_done
fi
echo "stimuli ready $(date), $(du -sh stim/FMB | cut -f1)"
if [ ! -f results/cond_metrics_fmb.csv ]; then
  (powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\\Users\\admin\\run_matlab.ps1" -Code "cd('C:\\Users\\admin\\Desktop\\matlab short paper\\metric_sanity'); cond_metrics_fmb(60000,6)" -TimeoutSec 62000 > results/cond_metrics_fmb.log 2>&1; echo "metrics ended $(date), csv $(ls results/cond_metrics_fmb.csv 2>/dev/null)") &
fi
for model in segformer mask2former; do
  for try in 1 2 3 4 5 6; do
    [ "$(ls results/seg_fmb/$model 2>/dev/null | wc -l)" = "75" ] && break
    .venv/Scripts/python fmb_seg_eval.py "$model" >> "results/seg_fmb_$model.log" 2>&1
    [ "$(ls results/seg_fmb/$model 2>/dev/null | wc -l)" = "75" ] || { echo "$model try $try failed $(date)"; sleep 600; }
  done
  echo "$model files $(ls results/seg_fmb/$model | wc -l)/75 $(date)"
done
for seed in 0 1 2; do
  for input in avg vi ir DCEvo EMMA LUTFuse MMDRFuse CDDFuse SeAFusion; do
    f="results/seg_retrain_fmb/${input}_s${seed}.npz"
    flag=""
    [ "$seed" = 0 ] && case "$input" in avg|vi|ir) ;; *) flag="--conds";; esac
    for try in 1 2 3 4 5 6; do
      [ -f "$f" ] && break
      .venv/Scripts/python seg_retrain_fmb.py "$input" "$seed" $flag >> "results/seg_retrain_fmb_${input}_s${seed}.log" 2>&1
      [ -f "$f" ] || { echo "$input s$seed try $try failed $(date)"; sleep 600; }
    done
    [ -f "$f" ] && echo "$input s$seed done $(date)" || echo "$input s$seed GAVE UP"
  done
done
wait
echo "QUEUE END $(date); trainings $(ls results/seg_retrain_fmb/*.npz | wc -l)/27; metrics csv $(ls results/cond_metrics_fmb.csv 2>/dev/null | wc -l)/1"
