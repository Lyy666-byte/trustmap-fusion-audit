#!/bin/bash
# Fused MSRS train images for the six methods, a check that the pipeline reproduces the test images, then all 27 trainings.
# Safe to run again: finished methods and runs are skipped.
TOP="/c/Users/admin/Desktop/matlab short paper"
FUS="$TOP/ir_vis_fusion"; MS="$TOP/metric_sanity"; TR="tmp_msrs/train"
export PYTHONIOENCODING=utf-8
cd "$FUS" || exit 1
for pair in CDDFuse:cddfuse EMMA:emma LUTFuse:lutfuse MMDRFuse:mmdrfuse DCEvo:dcevo; do
  m=${pair%%:*}; s=${pair##*:}; out="results/sota/${m}_MSRStrain"
  if [ "$(ls "$out" 2>/dev/null | wc -l)" != "1083" ]; then
    python "sota_${s}_infer.py" "$TR/ir" "$TR/vi" "$out" > "$MS/results/fuse_train_$m.log" 2>&1
    echo "$m exit $? files $(ls "$out" | wc -l)"
  else echo "$m already complete"; fi
done
out="$FUS/results/sota/SeAFusion_MSRStrain"
if [ "$(ls "$out" 2>/dev/null | wc -l)" != "1083" ]; then
  (cd sota/SeAFusion && python test.py -M ./model/Fusion/fusionmodel_final.pth -ir_dir "$FUS/$TR/ir" -vi_dir "$FUS/$TR/vi" -save_dir "$out" -j 0) > "$MS/results/fuse_train_SeAFusion.log" 2>&1
  echo "SeAFusion exit $? files $(ls "$out" | wc -l)"
else echo "SeAFusion already complete"; fi
for m in CDDFuse EMMA LUTFuse MMDRFuse DCEvo SeAFusion; do
  [ "$(ls "results/sota/${m}_MSRStrain" | wc -l)" = "1083" ] || { echo "STOP: $m incomplete"; exit 2; }
done
python "$MS/check_train_fusion.py" || { echo "STOP: reproduction check failed"; exit 3; }
touch "$MS/results/.fused_train_done"; echo "fused train images ready $(date)"
cd "$MS" || exit 1
for seed in 0 1 2; do
  for input in avg vi ir DCEvo EMMA LUTFuse MMDRFuse CDDFuse SeAFusion; do
    f="results/seg_retrain/${input}_s${seed}.npz"
    for try in 1 2 3 4 5 6; do
      [ -f "$f" ] && break
      .venv/Scripts/python seg_retrain.py "$input" "$seed" >> "results/seg_retrain_${input}_s${seed}.log" 2>&1
      [ -f "$f" ] || { echo "$input s$seed try $try failed $(date)"; sleep 600; }
    done
    [ -f "$f" ] && echo "$input s$seed done $(date)" || echo "$input s$seed GAVE UP"
  done
done
echo "QUEUE END $(date); files $(ls results/seg_retrain/*.npz | wc -l)/27"
