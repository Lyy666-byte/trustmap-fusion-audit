#!/bin/bash
# Six methods, seed 0, weights kept, tested on native and all transformed versions. Safe to run again: finished runs are skipped.
MS="/c/Users/admin/Desktop/matlab short paper/metric_sanity"
export PYTHONIOENCODING=utf-8
cd "$MS" || exit 1
for input in DCEvo EMMA LUTFuse MMDRFuse CDDFuse SeAFusion; do
  f="results/seg_retrain_cond/${input}_s0.npz"
  for try in 1 2 3 4 5 6; do
    [ -f "$f" ] && break
    .venv/Scripts/python seg_retrain.py "$input" 0 --conds >> "results/seg_retrain_cond_${input}_s0.log" 2>&1
    [ -f "$f" ] || { echo "$input try $try failed $(date)"; sleep 600; }
  done
  [ -f "$f" ] && echo "$input done $(date)" || echo "$input GAVE UP"
done
echo "QUEUE END $(date); files $(ls results/seg_retrain_cond/*.npz | wc -l)/6"
