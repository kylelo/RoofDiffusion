#!/bin/bash

python="/home/kylelo/anaconda3/envs/RoofDiffusion/bin/python"

base_src_dir="./dataset/PoznanRD/benchmark/wo_footprint/"
base_out_dir="./experiments/PoznanRD_wo_FP"
base_out_dir="$(readlink -f "$base_out_dir")"

si_array=("95 30" "95 60" "99 30" "99 60")

resume_state="./pretrained/wo_footprint/140"

for pair in "${si_array[@]}"; do
    read -r s i <<< "$pair"

    out_dir="${base_out_dir}/s${s}_i${i}"

    mkdir -p $out_dir

    ${python} run.py -p test -c config/roof_completion_no_footprint.json -o ${out_dir} -rs ${resume_state} --n_timestep 500 \
                    --data_root ${base_src_dir}/s${s}_i${i}/img.flist \
                    --footprint_root ${base_src_dir}/s${s}_i${i}/footprint.flist \
                    --scale_factor 1
done

for pair in "${si_array[@]}"; do
    read -r s i <<< "$pair"

    echo $pair

    ${python} data/util/roof_metric.py \
                --gt_dir ${base_src_dir}/s${s}_i${i}/roof_gt \
                --footprint_dir ${base_src_dir}/s${s}_i${i}/roof_footprint \
                --pred_dir ${base_out_dir}/s${s}_i${i}/results/test/0 \
                --img_name_prefix "BID"

done