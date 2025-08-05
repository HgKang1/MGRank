datasets=("SemEval2017")
std_scaling=0.1
att_weight=0.9
layers=(9)  # 여러 layer 값

for dataset in "${datasets[@]}"
do
    for layer in "${layers[@]}"
    do
        echo "Running $dataset with window_size $window_size and layer $layer"
        python PromptRank_cross_att_gemma2/main_ori.py \
            --dataset_name $dataset \
            --dataset_dir PromptRank_cross_att_gemma2/data/$dataset \
            --att_layer $layer \
            --std_scaling $std_scaling \
            --att_weight $att_weight \
            > PromptRank_cross_att_gemma2/data/logs/${window_size}_${layer}_${dataset}_0.3_.log 2>&1
    done
done

