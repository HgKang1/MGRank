datasets=("wikihow")
std_scaling=0.1
att_weight=0.9
layers=(9)  

for dataset in "${datasets[@]}"
do
    for layer in "${layers[@]}"
    do
        echo "Running $dataset with window_size $window_size and layer $layer"
        python MGRank/main_t5.py \
            --dataset_name $dataset \
            --dataset_dir MGRank/data/$dataset \
            --att_layer $layer \
            --std_scaling $std_scaling \
            --att_weight $att_weight \
            > MGRank/${dataset}_result.log 2>&1
    done
done

