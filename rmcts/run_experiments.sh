num_passes=5

models=(
    inceptionv3
    resnext50
    bert
    nasneta
    squeezenet
    mobilenetv2
    nasrnn
    resnet50
    vgg
)

experiments=(
    "egg_greedy, egg_greedy"
    "egg_greedy, tensat_ilp"
    "new_greedy, new_greedy"
    "new_greedy, tensat_ilp"
    "tensat_ilp, tensat_ilp"
)

# export TASO_GPU_WARMUP=0
# set -e

for seed in $(seq 0 $(expr $num_passes - 1)); do
    for model in "${models[@]}"; do
        for experiment in "${experiments[@]}"; do 
            IFS=", ";
            set $experiment;
            echo "Running experiment:" $seed $model $1 $2;
            RUST_BACKTRACE=1 cargo run --release -- --seed $seed --model $model --extraction $1 --final_extraction $2
        done
    done
done