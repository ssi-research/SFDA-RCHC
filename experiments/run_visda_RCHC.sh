#!/bin/sh
#!/bin/sh

# command line:
# /bin/bash ./experiments/run_visda_RCHC.sh

root_dir="$PWD"
exp_name="RCHC_V_exp1"

dataset_name=VisDA-C
da_name="TtoV"
source="train"
target="validation"

dataset_exp_name=$dataset_name"/"$da_name

#############
seed=2020
client_exp_name=$da_name"_client_cs_"$exp_name"_"$seed
log_file=$root_dir"/logs/"$client_exp_name".log"
echo "Running adaptation: $client_exp_name"

# You can download source model from SHOT repository and locate under load_path directory
load_path="./output/weights/seed"$seed"/VISDA-C/T"
data_root_path="/local_datasets/da_datasets"

python $root_dir/train_adapt.py --cnn_to_use resnet101 --lr_value 1e-3 --num_C 12 --max_epoch 15 \
--ssl 0.6 --use_rot 1 --dataset_name $dataset_name --short_exp_name $exp_name --interval 15 --seed $seed \
--data_root_path $data_root_path --load_downloaded_weights 1 --load_downloaded_weights_path $load_path \
--batch_size 32 --exp_name $client_exp_name --dataset_exp_name $dataset_exp_name \
--consistency_based_PL 1 --dist_ratio_threshold 0.65 --use_dist_for_consistency 1 \
--gpu 0 --source $source --target $target  >& $log_file

