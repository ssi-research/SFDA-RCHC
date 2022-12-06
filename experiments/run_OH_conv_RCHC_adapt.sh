#!/bin/sh
#!/bin/sh

# command line:
# /bin/bash ./experiments/run_OH_conv_RCHC_adapt.sh

root_dir="$PWD"
exp_name="RCHC_OH_convT22_exp1"
dataset_name=OfficeHome
cnn_to_use=convnextTiny22
vendor_exp="OH_convT22_exp1"


function run
{
  da_name=$1
  source=$2
  target=$3
  seedNum=$4
  vendor_exp_num=$5

  if [ $seedNum -eq 0 ]; then
    seed=2021
  elif [ $seedNum -eq 1 ]; then
    seed=2020
  else [ $seedNum -eq 2 ]
    seed=2019
  fi

#   names = ['Art', 'Clipart', 'Product', 'RealWorld']

  if [ "$source" = "Art" ]; then
    src_num=0
  elif [ "$source" = "Clipart" ]; then
    src_num=1
  elif [ "$source" = "Product" ]; then
    src_num=2
  elif [ "$source" = "RealWorld" ]; then
    src_num=3
  fi

  if [ "$target" = "Art" ]; then
    target_num=0
  elif [ "$target" = "Clipart" ]; then
    target_num=1
  elif [ "$target" = "Product" ]; then
    target_num=2
  elif [ "$target" = "RealWorld" ]; then
    target_num=3
  fi

  dataset_exp_name=$dataset_name"/"$da_name
  client_exp_name=$da_name"_client_cs_"$exp_name"_"$seedNum
  log_file=$root_dir"/logs/"$client_exp_name".log"

  vendor_exp_name=$source"_vendor_"$vendor_exp"_"$seedNum

########################################
  echo "Running $client_exp_name"

  data_root_path="/local_datasets/da_datasets"

  python $root_dir/train_adapt.py --cnn_to_use $cnn_to_use \
  --ssl 0.6 --use_rot 1 --short_exp_name $exp_name --interval 15 --seed $seed --data_root_path $data_root_path \
  --lr_value 1e-2 --num_C 65 --max_epoch 15 --dataset_name $dataset_name --load_downloaded_weights 0 \
  --batch_size 64 --exp_name $client_exp_name --load_exp_name $vendor_exp_name \
  --consistency_based_PL 1 --dist_ratio_threshold 0.65 --use_dist_for_consistency 1 \
  --gpu 0 --dataset_exp_name $dataset_exp_name --source $source --target $target >& $log_file

}

for ((s=0;s<1;s++))
do

    da_name="RtoA"
    source="RealWorld"
    target="Art"
    run $da_name $source $target $s

    da_name="AtoP"
    source="Art"
    target="Product"
    run $da_name $source $target $s

    da_name="AtoR"
    source="Art"
    target="RealWorld"
    run $da_name $source $target $s

    da_name="CtoA"
    source="Clipart"
    target="Art"
    run $da_name $source $target $s

    da_name="AtoC"
    source="Art"
    target="Clipart"
    run $da_name $source $target $s

    da_name="PtoA"
    source="Product"
    target="Art"
    run $da_name $source $target $s

    da_name="CtoR"
    source="Clipart"
    target="RealWorld"
    run $da_name $source $target $s

    da_name="RtoC"
    source="RealWorld"
    target="Clipart"
    run $da_name $source $target $s

    da_name="PtoR"
    source="Product"
    target="RealWorld"
    run $da_name $source $target $s

    da_name="RtoP"
    source="RealWorld"
    target="Product"
    run $da_name $source $target $s

    da_name="PtoC"
    source="Product"
    target="Clipart"
    run $da_name $source $target $s

    da_name="CtoP"
    source="Clipart"
    target="Product"
    run $da_name $source $target $s

done

echo "All done"
