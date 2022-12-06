#!/bin/sh
#!/bin/sh

# command line:
# /bin/bash ./experiments/run_OH_conv_src.sh

root_dir="$PWD"
exp_name="OH_convT22_exp1"
dataset_name=OfficeHome
cnn_to_use=convnextTiny22
vendor_max_epoch=50

function run
{
  source=$1
  seedNum=$2

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


  vendor_exp_name=$source"_vendor_"$exp_name"_"$seedNum

#######################################
  log_file=$root_dir"/logs/"$vendor_exp_name".log"
  echo "$vendor_exp_name"

  layer=wn
  OUT_DIR="./output/weights/"$vendor_exp_name
  python $root_dir/train_source.py --gpu_id 0 --seed $seed --output  $OUT_DIR --dset $dataset_name --max_epoch $vendor_max_epoch --s $src_num --layer $layer --net $cnn_to_use
########################################

}

for ((s=0;s<3;s++))
do

    source="Art"
    run $source $s

    source="RealWorld"
    run $source $s

    source="Clipart"
    run $source $s

    source="Product"
    run $source $s

done

echo "All done"
