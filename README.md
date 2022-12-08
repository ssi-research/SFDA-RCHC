# Reconciling a Centroid-Hypothesis Conflict in Source-Free Domain Adaptation

----
> Idit Diamant,  Roy H. Jennings, Oranit Dror, Hai Victor Habi, Arnon Netzer <br/> Sony Semiconductor Israel

This repository contains the official implementation of the paper: Reconciling a Centroid-Hypothesis Conflict in Source-Free Domain Adaptation [[arXiv]](https://arxiv.org/pdf/2212.03795.pdf). 

### Abstract

Source-free domain adaptation (SFDA) aims to transfer knowledge learned from a source domain to an unlabeled target domain, where the source data is unavailable during adaptation. Existing approaches for SFDA focus on self-training usually including well-established entropy minimization techniques. One of the main challenges in SFDA is to reduce accumulation of errors caused by domain misalignment. A recent strategy successfully managed to reduce error accumulation by pseudo-labeling the target samples based on class-wise prototypes (centroids) generated by their clustering in the representation space. However, this strategy also creates cases for which the cross-entropy of a pseudo-label and the minimum entropy have a conflict in their objectives. We call this conflict the centroid-hypothesis conflict. We propose to reconcile this conflict by aligning the entropy minimization objective with that of the pseudo labels' cross entropy. We demonstrate the effectiveness of aligning the two loss objectives on three domain adaptation datasets. 
In addition, we provide state-of-the-art results using up-to-date architectures also showing the consistency of our method across these architectures. 


### Prerequisites
  - pip install timm==0.4.12 
  - python==3.8
  - pytorch==1.10.0=py3.8_cuda11.3_cudnn8.2.0_0 
  - cudatoolkit=11.3
  - torchvision==0.11.0
  - numpy
  - scipy
  - scikit-learn
  - pillow
  - tqdm
  - natsort
  - six
  - opencv
  - scikit-image

### Datasets

- Please manually download the datasets Office-Home, VisDA-C and DomainNet (mini)

### Training

- Script examples are under experiments directory  

#### Source model training


    python train_source.py --output ./output/weights/Art_vendor_OH_2020 --dset OfficeHome --data_root_path <data_path> --max_epoch 50 --s 0 --layer wn --net convnextTiny22


#### Target adaptation


    python train_adapt.py --cnn_to_use convnextTiny22 --short_exp_name RCHC_OH --data_root_path <data_path> --dataset_exp_name OfficeHome/AtoP --lr_value 1e-2 --num_C 65 --dataset_name OfficeHome  --exp_name AtoP_client_RCHC_OH_2020 --load_exp_name Art_vendor_OH_2020 --consistency_based_PL 1 --dist_ratio_threshold 0.65 --use_dist_for_consistency 1 --source Art --target Product
  

#### MixMatch


    python train_mixmatch.py --source $source --target $target --ps 0.0 --ssl 0.6 --cls_par 0.3 --s 0 --t 1 --data_root_path <data_path> --output_tar  ./output/weights/$client_exp_name --output  ./output/weights/$client_exp_name$postfix  --dset $dataset_name --max_epoch $mm_epoch --net $cnn_to_use

