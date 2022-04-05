#!/bin/zsh
exp_name=$1
param_file=$2
python fromfile_apply_augment_voxforge.py ../voxforge/all_sets/ $param_file $exp_name
python create_melfs.py $exp_name/augmented_dataset $exp_name/augmented_melfs
python voxforge_K_creation.py $exp_name/augmented_melfs $exp_name/Ks
python voxforge_L_creation.py $exp_name/augmented_melfs $exp_name/Ls
python exp_testing.py $exp_name/Ks $exp_name/Ls

