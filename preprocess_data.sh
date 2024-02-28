export CUDA_VISIBLE_DEVICES=0
# DATASET=brac
# DATA_DIRECTORY=/data/smb/syh/WSI_cls/TCGA_BRCA/img/
# RESULTS_DIRECTORY=/data/smb/syh/WSI_cls/TCGA_BRCA/mask/

ROOT=/data/smb/syh/WSI_cls/
# DATASET=cam16
DATASET=brac
#DATASET=lung


# DATA_DIRECTORY=/data/smb/syh/WSI_cls/camelyon16/img
DATA_DIRECTORY=$ROOT/$DATASET/img
# RESULTS_DIRECTORY=/data/smb/syh/WSI_cls/camelyon16/mask/
RESULTS_DIRECTORY=$ROOT/$DATASET/mask
# CLAM_DIRECTORY=/data/smb/syh/WSI_cls/CLAM
CLAM_DIRECTORY=$ROOT/CLAM
CURRENT_DIRECTORY=`pwd`
# CKPT_PATH=/data/smb/syh/WSI_cls/vit256_small_dino.pth
CKPT_PATH=$ROOT/vit256_small_dino.pth



#download dataset
# python -u preprocess/download_tcga_wsis.py --dataset $DATASET

#seg tissue
cd $CLAM_DIRECTORY
# python -u create_patches_fp.py --source $DATA_DIRECTORY --save_dir $RESULTS_DIRECTORY  --preset tcga.csv --seg
# python -u create_patches_fp.py --source $DATA_DIRECTORY --save_dir $RESULTS_DIRECTORY  --preset bwh_biopsy.csv --seg
# python create_patches_fp.py --source $DATA_DIRECTORY --save_dir $RESULTS_DIRECTORY --patch_size 512 --step_size 512 --preset bwh_biopsy.csv --seg
# python -u create_patches_fp.py --source $DATA_DIRECTORY --save_dir $RESULTS_DIRECTORY  --seg
cd $CURRENT_DIRECTORY

# convert mask to json
# python -u preprocess/mask2json.py --dataset $DATASET
#
## extract patch into lmdb
# python -u preprocess/create_patch_lmdb.py --dataset $DATASET
#
## extract patch into feat
# python -u preprocess/create_feat_lmdb.py --dataset $DATASET
python -u preprocess/create_feat1.py --dataset $DATASET --ckpt  $CKPT_PATH
#  --ckpt $CKPT_PATH
