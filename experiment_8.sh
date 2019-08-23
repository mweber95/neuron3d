#!/usr/bin/env bash

 python tools/process3D.py --operation split --input_dir "datasets/train-input.tif" --output_dir "datasets/" --split em
 python tools/process3D.py --operation split --input_dir "datasets/train-labels.tif" --output_dir "datasets/" --split label

python tools/process3D.py --operation tif_to_png --input_dir "datasets/em_a.tif" --output_dir "datasets/em_a"
python tools/process3D.py --operation tif_to_png --input_dir "datasets/em_b.tif" --output_dir "datasets/em_b"
python tools/process3D.py --operation tif_to_png --input_dir "datasets/em_c.tif" --output_dir "datasets/em_c"

python tools/process3D.py --operation membrane --input_dir "datasets/label_a.tif" --output_dir "datasets/membrane_a"
python tools/process3D.py --operation membrane --input_dir "datasets/label_b.tif" --output_dir "datasets/membrane_b"
python tools/process3D.py --operation membrane --input_dir "datasets/label_c.tif" --output_dir "datasets/membrane_c"

python tools/process3D.py --operation cytoplasm --input_dir "datasets/label_b.tif" --output_dir "datasets/cytoplasm_b"
python tools/process3D.py --operation cytoplasm --input_dir "datasets/label_c.tif" --output_dir "datasets/cytoplasm_c"

python tools/process3D.py --operation overlap --input_dir "datasets/label_b.tif" --output_dir "datasets/overlap_b"
python tools/process3D.py --operation overlap --input_dir "datasets/label_c.tif" --output_dir "datasets/overlap_c"

python imagetranslation-tensorflow/tools/process.py \
  --operation combine \
  --input_dir datasets/em_a \
  --b_dir datasets/membrane_a \
  --output_dir datasets/combined/

python imagetranslation-tensorflow/translate.py --mode train \
 --input_dir datasets/combined \
 --output_dir results/T3_em_membrane_a \
 --which_direction AtoB --Y_loss square --model pix2pix \
 --generator resnet --fliplr --flipud --transpose --max_epochs 2000 --display_freq 500 --gan_weight 0

rm -r datasets/combined

python imagetranslation-tensorflow/tools/process.py \
  --operation combine \
  --input_dir datasets/em_b \
  --b_dir datasets/membrane_b \
  --output_dir datasets/combined/

python imagetranslation-tensorflow/translate.py   --mode test \
  --checkpoint results/T3_em_membrane_a \
  --input_dir datasets/combined/ \
  --output_dir results/P3_em_membrane_b \
  --image_height 512  --image_width 512 \
  --model pix2pix

rm -r datasets/combined

python imagetranslation-tensorflow/tools/process.py \
  --operation combine \
  --input_dir datasets/em_c \
  --b_dir datasets/membrane_c \
  --output_dir datasets/combined/

python imagetranslation-tensorflow/translate.py   --mode test \
  --checkpoint results/T3_em_membrane_a \
  --input_dir datasets/combined/ \
  --output_dir results/P3_em_membrane_c \
  --image_height 512  --image_width 512 \
  --model pix2pix
rm -r datasets/combined

python tools/process3D.py --operation png_to_tif \
   --input_dir results/P3_em_membrane_b/images/ \
   --output_dir datasets/ \
   --name 'membrane_predicted_b'

python tools/process3D.py --operation png_to_tif \
   --input_dir results/P3_em_membrane_c/images/ \
   --output_dir datasets/ \
   --name 'membrane_predicted_c'

python tools/process3D.py --operation tif_to_png \
   --input_dir datasets/membrane_predicted_b.tif \
   --output_dir datasets/membrane_predicted_b

python tools/process3D.py --operation consecutive \
   --input_dir datasets/membrane_predicted_b.tif \
   --output_dir datasets/membrane_predicted_cons_b

python tools/process3D.py --operation tif_to_png \
   --input_dir datasets/membrane_predicted_c.tif \
   --output_dir datasets/membrane_predicted_c

python tools/process3D.py --operation consecutive \
   --input_dir datasets/membrane_predicted_c.tif \
   --output_dir datasets/membrane_predicted_cons_c

python imagetranslation-tensorflow/tools/process.py \
  --operation combine \
  --input_dir datasets/membrane_predicted_b \
  --b_dir datasets/cytoplasm_b \
  --output_dir datasets/combined/

python imagetranslation-tensorflow/translate.py --mode train \
 --input_dir datasets/combined \
 --output_dir results/T4_membrane_cytoplasm_b \
 --which_direction AtoB --Y_loss square --model pix2pix \
 --generator resnet --fliplr --flipud --transpose --max_epochs 2000 --display_freq 500 --gan_weight 0

rm -r datasets/combined

python imagetranslation-tensorflow/tools/process.py \
  --operation combine \
  --input_dir datasets/membrane_predicted_cons_b \
  --b_dir datasets/overlap_b \
  --output_dir datasets/combined/

python imagetranslation-tensorflow/translate.py --mode train \
 --input_dir datasets/combined \
 --output_dir results/T5_membrane_overlap_b \
 --which_direction AtoB --Y_loss square --model pix2pix \
 --generator resnet --fliplr --flipud --transpose --max_epochs 2000 --display_freq 500 --gan_weight 0

rm -r datasets/combined

python imagetranslation-tensorflow/tools/process.py \
  --operation combine \
  --input_dir datasets/membrane_predicted_c \
  --b_dir datasets/cytoplasm_c \
  --output_dir datasets/combined/

python imagetranslation-tensorflow/translate.py   --mode test \
  --checkpoint results/T4_membrane_cytoplasm_b \
  --input_dir datasets/combined/ \
  --output_dir results/P4_membrane_cytoplasm_c \
  --image_height 512  --image_width 512 \
  --model pix2pix

rm -r datasets/combined

python imagetranslation-tensorflow/tools/process.py \
  --operation combine \
  --input_dir datasets/membrane_predicted_cons_c \
  --b_dir datasets/overlap_c \
  --output_dir datasets/combined/

python imagetranslation-tensorflow/translate.py   --mode test \
  --checkpoint results/T5_membrane_overlap_b \
  --input_dir datasets/combined/ \
  --output_dir results/P5_membrane_overlap_c \
  --image_height 512  --image_width 512 \
  --model pix2pix

rm -r datasets/combined

python tools/process3D.py --operation png_to_tif \
   --input_dir results/P4_membrane_cytoplasm_c/images/ \
   --output_dir datasets/ \
   --name 'cytoplasm_predicted_c'

python tools/process3D.py --operation png_to_tif \
   --input_dir results/P5_membrane_overlap_c/images/ \
   --output_dir datasets/ \
   --name 'overlap_predicted_c'

python tools/process3D.py --operation tif_to_png \
   --input_dir datasets/cytoplasm_predicted_c.tif \
   --output_dir datasets/cytoplasm_predicted_c

python tools/process3D.py --operation tif_to_png \
   --input_dir datasets/overlap_predicted_c.tif \
   --output_dir datasets/overlap_predicted_c

python tools/process3D.py --operation eroskele \
   --input_dir datasets/overlap_predicted_c/ \
   --output_dir datasets/overlap_predicted_c_eroskele

python tools/reconstruct3D.py \
   --input_dir_cytoplasm datasets/cytoplasm_predicted_c/ \
   --input_dir_overlap datasets/overlap_predicted_c_eroskele/ \
   --output_dir datasets/ \
   --name experiment_8

python tools/evaluate3D.py \
   --predicted datasets/experiment_8.tif \
   --true datasets/evaluate_c.tif
