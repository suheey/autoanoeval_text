#!/bin/bash

DATASET_INFO=(
    # "7_Cardiotocography"
    # "4_breastw"
    # "6_cardio"
    # "15_Hepatitis"
    "21_Lymphography"
    # "39_vertebral"
    # "42_WBC"
    # "45_wine"
    # "47_yeast"

    "3_backdoor"
    # "5_campaign"
    # "9_census"
    # "10_cover"
)

for dataset in "${DATASET_INFO[@]}"
do
    echo "Running dataset: $dataset"
    python main_text.py --dataset_name "$dataset"
done
