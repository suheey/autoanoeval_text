#!/bin/bash

DATASET_INFO=(
    # "4_breastw"
    # "6_cardio"
    # "7_Cardiotocography"
    # "2_annthyroid"
    # "12_fault"
    # "14_glass"
    # "15_Hepatitis"
    # "17_InternetAds"
    # "18_Ionosphere"
    # "19_landsat"
    # "20_letter"
    # "21_Lymphography"
    # "22_magic.gamma"
    # "23_mammography"
    # "24_mnist"
    # "25_musk"
    # "26_optdigits"
    # "27_PageBlocks"
    # "28_pendigits"
    # "29_Pima"
    # "30_satellite"
    # "31_satimage-2"
    # "32_shuttle"
    # "35_SpamBase"
    # "36_speech"
    # "37_Stamps"
    # "38_thyroid"
    # "39_vertebral"
    "40_vowels"
    "41_Waveform"
    "42_WBC"
    "43_WDBC"
    "44_Wilt"
    "45_wine"
    "46_WPBC"
    "47_yeast"
    "13_fraud"
    "1_ALOI"
    "3_backdoor"
    "8_celeba"
    "10_cover"
    "16_http"
    "5_campaign"
    "33_skin"
    "9_census"
    "11_donors"
    "34_smtp"
)

for dataset in "${DATASET_INFO[@]}"
do
    echo "Running dataset: $dataset"
    python main_module.py --dataset_name "$dataset"
done
