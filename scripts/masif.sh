#!/bin/bash
python prepare_masif.py
cd ./masif/data/masif_site
python cal_dataset.py
cd ../../../
python calculate_masif.py
