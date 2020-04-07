#!/bin/bash

# conda create -y -n rsna2 python=3.6
# conda activate rsna2

# conda install -y -n rsna pytorch=0.4.1 cuda90 -c pytorch
#pip install torch==0.4.1
#pip install torchvision==0.2.2
#pip install --upgrade pip
#pip install -r requirements.txt
#pip install pycocotools
#pip install pretrainedmodels
#pip install imgaug
pip install tqdm==4.19.9
cd src/pytorch_retinanet/lib
. build.sh
cd ../../..


