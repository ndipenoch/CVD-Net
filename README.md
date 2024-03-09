## CVD_Net: Head and Neck Tumor Segmentation and Generalization in PET/CT Scans Across Data from Multiple Medical Centers

This is the official pytorch implementation of the CVD_Net:<br />



## Requirements
CUDA 11.0<br />
Python 3.7<br /> 
Pytorch 1.7<br />
Torchvision 0.8.2<br />

## Usage

### 0. Installation
* Install Pytorch1.7, nnUNet and CVD_Net as below
  
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

cd nnUNet
pip install -e .

cd CVD_Net_package
pip install -e .
```

### 1. Data Preparation
* Download [HECKTOR 2022](https://hecktor.grand-challenge.org/)
* Preprocess the HECKTOR 2022 dataset according to the uploaded nnUNet package.
* Training and Testing ID are in `data/splits_final.pkl`.

### 2. Training 
cd CVD_Net_package/CVD_Net/run

* Run `nohup python run_training.py -gpu='0' -outpath='CVD_Net' 2>&1 &` for training.

### 3. Testing 
* Run `nohup python run_training.py -gpu='0' -outpath='CVD_Net' -val --val_folder='validation_output' 2>&1 &` for validation.


### 4. Acknowledgements
Part of codes are reused from the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [CoTr](https://github.com/YtongXie/CoTr) . Thanks to Fabian Isensee and Yutong Xie for the codes.

### Contact
Mark Ndipenoch (markndipenoch@gmail.com)
