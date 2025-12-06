## Usage

### Requirements
* Python 3.9
* PyTorch 
* Torchvision 

Install the require dependencies:
```bash
conda create -n hdr_transformer_pytorch python=3.9
conda activate hdr_transformer_pytorch
pip install -r requirements.txt
```
### Dataset
1. Download the dataset (include the training set and test set) from [Kalantari17's dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
2. Move the dataset to `./data` and reorganize the directories as follows:
```
./data/Training
|--001
|  |--262A0898.tif
|  |--262A0899.tif
|  |--262A0900.tif
|  |--exposure.txt
|  |--HDRImg.hdr
|--002
...
./data/Test (include 15 scenes from `EXTRA` and `PAPER`)
|--001
|  |--262A2615.tif
|  |--262A2616.tif
|  |--262A2617.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
|--BarbequeDay
|  |--262A2943.tif
|  |--262A2944.tif
|  |--262A2945.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
```
3. Prepare the data with degradation:
```
cd ./dataset
python synthesize.py
```
4. Prepare the corpped training set by running:
```
cd ./dataset
python gen_crop_data.py
```

### Training & Evaluaton

To train the model, run:
```
python train.py
```
To evaluate, run:
```
python fullimagetest.py 
```

## Acknowledgement
Our work is inspired the following works and uses parts of their official implementations:

* [Restormer](https://github.com/swz30/Restormer)
* [CA-VTI](https://github.com/megvii-research/HDR-Transformer)
* [SCTNet](https://steven-tel.github.io/sctnet)

We thank the respective authors for open sourcing their methods.

## Contact
If you have any questions, feel free to contact Zhou Gong at zhougong@mail.nwpu.edu.cn.
