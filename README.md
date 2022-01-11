# Aurora-Prediction

# Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed.

# Quick start

## Installation
1. Clone this repo:
```python
git clone 
```
2. Install dependencies:
```python
pip install requirements.txt
```
3. Delete readme files in all the output directories (data, logs, models, results)
Your directory tree should look like this:
```
Aurora-Prediction
├── data
├── example
├── lib
├── logs
├── models
├── results
├── tools 
├── README.md
└── requirements.txt
```
## Data preparation
For training and testing, we use npz files instead of oringinal jpg images. In tools/utils.py, we provide a function to transform images to npz files. 
You can also download all the npz files here.
## Training
```python
cd tools
python train.py --input <input frames> --out <output frame>
```
## Testing
```python
cd tools
python test.py --input <input frames> --model_no <list of model index>
```
## Test on a single case
We provide a jupyter notebook in example/. You should put test images in example/true_imgs first, including input 5 frames and true output 10 frames. The predicted images will be saved in example/pred_imgs.
