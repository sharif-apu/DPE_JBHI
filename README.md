#  Deep Perceptual Enhancement for Medical Image Analysis


This is the official implementation of paper title "Deep Perceptual Enhancement for Medical Image
Analysis. **[[Click Here](https://ieeexplore.ieee.org/abstract/document/9759833)]**.

**Please consider to cite this paper as follows:**

```
@article{sharif2022deep,
  title={Deep Perceptual Enhancement for Medical Image Analysis},
  author={Sharif, SMA and Naqvi, Rizwan Ali and Biswas, Mithun and Loh, Woong-Kee},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2022},
  publisher={IEEE}
}
```

# Overview
Medical image acquisition devices are susceptible to producing low-quality (i.e., low contrast, inappropriate brightness, noise, etc.) images.  Our work aims to comprehensively enhance low-quality images by incorporating end-to-end learning strategies for accelerating medical image analysis tasks.

# Qualitative Comparison

# Quantitative Comparison

# Prerequisites
```
Python 3.8
CUDA 10.1 + CuDNN
pip
Virtual environment (optional)
```

# Installation
**Please consider using a virtual environment to continue the installation process.**
```
git clone https://github.com/sharif-apu/DPE_JBHI.git
cd DPE_JBHI
pip install -r requirement.txt
```

# Dataset Prepration

# Testing
** [[Click Here](https://drive.google.com/drive/folders/1_ziIMjK9vGg-P_7Wxit96bnfHiO4_wQw?usp=sharinge)]** to download pretrained weights and save it to weights/ directory for inferencing with Quad-bayer CFA</br>
```python main.py -i``` </br>

A few testing images are provided in a sub-directory under testingImages (i.e., testingImages/sampleImages/)</br>
In such occasion, reconstructed image(s) will be available in modelOutput/sampleImages/. </br>

**To inference with custom setting execute the following command:**</br>
```python main.py -i -s path/to/inputImages -d path/to/outputImages -ns=sigma(s)``` </br>
Here,**-ns** specifies the standard deviation of a Gaussian distribution (i.e., -ns=5, 10, 15),**-s** specifies the root directory of the source images
 (i.e., testingImages/), and **-d** specifies the destination root (i.e., modelOutput/).


# Training
To start training we need to sampling the images according to the CFA pattern and have to pair with coresponding ground-truth images.
To sample images for pair training please execute the following command:

```python main.py -ds -s /path/to/GTimages/ -d /path/to/saveSamples/ -g 2 -n 10000 ```
</br> Here **-s** flag defines your root directory of GT images, **-d** flag defines the directory where sampled images should be saved, and **-g** flag defines the binnig factr (i.e., 1 for bayer CFA, 2 for Quad-bayer), **-n** defines the number of images have to sample (optional)</br>


</br> After extracting samples, please execute the following commands to start training:

```python main.py -ts -e X -b Y```
To specify your trining images path, go to mainModule/config.json and update "gtPath" and "targetPath" entity. </br>You can specify the number of epoch with **-e** flag (i.e., -e 5) and number of images per batch with **-b** flag (i.e., -b 12).</br>


**For transfer learning execute:**</br>
```python main.py -tr -e -b ```





