#  Deep Perceptual Enhancement for Medical Image Analysis
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-perceptual-enhancement-for-medical-image/medical-image-enhancement-on-brain-tumor-mri)](https://paperswithcode.com/sota/medical-image-enhancement-on-brain-tumor-mri?p=deep-perceptual-enhancement-for-medical-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-perceptual-enhancement-for-medical-image/medical-image-enhancement-on-human-protein)](https://paperswithcode.com/sota/medical-image-enhancement-on-human-protein?p=deep-perceptual-enhancement-for-medical-image)

This is the official implementation of the paper titled "Deep Perceptual Enhancement for Medical Image
Analysis. **[[Click Here](https://ieeexplore.ieee.org/abstract/document/9759833)]**.

**Please consider citing this paper as follows:**

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

<p align="center">
<img width=800 align="center" src = "https://github.com/sharif-apu/DPE_JBHI/blob/main/images/overview.png" alt="Overview"> </br>
</p>

# Qualitative Comparison


<p align="center">
<img width=800 align="center" src = "https://github.com/sharif-apu/DPE_JBHI/blob/main/images/qual.png" alt="Qualitative Comparison"> </br>
</p>

# Quantitative Comparison

<p align="center">
<img width=800 align="center" src = "https://github.com/sharif-apu/DPE_JBHI/blob/main/images/quan.png" alt="Quantitative Comparison"> </br>
</p>

</br>


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

# Dataset Prepration and Training
Place all training images into a unified directory. Please refer to the original article for the reference datasets. Low-quality images will be automatically generated during training. To specify the path of your training image, go to mainModule/config.json and update "gtPath" and "targetPath" entities (both paths should be the same). 

</br> To start training please execute the following command:
```
```python main.py -ts -e X -b Y```
```

</br>
You can specify the number of epoch with **-e** flag (i.e., -e 5) and number of images per batch with **-b** flag (i.e., -b 12).</br>
execute 

# Testing


**To inference with custom setting execute the following command:**</br>
```python main.py -i -s path/to/inputImages -d path/to/outputImages``` </br>
Here, **-s** specifies the root directory of the source images (i.e., testingImages/), and **-d** specifies the destination root (i.e., modelOutput/).






**For transfer learning/ resume code execute:**</br>
```python main.py -tr -e -b ```

# Contact
For any further query, feel free to contact us through the following emails: sma.sharif.cse@ulab.edu.bd, rizwanali@sejong.ac.kr, or mithun.bishwash.cse@ulab.edu.bd





