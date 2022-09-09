EgoCap and EgoFormer: First-Person Image Captioning
========

## EgoCap Dataset ##

EgoCap is a first sizable dataset that supports end-to-end egocentric image captioning. It contain 2.1K egocentric images, over 10K captions, and 6.3K contextual
label.

The EgoCap dataset can be downloaded from *public drive* if you contact z.dai1@aston.ac.uk for access. Please state your name, affiliation, and purpose of use in the email.

## EgoFormer ##

EgoFormer is a two-stream transformer based deep neural network utilizing visual-contextual attention for image caption generation in 1st-person narrative. EgoFormer accomplishes accurate and human-alike scene understanding with the aid of context encoding. You can run EgoFormer on a Jetson Nano device to let robot explore the uncharted on your behalf. A video demo can be seen **here**. A blog can be found **here**.

Please cite our paper as belew;

```
@inproceedings{egocap2022,
author = {Dai, Zhuangzhuang and Tran, Vu and Markham, Andrew and Trigoni, Niki and Rahman, M Arif and Wijayasingha, L. N. S. and Stankovic, John and Li, Chen},
year = {2022},
month = {},
pages = {},
booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
publisher={BMVA Press},
title = {EgoCap and EgoFormer: First-Person Image Captioning with Context Fusion}
}
```

## Prerequisites ##

* Python 3.7
* Pytorch 1.7
* torchvision 0.8.2
* transformers 4.12
* pycocoevalcap
* sklearn

Microsoft [COCO-2017](http://cocodataset.org/#download) dataset and EgoCap dataset are required. After acquiring these datasets locally, specify source directory links, training settings, and hyperparameters in *configuration.py*.

## Usage ##

This repository implements the training and evaluation of EgoFormer. It is modified based on repository [CATR](https://github.com/saahiluppal/catr).

Make sure you train the baseline with COCO first (configure **self.modality** as "image"). Then use the following command for context learning;

```python
python3 vit_pretrain.py  # Pre-train ViT context encoder
```

Finally use the following command to train EgoFormer with updated link to converged context ViT model;

```python
python3 main.py
```

At deployment stage, e.g., on an NVIDIA Jetson Nano, run inference model (e.g., in ONNX) by;

```python
python3 egobot_talk.py
```

## Evaluation ##

It is recommended to perform evaluation through the *Inference_and_Analysis.ipynb*. Otherwise, use **predict_qualitative()** in *Eval.py* to generate caption of an image, or conduct quantitative analysis on a directory of images.

Some EgoFormer caption results in comparison to baseline transformer are shown below

<p align="center">
  <img src="Qualitative_samples/fjDvKHkmxs0_119_126.avi00001.jpg" />
  <img src="Qualitative_samples/0ebffd92-5fa8-4dd4-9349-b45b785d720e_small.jpg" />
  <img src="Qualitative_samples/0f4e630b-e834-4ff4-9418-ccfdbdc4ee37_small.jpg" />
</p>

## Acknowledge ##

We thank the support of the National Institute of Standards and Technology (NIST) in project: Pervasive, Accurate, and Reliable Location-based Services for Emergency Responders.

We thank Professor Bongjun Choi's team of Dongseo University for helping with data validation.
