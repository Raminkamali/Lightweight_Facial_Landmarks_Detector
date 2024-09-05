# Lightweight Facial Landmarks Detector

## Overview

**Lightweight Facial Landmarks Detector** is a compact and efficient model designed to extract 5 facial key points from images. It is built on a lightweight PyTorch mobilenet V3 small backbone and is optimized for use in face recognition tasks where landmarks are not provided by the primary detector.

## Model Architecture

- **Backbone:** PyTorch mobilenet V3 Small
- **Input Size:** 128x128 face image
- **Output:** 5 facial key points
- **TorchScript Model:** https://drive.google.com/file/d/13_VkO6LV5QrqhdGAvUA9XZ3Md0k5Tf-u/view?usp=sharing

## Dataset

- **Training Dataset:** CelebA dataset
- **Private Dataset:** Consists of 500,000 face images with ground truth landmarks generated using RetinaFace detector.
