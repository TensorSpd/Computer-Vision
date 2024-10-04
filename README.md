# [Computer Vision Nanodegree Projects](https://www.udacity.com/course/computer-vision-nanodegree--nd891?promo=school&coupon=FALL50&utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=19167921312_c_individuals&utm_term=143524481199&utm_keyword=computer%20vision%20udacity_e&utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=19167921312_c_individuals&utm_term=143524481199&utm_keyword=computer%20vision%20udacity_e&gad_source=1&gclid=EAIaIQobChMIr7umz4_0iAMVr5paBR0ACzOqEAAYASAAEgIxSfD_BwE)

## **Overview**
This repository contains projects completed as part of the Udacity Computer Vision Nanodegree. Each project focuses on applying deep learning and computer vision techniques to solve real-world problems like facial keypoint detection, image captioning, and Simultaneous Localization and Mapping (SLAM).

## **Projects**

### **1. [Facial Keypoints Detection](https://github.com/TensorSpd/Computer-Vision/tree/main/Facial_Keypoints)**
- **Dataset**: A dataset of face images, annotated with facial keypoints (68 keypoints per face).
- **Objective**: Detect facial keypoints in an image using a convolutional neural network (CNN) model.
- **Approach**:
  - Pre-process images by converting them to grayscale, normalizing, and resizing.
  - Define and train a CNN to detect key facial landmarks.
  - Use OpenCV's Haar Cascade classifier to detect faces and apply the trained CNN to predict keypoints on detected faces.
  - Visualize the detected facial keypoints and compare them with ground truth points.
- **Outcome**: Successfully trained a CNN model that can detect facial keypoints with a reasonable accuracy when applied to new face images.

### **2. [Image Captioning](https://github.com/TensorSpd/Computer-Vision/tree/main/Image_Captioning)**
- **Dataset**: COCO dataset, which contains images paired with descriptive captions.
- **Objective**: Automatically generate captions for images using a CNN-RNN architecture.
- **Approach**:
  - Use a pre-trained ResNet-50 model as the CNN encoder to extract image features.
  - Design and implement a Recurrent Neural Network (RNN) decoder (LSTM) to generate captions from the extracted image features.
  - Train the model on the COCO dataset and evaluate performance using BLEU scores.
  - Experiment with different hyperparameters such as embedding size, hidden layer size, and batch size to optimize model performance.
- **Outcome**: The CNN-RNN model successfully generates meaningful captions for new images. Further improvements were made by tuning hyperparameters and using data augmentation techniques.

### **3. [Landmark Detection and Tracking (SLAM)](https://github.com/TensorSpd/Computer-Vision/tree/main/Object_Detection_and_Tracking)**
- **Dataset**: A simulated grid world with robot motion and landmark data.
- **Objective**: Implement Simultaneous Localization and Mapping (SLAM) to estimate the robot's path and landmark positions using noisy sensor and motion data.
- **Approach**:
  - Initialize omega and xi matrices to represent constraints for robot positions and landmarks in a 2D world.
  - Implement the SLAM algorithm, which updates these matrices based on motion and sensor measurements, accounting for noise.
  - Solve for the robot's position and landmarks using matrix inversion (mu = omega^(-1) * xi).
  - Visualize the estimated robot path and landmark positions compared to the true values.
- **Outcome**: Successfully implemented a SLAM algorithm that accurately estimates the robot's path and landmark locations, with reasonable differences due to motion and measurement noise. Improvements were observed by increasing the number of sensor measurements and reducing noise.

## **Key Learnings**
Through these projects, I gained hands-on experience in several core areas of computer vision, including facial landmark detection, image caption generation, and SLAM for robotics. I developed expertise in:
- Convolutional Neural Networks (CNNs) for object detection and image feature extraction.
- Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models for sequence-based tasks such as image captioning.
- SLAM algorithms for robotics, including handling sensor noise and probabilistic mapping techniques.
- Data pre-processing, normalization, and augmentation techniques to improve model training and performance.

## **Technologies Used**
- **Languages**: Python
- **Libraries**: PyTorch, OpenCV, Matplotlib, NumPy, Pandas
- **Deep Learning Techniques**: CNNs, RNNs, LSTMs
- **Other Tools**: Jupyter Notebooks, Git, COCO API

## **Installation**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/computer-vision-projects.git
   cd computer-vision-projects
