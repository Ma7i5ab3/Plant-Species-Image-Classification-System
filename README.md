# Plant Species Classification

## 1. Introduction
The objective of this challenge is to classify species of plants, which are divided into 8 categories according to their species. Given an image, the goal is to predict the correct class label from species1 to species8.

## 2. Dataset Preprocessing
The dataset consists of 3542 images of size 96x96 divided into 8 sub-folders, one for each plant species.

To test the neural network, the dataset was split into two folders:
- **Training set**: 80% of the data
- **Validation set**: 20% of the data

The splitting was done randomly to ensure a diverse dataset. However, we found that the dataset was highly imbalanced, with species1 and species6 having 40% fewer images than others. To mitigate this, we used **class weighting**, giving higher weight to underrepresented classes and lower weight to overrepresented ones.

Since we had a limited number of images, we implemented data augmentation techniques to increase diversity:
- **Data Augmentation** (flipping, rotation, and shifting using Keras ImageDataGenerator)
- **MixUp Augmentation** (mixing features and labels with a weighted combination)
- **CutOut Augmentation** (randomly masking regions of input images to encourage holistic learning)

## 3. From Simpler to More Complex Models
We used a **bottom-up approach**, starting with simpler models and progressively increasing complexity.

- Built a **CNN from scratch** with 11 convolutional layers, 6 pooling layers, and 3 dense layers (14 million parameters).
- Achieved **50-60% accuracy**, highlighting the need for more complex models.
- Explored **pre-trained models** from Keras Applications.

## 4. Models

### 4.1 Transfer Learning
We experimented with several **pre-trained models**, including:
- **VGG16**
- **ResNet50v2**
- **EfficientNetV7**
- **ConvNextLarge** (best results)

We replaced their top classification layers with custom dense layers while maintaining their image size and preprocessing functions.

### 4.2 Hyper-parameter Tuning
We automated **grid search** using **KerasTuner** with **BayesianOptimization**, finding the best parameters:
- **Activation function**: ReLU
- **Kernel initializer**: He-uniform
- **Optimizer**: RectifiedAdam (efficient training in fewer epochs)
- **Best model**: 3 dense layers (512 neurons each) with **learning rate 1e-5**
- **Adaptive learning rate**: ReduceLROnPlateau

### 4.3 Regularization
Regularization techniques used:
- **Early Stopping** (patience = 15)
- **Lasso Regularization**
- **Dropout & GaussianNoise** (for MixUp augmentation)
- **Weight Decay & Batch Normalization**

### 4.4 Classification
We tested two classifiers:
1. **Three dense layers** (512 units each) + softmax activation (loss: CategoricalCrossentropy)
2. **Hybrid model**:
   - **One dense layer (512 units)**
   - **Three dense layers (256 units each)**
   - **One dense layer (128 units)**
   - **Quasi-SVM** with **RandomFourierFeatures** (loss: Hinge Loss)

SVM **outperformed** in both accuracy and loss.

## 5. Fine-Tuning
After achieving **86% accuracy**, we performed **fine-tuning**:
- **Froze the first 240 layers** and retrained the model.
- Used **MixUp Augmentation** (alpha = 0.1) and **CutOut Augmentation**.
- Used **learning rate = 1e-6** and **ReduceLROnPlateau**.
- **Kept BatchNormalization layers frozen** to maintain learned statistics.

## 6. Ensemble
We combined the **two best models** to improve final prediction accuracy, achieving around **90% accuracy** on the validation set.

## 7. Training
- Used **cross-validation via holdout** to track accuracy and loss.
- **Prevented overfitting** by monitoring validation accuracy/loss.
- Training was performed on **local hardware** and **Google Colab**.

## 8. Results
The best model evaluations showed:
- Similar performance across top models.
- **Species1 had the lowest classification accuracy** due to dataset imbalance.
- Final model accuracy: **~90% on validation set**.

For detailed evaluation metrics, refer to images in the zip file:
- **cutout.jpeg**
- **mixup.jpeg**
- **ensemble.jpeg**
- **leaderboard_result.jpeg** (final accuracy and F1 scores on the test set)

The **test set results closely matched validation results**, proving that our model generalizes well.

---
This project demonstrates a robust **deep learning pipeline** for **plant species classification** using **CNNs, Transfer Learning, Fine-Tuning, and Ensemble Learning**.

