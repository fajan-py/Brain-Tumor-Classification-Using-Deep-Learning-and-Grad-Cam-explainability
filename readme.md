

---

# Brain Tumor Classification Using Deep Learning

## Executive Summary

This project presents a deep learning-based solution for classifying brain tumors using MRI images. Using EfficientNet-B0 and PyTorch, the model has demonstrated exceptional performance in categorizing brain tumors into four distinct classes: **glioma, pituitary, meningioma, and no tumor**. The accuracy and reliability of this approach make it a powerful tool for aiding radiologists in diagnosing brain tumors, improving diagnostic efficiency and patient care.

---

## Dataset Overview

The dataset used in this project is sourced from [Masoud Nickparvar’s Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) in Kaggle. This dataset is well-curated and contains high-quality MRI images, divided into training and testing subsets:

- **Training Dataset**: Contains 80% of the images, used to train the model.
- **Testing Dataset**: Consists of 20% of the images, used to evaluate performance.
- **Classes**: The dataset includes the following tumor types:
  - **Glioma**
  - **Pituitary**
  - **Meningioma**
  - **No Tumor**

**Normalization**: Images were resized to 224x224 pixels and normalized using ImageNet standards for mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).

---

## Model Architecture

### EfficientNet-B0
EfficientNet-B0 was chosen due to its efficiency and state-of-the-art performance in image classification tasks. To optimize the model for our specific problem:

- **Final Layer Modification**: The original classification layer, designed for 1,000 classes, was replaced with a fully connected layer that outputs four classes. This modification not only aligns the model with the problem requirements but also improves inference speed significantly.
- **Loss Function**: CrossEntropyLoss was employed for multi-class classification.
- **Optimizer**: Adam optimizer with a learning rate of 1e-4 ensured stable convergence.

---

## Training and Evaluation

1. **Training Process**:
   - Number of epochs: 20
   - Batch size: 32
   - Hardware: GPU-enabled training with support for mixed-precision to improve speed and efficiency.

2. **Evaluation Metrics**:
   - **Accuracy**: 99.96%
   - **Weighted F1-Score**: 99.96%

These results highlight the model's capability to distinguish between tumor classes with remarkable precision, setting a new benchmark for reliability in automated radiological tools.

---

## Clinical Relevance

The model's high accuracy and robustness underline its potential for clinical application. Accurate and rapid classification of brain tumors can:

- **Assist Radiologists**: Serve as a second opinion for prioritizing and reviewing MRI scans.
- **Enhance Diagnostic Speed**: Reduce time to diagnosis in busy medical facilities.
- **Support Early Detection**: Improve patient outcomes by enabling timely treatment decisions.

### Real-World Example
The model correctly classified a randomly selected meningioma MRI scan during testing, showcasing its practical utility and robustness. Here are results:
![Meningioma example tumor](https://github.com/fajan-py/Brain-Tumor-Classification-Using-Deep-Learning-and-Grad-Cam-explainability/blob/main/meningioma.jpg)

![Gradcam photo](https://github.com/fajan-py/Brain-Tumor-Classification-Using-Deep-Learning-and-Grad-Cam-explainability/blob/main/meningioma%20heatmap.png)


---

## Deployment

### How to Run

**My recommendation for running this code is to use a Kaggle notebook since codes that load data are compatible with Kaggle directory. For train and evaluate codes, you can simply copy them in a cell and run them easily. Please note that you need to delete these two lines to avoid errors:**

```
from train import train_one_epoch
from evaluate import evaluate_model
```

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   Execute the training pipeline:
   ```bash
   python train.py
   ```

3. **Evaluate the Model**:
   Assess model performance:
   ```bash
   python evaluate.py
   ```

4. **Perform Inference**:
   Use the trained model to classify new MRI scans:
   ```bash
   python main.py
   ```


### Additional Setup for Grad-CAM Integration
If you wish to use Grad-CAM to visualize model predictions and enhance interpretability, set up the required repository and dependencies with the following commands:

```
!git clone https://github.com/jacobgil/pytorch-grad-cam.git
%cd pytorch-grad-cam
!pip install ttach
```
These commands clone the PyTorch Grad-CAM repository and install necessary dependencies, enabling you to generate visual explanations of the model’s predictions.



### Saved Model
The model weights are stored in `efficientnet_b0_on_nickparvar_20epochs_braintumor_dataset.pth`. This file can be loaded for future predictions without retraining.

---

## Key Results

1. **High Diagnostic Accuracy**:
   - Achieved a test accuracy of 99.96%.
   - Weighted F1-Score of 99.96%, confirming excellent performance across all classes.

2. **Robustness**:
   - Successfully predicted tumor types even on previously unseen MRI images, underscoring its reliability.

3. **Scalability**:
   - Designed for integration into real-world radiology workflows with potential for large-scale deployment.

---

## Limitations and Future Scope

### Limitations
- The model's performance is dependent on the quality and diversity of the training data.
- Additional testing on diverse datasets is needed to ensure generalizability.

### Future Directions
- Extending the approach to multi-modal imaging techniques such as CT or PET scans.


---

## Conclusion

This deep learning-based solution sets a high standard for automated brain tumor classification, achieving near-perfect accuracy. Its clinical relevance, robustness, and ease of integration into diagnostic workflows position it as a valuable tool in modern radiology.



