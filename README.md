\# Road Sign and Pedestrian Classification Under Varying Weather Conditions



A machine learning system for classifying pedestrians and traffic signs under varying weather conditions using the BDD100K dataset.



\## Project Overview



This project implements a machine learning pipeline involving:



\- BDD100K dataset analysis

\- Object extraction and dataset balancing

\- Image cropping using bounding-box annotations

\- Grayscale image preprocessing

\- Feature standardization

\- PCA-based dimensionality reduction

\- MLP classification

\- Stochastic Gradient Descent optimization

\- Cross-validation

\- ROC and Precision-Recall analysis

\- Accuracy, Precision, Recall and F1-score evaluation



\## Current Progress



\### PCA Preprocessing



The selected object images are resized to 32 × 32 grayscale images.



Original feature size:



32 × 32 = 1024 features



PCA reduces the feature space to 57 components while retaining approximately 95% of the variance.



\## Dataset



The project uses the BDD100K dataset.



The dataset contains images captured under different weather and environmental conditions.



Target classes:



\- Pedestrian

\- Traffic Sign



\## Pipeline



BDD100K  

↓  

Object Annotation Extraction  

↓  

Balanced Dataset Creation  

↓  

Object Cropping  

↓  

32 × 32 Grayscale Images  

↓  

1024 Pixel Features  

↓  

Standardization  

↓  

PCA  

↓  

57 Features  

↓  

MLP + SGD  

↓  

Classification and Evaluation



\## Project Status



\- \[x] Dataset analysis

\- \[x] Object extraction

\- \[x] Balanced dataset creation

\- \[x] Image downloading

\- \[x] Object cropping

\- \[x] PCA preprocessing

\- \[x] Explained variance analysis

\- \[ ] MLP implementation

\- \[ ] SGD optimization

\- \[ ] Cross-validation

\- \[ ] ROC and Precision-Recall evaluation

\- \[ ] Final model evaluation

