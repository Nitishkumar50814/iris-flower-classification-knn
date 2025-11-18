# iris-flower-classification-knn
Machine learning project to classify Iris species using KNN.
# Iris Flower Classification using K-Nearest Neighbors (KNN)
A simple and beginner-friendly machine learning project built using the **Iris dataset**.  
This project uses the **K-Nearest Neighbors (KNN)** algorithm to classify Iris flowers into three species.

---

##  Project Objective
Predict the species of an Iris flower based on four input features:
- Sepal length  
- Sepal width  
- Petal length  
- Petal width  

Model used → **KNN Classifier (k = 3)**

---

## 📂 Dataset Information
The Iris dataset contains:
- **150 samples**
- **4 features**
- **3 target classes:**  
  - 0 = Setosa  
  - 1 = Versicolor  
  - 2 = Virginica

---

# 🛠 Technologies Used
- Python
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Scikit-Learn
##  Machine Learning Workflow
### ✔ Import Libraries
```python
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
