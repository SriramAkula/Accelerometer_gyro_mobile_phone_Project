# Human Activity Recognition (HAR) using Machine Learning

## Overview
This project focuses on developing and evaluating machine learning classification models for **Human Activity Recognition (HAR)** using mobile phone sensor data. The dataset used in this study is obtained from the **UCI Machine Learning Repository** and consists of accelerometer and gyroscope readings recorded during different human activities.

## Dataset
- **Source**: UCI Machine Learning Repository
- **Sensors Used**: Accelerometer, Gyroscope
- **Recorded Axes**: X, Y, Z
- **Data Attributes**: Timestamps, Activity Labels

## Methodology
### 1. Data Preprocessing
- Handling missing values
- Feature scaling and normalization
- Encoding categorical variables
- Exploratory Data Analysis (EDA) to understand data distribution

### 2. Model Training
The following machine learning classification models were implemented:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**

### 3. Hyperparameter Tuning
To enhance model performance, **Grid Search** and **Randomized Search** were used to optimize hyperparameters such as:
- Regularization strength
- Number of neighbors (for KNN)
- Tree depth (for Decision Tree and Random Forest)

### 4. Model Evaluation
The models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

## Results
| Model | Accuracy |
|--------|------------|
| Logistic Regression | 87.74% |
| Decision Tree | 98.15% |
| Random Forest | 98.87% |
| K-Nearest Neighbors (KNN) | 98.99% |

- **KNN performed the best**, achieving an accuracy of **98.99%**.

## Future Scope
1. **Mobile Application Integration**: Implementing a mobile app for real-time HAR.
2. **Feature Expansion**: Incorporating biometric and environmental data for better classification.
3. **Advanced Activity Detection**: Recognizing additional human activities and anomalies.
4. **Healthcare Applications**: Using HAR for elderly care, rehabilitation monitoring, and fitness tracking.

## References
1. AlSahly, Abdullah. (2022). *Accelerometer Gyro Mobile Phone Dataset*, UCI Machine Learning Repository.
2. Bulling, Andreas; Blanke, Ulf; Schiele, Bernt. *A tutorial on human activity recognition using body-worn inertial sensors*. ACM Computing Surveys, 2014.
3. Kaghyan, Sahak, and Hakob Sarukhanyan. *Activity recognition using K-nearest neighbor algorithm on smartphones*. IJIMA, 2012.
4. Praveen Kumar Shukla et al. *Human activity recognition using accelerometer and gyroscope data from smartphones*. ICONC3, IEEE, 2020.
5. Andrea Mannini, Angelo Maria Sabatini. *Machine learning methods for classifying human physical activity from on-body accelerometers*. Sensors, 2010.

## Contact
**Author:** Akula Hema Venkata Sriram  
**Email:** sriramakula212@gmail.com  
**Institution:** Lovely Professional University, Jalandhar, India

---
This project demonstrates the **power of machine learning** in recognizing human activities based on **sensor data**. Feel free to contribute or reach out for collaboration!

