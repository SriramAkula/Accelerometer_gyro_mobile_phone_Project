# %%
# NAME :- AKULA HEMA VENKATA SRIRAM
# ROLL NO :- 04
# REGISTRATION NO :- 12210461
# SECTION :- K22BW
# COURSE CODE :- INT-354

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# PROJECT TITLE :- Analyzing Machine Learning Models for Human Activity Recognition: A Comparative Study
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# %% [markdown]
# Reading The Dataset

# %%
import pandas as pd
data=pd.read_csv('accelerometer_gyro_mobile_phone_dataset.csv')
data.head()

# %% [markdown]
# Data Exploration and Preprocessing

# %%
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check the shape of the dataset
print("\nShape of the dataset:")
print(data.shape)

# %% [markdown]
# Check For Missing Values

# %%
print("Missing values:")
print(data.isnull().sum())

# %%
# Check unique values in the 'activity' column
print("Unique activities:")
print(data['Activity'].unique())

# %%
# Summary statistics of numerical columns
print("Summary statistics:")
print(data.describe())


# %% [markdown]
# Scaling

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Scaling numerical features
# We'll scale the accelerometer and gyroscope signals using StandardScaler
scaler = StandardScaler()
numerical_cols = [
    'accX', 'accY', 'accZ', 
    'gyroX', 'gyroY', 'gyroZ'
]

data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Display the first few rows of the updated dataset with new features and preprocessed data
print("Updated dataset after preprocessing:")
print(data.head())


# %% [markdown]
# Check if the data is imbalanced

# %%
print(100*data['Activity'].value_counts()/len(data['Activity']))
print(data['Activity'].value_counts())

# %% [markdown]
# Plotting Heatmap

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Drop the 'timestamp' column
data = data.drop(columns=['timestamp'])

# Compute the correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# %%
df=data.drop(columns=['gyroX','gyroY'])
df.head()

# %% [markdown]
# Model Training & Evaluation

# %% [markdown]
# Splitting data and target

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Split the data into features (X) and target (y)
X = df.drop(columns=['Activity'])  # Features
y = df['Activity']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define the Random Forest Classifier with default hyperparameters
rf_classifier = RandomForestClassifier(random_state=42)  # Set random_state for reproducibility

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


# %% [markdown]
# OverSampling

# %%
from imblearn.combine import SMOTEENN
sm=SMOTEENN()
X_resampled,Y_resampled=sm.fit_resample(X,y)
Xr_train,Xr_test,Yr_train,Yr_test=train_test_split(X_resampled,Y_resampled,test_size=0.2,random_state=42)
print(y.value_counts())
print("\nAfter Resampling")
print(Y_resampled.value_counts())

# %% [markdown]
# WITHOUT HYPERPARAMETER TUNING

# %% [markdown]
# Random Forest Classifier

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define the Random Forest Classifier with default hyperparameters
rf_classifier = RandomForestClassifier(random_state=42)  # Set random_state for reproducibility

# Train the model on the training data
rf_classifier.fit(Xr_train, Yr_train)

# Make predictions on the testing data
Y_pred_rf = rf_classifier.predict(Xr_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(Yr_test, Y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(Yr_test, Y_pred_rf))


# %% [markdown]
# Decision Tree Classifier

# %%
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Define pipeline with Decision Tree Classifier (default hyperparameters)
pipeline = Pipeline([
  ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train the pipeline on the training data
pipeline.fit(Xr_train, Yr_train)

# Make predictions on the testing data
Y_pred_dt = pipeline.predict(Xr_test)

# Evaluate the model's performance
print("\nClassification Report:")
print(classification_report(Yr_test, Y_pred_dt))

print("\nConfusion Matrix:")
print(confusion_matrix(Yr_test, Y_pred_dt))


# %% [markdown]
# Logistic Regression

# %%
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Define pipeline with Logistic Regression (default hyperparameters)
pipeline = Pipeline([
  ('classifier', LogisticRegression(random_state=42))
])

# Train the pipeline on the training data
pipeline.fit(Xr_train, Yr_train)

# Make predictions on the testing data
Y_pred_lr = pipeline.predict(Xr_test)

# Evaluate the performance of the Logistic Regression model
print("\nClassification Report for Logistic Regression:")
print(classification_report(Yr_test, Y_pred_lr))

print("\nConfusion Matrix for Logistic Regression:")
print(confusion_matrix(Yr_test, Y_pred_lr))


# %% [markdown]
# K Neighbors Classifier

# %%
from sklearn.neighbors import KNeighborsClassifier

# Define the pipeline with KNeighborsClassifier (default hyperparameters)
pipeline = Pipeline([
    ('classifier', KNeighborsClassifier())
])

# Train the pipeline on the training data
pipeline.fit(Xr_train, Yr_train)

# Make predictions on the testing data
Y_pred_knn = pipeline.predict(Xr_test)

# Evaluate the model's performance
print("\nClassification Report:")
print(classification_report(Yr_test, Y_pred_knn))

print("\nConfusion Matrix:")
print(confusion_matrix(Yr_test, Y_pred_knn))


# %% [markdown]
# WITH HYPERPArAMETER TUNING - GRIDSEARCH

# %% [markdown]
# Decision tree classifier

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Define pipeline with Decision Tree Classifier
pipeline = Pipeline([
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Define hyperparameters grid for grid search
param_grid = {
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(Xr_train, Yr_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Predict on the testing set using the best model
best_classifier = grid_search.best_estimator_
Y_pred_dt_grid = best_classifier.predict(Xr_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(Yr_test, Y_pred_dt))

print("\nConfusion Matrix:")
print(confusion_matrix(Yr_test, Y_pred_dt))


# %% [markdown]
# Logistic Regression

# %%
pipeline = Pipeline([
  ('classifier', LogisticRegression(random_state=42))  # Logistic Regression classifier
])

# Define hyperparameter grid for GridSearchCV
param_grid = {
  'classifier__C': [0.001, 0.01, 0.1, 1, 10],  # Specify a list of values for C
  'classifier__solver': ['lbfgs', 'liblinear']  # Solvers to try
}

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(Xr_train, Yr_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Predict on the testing set using the best model
best_classifier = grid_search.best_estimator_
Y_pred_lr_grid = best_classifier.predict(Xr_test)

# Evaluate the performance of the Logistic Regression model
print("\nClassification Report for Logistic Regression:")
print(classification_report(Yr_test, Y_pred_lr))

print("\nConfusion Matrix for Logistic Regression:")
print(confusion_matrix(Yr_test, Y_pred_lr))


# %% [markdown]
# WITH HYPERPARAMETERTUNING RANDOMIZEDSEARCHCV

# %% [markdown]
# Decision Tree Classifier

# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import uniform, randint

# Define pipeline with Decision Tree Classifier
pipeline = Pipeline([
  ('classifier', DecisionTreeClassifier(random_state=42))
])

# Define hyperparameter distributions for RandomizedSearchCV
# Define hyperparameter distributions for RandomizedSearchCV
param_dist = {
  'classifier__criterion': ['gini', 'entropy'],  # Options for criterion
  'classifier__max_depth': randint(2, 20),       # Integer values between 2 and 20
  'classifier__min_samples_split': randint(2, 10),# Integer values between 2 and 10
  'classifier__min_samples_leaf': randint(1, 4)   # Integer values between 1 and 4
}


# Perform hyperparameter tuning using RandomizedSearchCV
random_dt = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_dt.fit(Xr_train, Yr_train)

# Get the best parameters
best_params = random_dt.best_params_
print("Best Parameters:", best_params)

# Predict on the testing set using the best model
best_classifier = random_dt.best_estimator_
Y_pred_dt_rand = best_classifier.predict(Xr_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(Yr_test, Y_pred_dt)) 

print("\nConfusion Matrix:")
print(confusion_matrix(Yr_test, Y_pred_dt))


# %% [markdown]
# Logistic Regression

# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform

# Define the pipeline
pipeline = Pipeline([
  ('classifier', LogisticRegression(random_state=42))  
])

# Define hyperparameters to search
param_dist = {
  'classifier__C': uniform(0.001, 10),  # Inverse regularization strength
  'classifier__solver': ['lbfgs', 'liblinear']  # Solvers to try
}

# Perform hyperparameter tuning using RandomizedSearchCV
random_lr = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_lr.fit(Xr_train, Yr_train)

# Get the best parameters
best_params = random_lr.best_params_
print("Best Parameters:", best_params)

# Predict on the testing set using the best model
best_classifier = random_lr.best_estimator_
Y_pred_lr_rand = best_classifier.predict(Xr_test)

# Evaluate the performance of the Logistic Regression model
print("\nClassification Report for Logistic Regression:")
print(classification_report(Yr_test, Y_pred_lr))

print("\nConfusion Matrix for Logistic Regression:")
print(confusion_matrix(Yr_test, Y_pred_lr))


# %% [markdown]
# KNN

# %%
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint

# Define the pipeline
pipeline = Pipeline([ ('classifier', KNeighborsClassifier())])

param_dist = {
    'classifier__n_neighbors': randint(1, 20),  # Number of neighbors
    'classifier__p': [1, 2]  # Distance metric: 1 for Manhattan, 2 for Euclidean
}

# Perform hyperparameter tuning using RandomizedSearchCV
random_knn = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_knn.fit(Xr_train, Yr_train)

# Get the best parameters
best_params = random_knn.best_params_
print("Best Parameters:", best_params)

# Predict on the testing set using the best model
best_classifier = random_knn.best_estimator_
Y_pred_knn_rand = best_classifier.predict(Xr_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(Yr_test, Y_pred_knn))

print("\nConfusion Matrix:")
print(confusion_matrix(Yr_test, Y_pred_knn))


# %% [markdown]
# Comparing Scores

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics for each model
dt_metrics = [accuracy_score(Yr_test, Y_pred_dt),  
              precision_score(Yr_test, Y_pred_dt, average='weighted'),
              recall_score(Yr_test, Y_pred_dt, average='weighted'),
              f1_score(Yr_test, Y_pred_dt, average='weighted')]

rf_metrics = [accuracy_score(Yr_test, Y_pred_rf), 
              precision_score(Yr_test, Y_pred_rf, average='weighted'),
              recall_score(Yr_test, Y_pred_rf, average='weighted'),
              f1_score(Yr_test, Y_pred_rf, average='weighted')]

knn_metrics = [accuracy_score(Yr_test, Y_pred_knn),  
              precision_score(Yr_test, Y_pred_knn, average='weighted'),
              recall_score(Yr_test, Y_pred_knn, average='weighted'),
              f1_score(Yr_test, Y_pred_knn, average='weighted')]

lr_metrics = [accuracy_score(Yr_test, Y_pred_lr), 
              precision_score(Yr_test, Y_pred_lr, average='weighted'),
              recall_score(Yr_test, Y_pred_lr, average='weighted'),
              f1_score(Yr_test, Y_pred_lr, average='weighted')]
# Metrics dataframe
metrics_df = pd.DataFrame([dt_metrics, rf_metrics, knn_metrics, lr_metrics],
                          columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                          index=['DecisionTreeClassifier', 'RandomForestClassifier', 'KNN', 'LogisticRegression'])

# Print the metrics
print("Metrics for DecisionTreeClassifier:")
print(metrics_df.loc['DecisionTreeClassifier'])
print("\nMetrics for RandomForestClassifier:")
print(metrics_df.loc['RandomForestClassifier'])
print("\nMetrics for KNN:")
print(metrics_df.loc['KNN'])
print("\nMetrics for Logistic Regression:")
print(metrics_df.loc['LogisticRegression'])

# Print the model with the best F1-score
best_model = metrics_df.idxmax()['Accuracy']
print("\nBest Model for the Project based on Accuracy:", best_model)


# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics for each model
dt_metrics_grid = [accuracy_score(Yr_test, Y_pred_dt_grid), 
              precision_score(Yr_test, Y_pred_dt_grid, average='weighted'),
              recall_score(Yr_test, Y_pred_dt_grid, average='weighted'),
              f1_score(Yr_test, Y_pred_dt_grid, average='weighted')]

lr_metrics_grid = [accuracy_score(Yr_test, Y_pred_lr_grid),  
              precision_score(Yr_test, Y_pred_lr_grid, average='weighted'),
              recall_score(Yr_test, Y_pred_lr_grid, average='weighted'),
              f1_score(Yr_test, Y_pred_lr_grid, average='weighted')]

# Metrics dataframe
metrics_df_grid = pd.DataFrame([dt_metrics_grid,lr_metrics_grid],
                          columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                          index=['DecisionTreeClassifier_grid','LogisticRegression_grid'])

# Print the metrics
print("Metrics for DecisionTreeClassifier:")
print(metrics_df_grid.loc['DecisionTreeClassifier_grid'])
print("\nMetrics for Logistic Regression:")
print(metrics_df_grid.loc['LogisticRegression_grid'])

# Print the model with the best F1-score
best_model = metrics_df_grid.idxmax()['Accuracy']
print("\nBest Model for the Project based on Accuracy:", best_model)


# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics for each model
dt_metrics_rand = [accuracy_score(Yr_test, Y_pred_dt_rand),  
              precision_score(Yr_test, Y_pred_dt_rand, average='weighted'),
              recall_score(Yr_test, Y_pred_dt_rand, average='weighted'),
              f1_score(Yr_test, Y_pred_dt_rand, average='weighted')]
knn_metrics_rand = [accuracy_score(Yr_test, Y_pred_knn_rand),  
              precision_score(Yr_test, Y_pred_knn_rand, average='weighted'),
              recall_score(Yr_test, Y_pred_knn_rand, average='weighted'),
              f1_score(Yr_test, Y_pred_knn_rand, average='weighted')]

lr_metrics_rand = [accuracy_score(Yr_test, Y_pred_lr_rand),  
              precision_score(Yr_test, Y_pred_lr_rand, average='weighted'),
              recall_score(Yr_test, Y_pred_lr_rand, average='weighted'),
              f1_score(Yr_test, Y_pred_lr_rand, average='weighted')]
# Metrics dataframe
metrics_df_rand = pd.DataFrame([dt_metrics_rand, knn_metrics_rand, lr_metrics_rand],
                          columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                          index=['DecisionTreeClassifier_rand', 'KNN_rand', 'LogisticRegression_rand'])

# Print the metrics
print("Metrics for DecisionTreeClassifier:")
print(metrics_df_rand.loc['DecisionTreeClassifier_rand'])
print("\nMetrics for KNN:")
print(metrics_df_rand.loc['KNN_rand'])
print("\nMetrics for Logistic Regression:")
print(metrics_df_rand.loc['LogisticRegression_rand'])

# Print the model with the best F1-score
best_model = metrics_df_rand.idxmax()['Accuracy']
print("\nBest Model for the Project based on Accuracy:", best_model)


# %% [markdown]
# Accuracy Visualization

# %% [markdown]
# RandomizedSearchCv

# %%
from sklearn.metrics import accuracy_score

# Calculate accuracy for each model
dt_accuracy_rand = accuracy_score(Yr_test, Y_pred_dt_rand)
knn_accuracy_rand = accuracy_score(Yr_test, Y_pred_knn_rand)
lr_accuracy_rand = accuracy_score(Yr_test, Y_pred_lr_rand)

# Create the accuracy_rand list
accuracy_rand = [dt_accuracy_rand, knn_accuracy_rand, lr_accuracy_rand]

# Model names list
model_names_rand = ['DecisionTreeClassifier', 'KNN', 'LogisticRegression']


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(model_names_rand, accuracy_rand, color='coral', alpha=0.7)
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.title('Model Accuracy (Random Hyperparameters)')
plt.gca().invert_yaxis()  # Invert y-axis for readability (highest accuracy on top)
plt.xlim(0, 1)
for i, v in enumerate(accuracy_rand):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center', ha='left', fontsize=10)  # Adjust offset and precision as needed

plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# %% [markdown]
# GridSearchCv

# %%
from sklearn.metrics import accuracy_score

# Calculate accuracy for each model
dt_accuracy_grid = accuracy_score(Yr_test, Y_pred_dt_grid)
lr_accuracy_grid = accuracy_score(Yr_test, Y_pred_lr_grid)

# Create the accuracy_rand list
accuracy_grid = [dt_accuracy_grid, lr_accuracy_grid]

# Model names list
model_names_grid = ['DecisionTreeClassifier', 'LogisticRegression']


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(model_names_grid, accuracy_grid, color='green', alpha=0.7)
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.title('Model Accuracy (GridSearch)')
plt.gca().invert_yaxis()
plt.xlim(0, 1)
for i, v in enumerate(accuracy_grid):
    plt.text(v + 0.01, i, f"{v:.2f}", va='center', ha='left', fontsize=10)  # Adjust offset and precision as needed

plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()


# %% [markdown]
# Without Hyperparameter tuning

# %%
from sklearn.metrics import accuracy_score

# Calculate accuracy for each model
dt_accuracy = accuracy_score(Yr_test, Y_pred_dt)
knn_accuracy = accuracy_score(Yr_test, Y_pred_knn)
lr_accuracy = accuracy_score(Yr_test, Y_pred_lr)
rf_accuracy = accuracy_score(Yr_test,Y_pred_rf)

# Create the accuracy_rand list
accuracy = [dt_accuracy, knn_accuracy, lr_accuracy,rf_accuracy]

# Model names list
model_names = ['DecisionTreeClassifier', 'KNN', 'LogisticRegression','RandomForestClassifier']


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(model_names, accuracy, color='yellow', alpha=0.7)
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.title('Model Accuracy ')
plt.gca().invert_yaxis()  # Invert y-axis for readability (highest accuracy on top)
plt.xlim(0, 1)
for i, v in enumerate(accuracy):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center', ha='left', fontsize=10)  # Adjust offset and precision as needed

plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# %% [markdown]
# BEST MODEL

# %%
# Combine the three dataframes
combined_metrics_df = pd.concat([metrics_df_rand, metrics_df_grid, metrics_df])

# Find the model with the highest accuracy
best_model = combined_metrics_df['Accuracy'].idxmax()

print(f"Best model based on accuracy: {best_model}")


# %%



