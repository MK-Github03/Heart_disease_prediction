# Cell
# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Loading the dataset
file_path = "heart.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
df.head()

# Cell
#Data Preprocessing
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables using Label Encoding
label_encoders = {}
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  

# Displaying the first few
df.head()
df.describe()

# Cell
#Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")

# Plot distribution of numerical features
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], kde=True, bins=30, color="blue")
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()

# Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["HeartDisease"], palette="viridis")
plt.title("Class Distribution (Heart Disease vs. No Heart Disease)")
plt.xlabel("Heart Disease (0: No, 1: Yes)")
plt.ylabel("Count")
plt.show()

# Compute correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()


# Cell
#Feature selection using random forest

from sklearn.ensemble import RandomForestClassifier

# Defining the  features and target variable
X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importance
feature_importances = rf_model.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

#printing the feature importance
print(importance_df.to_string(index=False))


# Cell
#ML models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability scores for ROC-AUC

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })




#MLP 

from sklearn.neural_network import MLPClassifier

# Initialize and train MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# Make predictions
y_pred_mlp = mlp_model.predict(X_test)
y_proba_mlp = mlp_model.predict_proba(X_test)[:, 1]

results_df = pd.DataFrame(results)
# Evaluate MLP model
mlp_results = {
    "Model": "Multi-Layer Perceptron (MLP)",
    "Accuracy": accuracy_score(y_test, y_pred_mlp),
    "F1-Score": f1_score(y_test, y_pred_mlp),
    "ROC-AUC": roc_auc_score(y_test, y_proba_mlp)
}

# Convert MLP results into a DataFrame and add to existing results

mlp_results_df = pd.DataFrame([mlp_results])
results_df = pd.concat([results_df, mlp_results_df], ignore_index=True)

# Display updated results
display(results_df)




# Cell
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


#splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define hyperparameter grid for Random Forest
rf_param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# Define hyperparameter grid for Gradient Boosting
gb_param_grid = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 10]
}

# Perform GridSearchCV for Random Forest
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Perform GridSearchCV for Gradient Boosting
gb_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gb_grid_search.fit(X_train, y_train)

# Get best parameters and best accuracy scores
best_rf_params = rf_grid_search.best_params_
best_rf_score = rf_grid_search.best_score_

best_gb_params = gb_grid_search.best_params_
best_gb_score = gb_grid_search.best_score_

# Store tuning results in a DataFrame
tuning_results = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting"],
    "Best Parameters": [best_rf_params, best_gb_params],
    "Best Accuracy": [best_rf_score, best_gb_score]
})

# Display the results
from IPython.display import display
display(tuning_results)


# Cell
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define hyperparameter grid for Random Forest
rf_param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# Define hyperparameter grid for Gradient Boosting
gb_param_grid = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 10]
}

# Define hyperparameter grid for MLP
mlp_param_grid = {
    "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "sgd"],
    "learning_rate_init": [0.001, 0.01, 0.1],
    "max_iter": [500, 1000]
}

# Perform GridSearchCV for Random Forest
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Perform GridSearchCV for Gradient Boosting
gb_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gb_grid_search.fit(X_train, y_train)

# Perform GridSearchCV for MLP
mlp_grid_search = GridSearchCV(MLPClassifier(random_state=42), mlp_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
mlp_grid_search.fit(X_train, y_train)

# Get best parameters and best accuracy scores
best_rf_params = rf_grid_search.best_params_
best_rf_score = rf_grid_search.best_score_

best_gb_params = gb_grid_search.best_params_
best_gb_score = gb_grid_search.best_score_

best_mlp_params = mlp_grid_search.best_params_
best_mlp_score = mlp_grid_search.best_score_

best_rf_model = RandomForestClassifier(**best_rf_params, random_state=42)
best_gb_model = GradientBoostingClassifier(**best_gb_params, random_state=42)
best_mlp_model = MLPClassifier(**best_mlp_params, random_state=42)

# Fit models on training data
best_rf_model.fit(X_train, y_train)
best_gb_model.fit(X_train, y_train)
best_mlp_model.fit(X_train, y_train)

# Store tuning results in a DataFrame
tuning_results = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "MLP Neural Network"],
    "Best Parameters": [best_rf_params, best_gb_params, best_mlp_params],
    "Best Accuracy": [best_rf_score, best_gb_score, best_mlp_score]
})

# Display the results
from IPython.display import display
display(tuning_results)


# Cell
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns


# Retrieve feature importances for Random Forest and Gradient Boosting
rf_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": best_rf_model.feature_importances_}).sort_values(by="Importance", ascending=False)
gb_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": best_gb_model.feature_importances_}).sort_values(by="Importance", ascending=False)

# Retrieve feature importances for MLP using permutation importance
mlp_importance = permutation_importance(best_mlp_model, X_test, y_test, n_repeats=10, random_state=42, scoring="accuracy")
mlp_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": mlp_importance.importances_mean}).sort_values(by="Importance", ascending=False)

# Plot feature importance for all models
plt.figure(figsize=(12, 8))

# Random Forest
plt.subplot(1, 3, 1)
sns.barplot(x=rf_importance_df["Importance"], y=rf_importance_df["Feature"], palette="viridis")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance (Random Forest)")

# Gradient Boosting
plt.subplot(1, 3, 2)
sns.barplot(x=gb_importance_df["Importance"], y=gb_importance_df["Feature"], palette="plasma")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance (Gradient Boosting)")

# MLP Neural Network
plt.subplot(1, 3, 3)
sns.barplot(x=mlp_importance_df["Importance"], y=mlp_importance_df["Feature"], palette="magma")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance (MLP)")

plt.tight_layout()
plt.show()


# Cell


