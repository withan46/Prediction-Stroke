#import data from drive
from google.colab import drive
drive.mount('/content/drive')

# Core libraries for data handling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Sklearn libraries for machine learning models and utilities
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

# Additional libraries for specific functionalities
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.sparse import hstack
import time
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.colors as mcolors

# 1. Reading data from the Excel file
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/healthcare-dataset-stroke-data.csv')
df.head(5)

# Encode categorical features
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# df=df.fillna(method='bfill')
# df.isnull().sum()

# # Drop rows with any NaN values
# df = df.dropna()

# # Drop columns with any NaN values
# df = df.dropna(axis=1)
# df.head(5)

# # Example: Use Linear Regression to predict missing values of 'bmi'
# from sklearn.linear_model import LinearRegression

# # Split the data into two parts: with and without the missing values in 'bmi'
# df_with_bmi = df[df['bmi'].notnull()]
# df_without_bmi = df[df['bmi'].isnull()]

# # Use other features to predict 'bmi'
# features = df_with_bmi.columns.difference(['bmi'])
# model = LinearRegression().fit(df_with_bmi[features], df_with_bmi['bmi'])
# predicted_bmi = model.predict(df_without_bmi[features])

# # Fill in the missing values with the predictions
# df.loc[df['bmi'].isnull(), 'bmi'] = predicted_bmi
# df.head(5)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = imputer.fit_transform(df)
df = pd.DataFrame(df_imputed, columns=df.columns)
df.head(5)

# Drop the 'id' column
df = df.drop('id', axis=1)

# Split the data into features and target variable
X = df.drop('stroke', axis=1)
y = df['stroke']

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ----------------------unbalanced--------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Count the number of instances in each class
# import matplotlib.pyplot as plt
# class_counts = y_resampled.value_counts()
# print(class_counts)
# # Plot the class distribution
# beingsaved = plt.figure(figsize=(5, 4))
# colors = ['red', 'green']
# plt.bar(['Normal', 'Stroke'], class_counts.values, color=colors)
# plt.xlabel('Result')
# plt.ylabel('Count')
# plt.title('Balanced Train Set Class Distribution')
# plt.show()

labels =df['stroke'].value_counts(sort = True).index
sizes = df['stroke'].value_counts(sort = True)

colors = ["#C4A484","yellow"]
explode = (0.05,0)

plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,)

plt.title('Chart of stroke')
plt.show()

# -------------------- Second plot ------------------------------
# Count the number of instances in each class
class_counts = df['stroke'].value_counts()
print(class_counts)

# Plot the class distribution
plt.figure(figsize=(6, 5))
colors = ['#C4A484', 'yellow']
bars = plt.bar(['No stroke', 'Stroke'], class_counts.values, color=colors)

# Adding data labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

# Set labels and title
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Dataset Information', fontsize=14)

# Improve layout and show the plot
plt.tight_layout()
plt.show()

df_numerical = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
numeric_columns = df[df_numerical]
corr_matrix = numeric_columns.corr()

sns.heatmap(corr_matrix, cmap='YlOrBr')
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = '{:.2f}'.format(corr_matrix.iloc[i, j])
        plt.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=10)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.show()

# Define the background color and continuous variables to plot
background_color = '#fbfbfb'
conts = ['age', 'avg_glucose_level', 'bmi']

# Create a figure with a specified background color, size, and dpi
fig = plt.figure(figsize=(12, 12), dpi=150, facecolor=background_color)
gs = fig.add_gridspec(4, 3)
gs.update(wspace=0.1, hspace=0.4)

plot = 0
# Create subplots for the first row of KDE plots
for col in range(3):
    locals()["ax"+str(plot)] = fig.add_subplot(gs[0, col])
    locals()["ax"+str(plot)].set_facecolor(background_color)
    locals()["ax"+str(plot)].tick_params(axis='y', left=False)
    locals()["ax"+str(plot)].get_yaxis().set_visible(False)
    for s in ["top", "right", "left"]:
        locals()["ax"+str(plot)].spines[s].set_visible(False)
    plot += 1

# Reset plot counter for the KDE plots
plot = 0

# Separate the dataset into stroke and no stroke groups
s = df[df['stroke'] == 0]
ns = df[df['stroke'] == 1]

# Generate KDE plots for the specified continuous features
for feature in conts:
    sns.kdeplot(s[feature], ax=locals()["ax"+str(plot)], color="#C4A484", fill=True, linewidth=1.5, ec='black', alpha=0.9, zorder=3, legend=False)
    sns.kdeplot(ns[feature], ax=locals()["ax"+str(plot)], color="yellow", fill=True, linewidth=1.5, ec='black', alpha=0.9, zorder=3, legend=False)
    locals()["ax"+str(plot)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(plot)].set_xlabel(feature.capitalize())
    plot += 1

ax0.text(0, 0.056, 'Numeric Variables by Stroke & No Stroke', fontsize=20, fontweight='bold', fontfamily='serif')
plt.show()

no_stroke = df[df['stroke'] == 0]
indicators = ['age', 'avg_glucose_level', 'bmi']

mins = no_stroke[indicators].min()
maxs = no_stroke[indicators].max()
averages = no_stroke[indicators].mean()

plt.figure(figsize=(15, 10))
bar_width = 0.25
index = np.arange(len(indicators))

# Average values - Bars
plt.bar(index, averages, bar_width, color='skyblue', label='Average', alpha=0.7)

# Min and Max values - Lines with markers
plt.plot(index, mins, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Min')
plt.plot(index, maxs, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Max')

# Adding data labels
for i in index:
    plt.text(i, averages[i], round(averages[i], 2), ha='center', va='bottom')
    plt.text(i, mins[i], round(mins[i], 2), ha='center', va='top', color='red')
    plt.text(i, maxs[i], round(maxs[i], 2), ha='center', va='bottom', color='green')

adjusted_index = index + bar_width / 2
plt.xticks(adjusted_index, indicators, rotation=45)
plt.title('Comparative Analysis of Min, Max, and Average Values for Health Indicators')
plt.xlabel('Health Indicators')
plt.ylabel('Indicator Values')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()

no_stroke = df[df['stroke'] == 1]
indicators = ['age', 'avg_glucose_level', 'bmi']

mins = no_stroke[indicators].min()
maxs = no_stroke[indicators].max()
averages = no_stroke[indicators].mean()

plt.figure(figsize=(15, 10))
bar_width = 0.25
index = np.arange(len(indicators))

# Average values - Bars
plt.bar(index, averages, bar_width, color='skyblue', label='Average', alpha=0.7)

# Min and Max values - Lines with markers
plt.plot(index, mins, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Min')
plt.plot(index, maxs, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Max')

# Adding data labels
for i in index:
    plt.text(i, averages[i], round(averages[i], 2), ha='center', va='bottom')
    plt.text(i, mins[i], round(mins[i], 2), ha='center', va='top', color='red')
    plt.text(i, maxs[i], round(maxs[i], 2), ha='center', va='bottom', color='green')

adjusted_index = index + bar_width / 2
plt.xticks(adjusted_index, indicators, rotation=45)
plt.title('Comparative Analysis of Min, Max, and Average Values for Health Indicators')
plt.xlabel('Health Indicators')
plt.ylabel('Indicator Values')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()

# List of variables for which you want to plot box plots
variables_to_plot = ['age', 'avg_glucose_level', 'bmi']

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=len(variables_to_plot), ncols=1, figsize=(10, 20))

# Loop through the variables and create box plots
for i, var in enumerate(variables_to_plot):
    sns.boxplot(x='stroke', y=var, data=df, ax=axes[i])
    axes[i].set_title(f'Box Plot of {var} by Stroke Status')
    axes[i].set_xlabel('Stroke Status (0: No Stroke, 1: Stroke)')
    axes[i].set_ylabel(var)

# Adjust layout
plt.tight_layout()
plt.show()

# Initialize the classifiers
svm = SVC()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
logistic_regression = LogisticRegression()
gradient_boosting = GradientBoostingClassifier()
gaussian_nb = GaussianNB()
knn = KNeighborsClassifier()
neural_network = MLPClassifier()

def check_balance(y):
    counts = y.value_counts()
    return 'Balanced' if min(counts) / max(counts) > 0.5 else 'Unbalanced'


results = []
missing_values_method = 'bfill'
balance_status = check_balance(y_resampled)
# balance_status = check_balance(y_train)

for i in range(10):
    # Modify random_state for each classifier
    classifiers = {
        'AdaBoost': AdaBoostClassifier(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'MLP Classifier': MLPClassifier(max_iter=500),
        'SVM': svm,
        'Decision Tree': decision_tree,
        'Random Forest': random_forest,
        'Logistic Regression': logistic_regression,
        'Gradient Boosting': gradient_boosting,
        'Gaussian Naive Bayes': gaussian_nb,
        'K-Nearest Neighbors': knn,
        'Neural Network': neural_network
    }

    # Shuffle and split the data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=i)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Define a colormap
    colors = ["#C4A484", "#FDFD96"]
    cmap_name = "my_custom_map"
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    for clf_name, clf in classifiers.items():
      start_time = time.time()
      clf.fit(X_train, y_train)

      # Predict on train set
      y_train_pred = clf.predict(X_train)
      # Predict on test set
      y_test_pred = clf.predict(X_test)

      end_time = time.time()

      for set_name, y_true, y_predictions in [('Train', y_train, y_train_pred), ('Test', y_test, y_test_pred)]:
        accuracy = accuracy_score(y_true, y_predictions)
        cm = confusion_matrix(y_true, y_predictions)
        f1 = f1_score(y_true, y_predictions)
        recall = recall_score(y_true, y_predictions)
        precision = precision_score(y_true, y_predictions)
        roc_auc = roc_auc_score(y_true, y_predictions)

        print(f"Model: {clf_name}")
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"F1 Score: {f1}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        # print(f"Mean Absolute Error: {mae}")
        # print(f"Root Mean Squared Error: {rmse}")
        print(f"ROC AUC Score: {roc_auc}")

        # Store the results in a dictionary
        result = {
            'Experiment ID': i + 1,
            'Model': clf_name,
            'Missing values method': missing_values_method,
            'Set': set_name,
            'Balance': balance_status,
            'Number of Training Samples': len(X_train),
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Recall': recall,
            'Precision': precision,
            'ROC AUC Score': roc_auc,
            'Execution Time (seconds)': end_time - start_time,
            'TP': cm[1, 1],
            'TN': cm[0, 0],
            'FP': cm[0, 1],
            'FN': cm[1, 0]
            }
        results.append(result)

        # Visualization (Optional)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=True,
                      xticklabels=["Normal", "Stroke"], yticklabels=["Normal", "Stroke"])
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title(f"{clf_name} - Experiment {i + 1}")
        plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Export to Excel
results_df.to_excel('experiment_results.xlsx', index=False)

from google.colab import files
files.download('experiment_results.xlsx')

