# Customer-Churn-Analysis
Customer churn analysis project using Python, Excel, SQL & Power BI
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#TASK 1
df = pd.read_csv('customer_churn.csv')
# Display basic info
print("Initial shape:", df.shape)
print(df.head())

print("Missing values per column:", df.isnull().sum())

#handle missing values
# Fill numerical columns with median and categorical columns with mode
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median())
    else:
        df[col].fillna(df[col].mode()[0])

#to check duplicates
df.drop_duplicates(inplace=True)
print("Duplicate rows:", df.duplicated().sum())

# Encode categorical features using LabelEncoder
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# to check data is encoded or not
non_numeric_cols = df.select_dtypes(include=['object']).columns
print("Non-numeric columns:", list(non_numeric_cols))

#Strandardize numerical values
scaler= StandardScaler()
numerical_cols=df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print(df[numerical_cols].describe())

#split train and test sets
print(df['Churn Label'].value_counts())
x = df.drop('Churn Label', axis=1)
y = df['Churn Label']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print(y_train.value_counts())

#TASK 2
# Percentage of customers who churned
churn_counts = df['Churn Label'].value_counts()
churn_percentage = churn_counts / len(df) * 100
print(churn_percentage)
sns.countplot(x='Churn Label', data=df)
plt.title('Churn Count')
plt.show()

# how does churn vary by gender,contract,age,tenure
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
sns.countplot(x='Gender', hue='Churn Label', data=df)
plt.title('Churn by Gender')
plt.subplot(2, 2, 2)
sns.countplot(x='Contract', hue='Churn Label', data=df)
plt.title('Churn by Contract Type')
plt.subplot(2, 2, 3)
sns.boxplot(x='Churn Label', y='Age', data=df)
plt.title('Age Distribution by Churn')
plt.subplot(2, 2, 4)
sns.boxplot(x='Churn Label', y='Tenure in Months', data=df)
plt.title('Tenure Distribution by Churn')
plt.tight_layout()
plt.show()

#Most correlated churn
corr_matrix = df.corr()
churn_corr = corr_matrix['Churn Label'].drop('Churn Label')
top_corr_features = churn_corr.abs().sort_values(ascending=False).head(10)
top_features = top_corr_features.index.tolist() + ['Churn Label']
top_corr_matrix = df[top_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(top_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Most Correlated with Churn')
plt.show()

#top 5 factors influencing customer retention
corr_matrix = df.corr()
churn_corr = corr_matrix['Churn Label']
top_retention_factors = churn_corr[churn_corr < 0].sort_values().head(5)
print(top_retention_factors)
plt.figure(figsize=(10, 6))
for i, feature in enumerate(top_retention_factors.index, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=feature, y='Churn Label', data=df, hue='Churn Label', alpha=0.6)
    plt.title(f'{feature} vs Churn Label')
plt.tight_layout()
plt.show()

# Correlation with target
corr = df.corr()['Churn Label'].sort_values(ascending=False)
print("Top features correlated with target:\n", corr.head(10))


categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(df[col].value_counts())
leak_features = ['Churn Reason', 'Churn Category', 'Customer Status']
x = df.drop(columns=['Churn Label'] + leak_features)
y = df['Churn Label']

print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train distribution:\n", y_train.value_counts())
print("y_test distribution:\n", y_test.value_counts())


#TASK 3 - Predictive modelling- build and create model , compared by evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }
results_df = pd.DataFrame(results).T
print(results_df)

#Confusion Matrix
best_model_name = results_df['F1-score'].idxmax()
best_model = models[best_model_name]
y_pred = best_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'{best_model_name} - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print(y_pred)

#ROC and AUC curve
plt.figure(figsize=(7,6))
for name, model in models.items():
    y_proba = model.predict_proba(x_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

importances = best_model.feature_importances_
features = x.columns
fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
fi_df = fi_df.sort_values('Importance', ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=fi_df.head(10))
plt.title(f'Top 10 Important Features - {best_model_name}')
plt.show()


#TASK 4
# Example: Percentage of churn by contract type
churn_by_contract = df.groupby('Contract')['Churn Label'].mean().sort_values(ascending=False)
print(churn_by_contract)
sns.barplot(x=churn_by_contract.index, y=churn_by_contract.values)
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate')
plt.show()

#Highlight factors driving dissatisfaction.
top_factors = ['Satisfaction Score', 'Customer Status', 'Churn Reason', 'Contract', 'Monthly Charge']
plt.figure(figsize=(12, 10))
for i, feature in enumerate(top_factors, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Churn Label', y=feature, data=df)
    plt.title(f'{feature} vs Churn')
plt.subplot(2, 3, 6)
plt.axis('off')
plt.tight_layout()
plt.show()

df.to_csv('processed_churn_data.csv', index=False)
from google.colab import files
files.download('processed_churn_data.csv')
