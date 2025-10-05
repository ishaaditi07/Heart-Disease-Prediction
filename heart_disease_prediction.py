import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:/Users/HP/Downloads/archive/heart_disease_data.csv")

# Filter for Cleveland records if column exists
if 'dataset' in df.columns:
    df = df[df['dataset']=='Cleveland']

# Clean column names
df.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in df.columns]

# Use corrected column list with 'thalch' not 'thalach'
cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

# Select columns that exist in the dataset
df = df.loc[:, [c for c in cols if c in df.columns]]

# Encode categorical variables if they are of object type
for cat in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
    if cat in df.columns and df[cat].dtype == 'object':
        df[cat] = LabelEncoder().fit_transform(df[cat].astype(str))

# Separate features and target
X = df.drop('num', axis=1)
y = df['num']

# Handle missing values by mode for features and target
X = X.fillna(X.mode().iloc[0])
y = y.fillna(y.mode()[0])

# Normalize numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

# Print basic statistics
print("Average Age:", df['age'].mean())
print("Average Cholesterol:", df['chol'].mean())
print("Target Distribution:\n", y.value_counts())

# Visualizations
sns.histplot(df['age'], bins=15)
plt.title("Age distribution")
plt.show()

sns.countplot(x=y)
plt.title("Target class counts")
plt.show()

df['sex'].value_counts().plot.pie(labels=['Male', 'Female'], autopct='%1.1f%%')
plt.title("Gender ratio")
plt.show()

sns.boxplot(x=df['chol'])
plt.title("Cholesterol boxplot")
plt.show()

sns.scatterplot(x=df['age'], y=df['thalch'])
plt.title("Age vs Max Heart Rate")
plt.show()

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Ordinal Logistic Regression model training and evaluation
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=300)
clf.fit(X_res, y_res)
y_pred = clf.predict(X)
print("Logistic Regression Classification Report:\n", classification_report(y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

# Decision Tree Classifier model training and evaluation
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_res, y_res)
y_pred_dt = dt.predict(X)
print("Decision Tree Classification Report:\n", classification_report(y, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_dt))

# Anomaly Detection: One-Class SVM
ocsvm = OneClassSVM(gamma='auto')
ocsvm.fit(X)
anomaly_pred_svm = ocsvm.predict(X)
print("Number of anomalies detected by One-Class SVM:", np.sum(anomaly_pred_svm == -1))

# Anomaly Detection: Isolation Forest
iso = IsolationForest(contamination=0.12, random_state=42)
iso.fit(X)
anomaly_pred_if = iso.predict(X)
print("Number of anomalies detected by Isolation Forest:", np.sum(anomaly_pred_if == -1))

# Print workflow summary
print("""
Workflow:
1. Data Preprocessing (missing values, encoding, normalization)
2. Feature Selection (using standard Cleveland dataset columns with correction)
3. SMOTE for addressing class imbalance
4. Model Training for Ordinal Severity Prediction (Logistic Regression, Decision Tree)
5. Evaluation using classification reports and confusion matrices
6. Anomaly Detection using One-Class SVM and Isolation Forest
7. Visualization of key dataset properties and relationships
""")
