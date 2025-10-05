from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and prepare dataset
df = pd.read_csv("C:/Users/HP/Downloads/archive/heart_disease_data.csv")
if 'dataset' in df.columns:
    df = df[df['dataset']=='Cleveland']
df.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in df.columns]
cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = df.loc[:, [c for c in cols if c in df.columns]]

# Label encode categorical columns
label_encoders = {}
for cat in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
    if cat in df.columns and df[cat].dtype == 'object':
        le = LabelEncoder()
        df[cat] = le.fit_transform(df[cat].astype(str))
        label_encoders[cat] = le

X = df.drop('num', axis=1)
y = df['num']

X = X.fillna(X.mode().iloc[0])
y = y.fillna(y.mode()[0])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train model
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=300)
clf.fit(X_res, y_res)

# Save model, scaler, and encoders
joblib.dump(clf, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Load saved objects
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return "Heart Disease Prediction API running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])

    # Encode categorical variables carefully
    for cat in label_encoders:
        if cat in input_df.columns:
            le = label_encoders[cat]
            try:
                input_df[cat] = le.transform(input_df[cat].astype(str))
            except ValueError as e:
                return jsonify({"error": f"Invalid value for {cat}. Allowed values: {list(le.classes_)}"}), 400
        else:
            return jsonify({'error': f'Field {cat} missing'}), 400

    input_df = input_df.fillna(X.mode().iloc[0])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return jsonify({'predicted_severity': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
