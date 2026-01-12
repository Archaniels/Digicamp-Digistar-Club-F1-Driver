import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

df = pd.read_csv('F1Drivers_Dataset.csv')

print("Dataset shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)

df = df.fillna(0)

df['Champion'] = df['Champion'].astype(int)

features = [
    'Race_Entries',
    'Race_Starts',
    'Pole_Positions',
    'Race_Wins',
    'Podiums',
    'Fastest_Laps',
    'Points',
    'Pole_Rate',
    'Start_Rate',
    'Win_Rate',
    'Podium_Rate',
    'FastLap_Rate',
    'Points_Per_Entry',
    'Years_Active'
]

X = df[features]
y = df['Champion']

print(f"\nTotal drivers: {len(df)}")
print(f"Champions: {y.sum()}")
print(f"Non-champions: {len(y) - y.sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'  # Handle imbalanced data
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Champion', 'Champion']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

joblib.dump(model, 'f1_champion_model.pkl')
joblib.dump(scaler, 'f1_scaler.pkl')
joblib.dump(features, 'f1_features.pkl')

print("\n✅ Model, scaler, and features saved successfully!")
print("Files created:")
print("  - f1_champion_model.pkl")
print("  - f1_scaler.pkl")
print("  - f1_features.pkl")

print("\n" + "="*50)
print("SAMPLE PREDICTION TEST")
print("="*50)

sample_data = pd.DataFrame([{
    'Race_Entries': 308,
    'Race_Starts': 306,
    'Pole_Positions': 68,
    'Race_Wins': 91,
    'Podiums': 155,
    'Fastest_Laps': 77,
    'Points': 1566.0,
    'Pole_Rate': 0.22,
    'Start_Rate': 0.99,
    'Win_Rate': 0.30,
    'Podium_Rate': 0.50,
    'FastLap_Rate': 0.25,
    'Points_Per_Entry': 5.08,
    'Years_Active': 19
}])

sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled)
probability = model.predict_proba(sample_scaled)

print("\nSample Driver Stats (Schumacher-like):")
print(sample_data.T)
print(f"\nPrediction: {'CHAMPION' if prediction[0] == 1 else 'NON-CHAMPION'}")
print(f"Probability of being Champion: {probability[0][1]:.2%}")