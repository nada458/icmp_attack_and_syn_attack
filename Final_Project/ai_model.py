import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


DATA_FILE = 'final_multiclass_dataset.csv'
MODEL_FILE = 'multiclass_ids_model.joblib'
RANDOM_STATE = 42


FEATURES = [
    'frame.time_delta', 'ip.proto', 'ip.len', 
    'tcp.srcport', 'tcp.dstport', 'tcp.flags.syn', 
    'icmp.type', 'icmp.seq', 'icmp.code'
]

print(f"Loading dataset from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)

    
    df.columns = df.columns.str.strip() 

    
    for feature in FEATURES:
        
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    
    df = df.fillna(0)
    
    
    X = df[FEATURES]
    y = df['Label']
    
except KeyError as e:
    
    print(f" Error: Feature column {e} not found. (The CSV header is still corrupted).")
    exit()
except Exception as e:
    print(f" An unexpected error occurred during data loading: {e}")
    exit()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print("-" * 50)


print("Training Random Forest Multiclass Model...")
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1) 
model.fit(X_train, y_train)


print("Evaluating...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%")
print("-" * 50)


print("Classification Report (0=Normal, 1=ICMP, 2=SYN):")
print(classification_report(y_test, y_pred, zero_division=0))


print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("-" * 50)


print("Feature Importance (Most influential features):")
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

for i in range(len(X.columns)):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
print("-" * 50)


joblib.dump(model, MODEL_FILE)
print(f" Model saved successfully as '{MODEL_FILE}'")
