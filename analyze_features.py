import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import joblib

model  = joblib.load('best_heart_disease_model.pkl')
cols   = joblib.load('model_columns.pkl')

print("=== Random Forest Feature Importances ===\n")
importances = model.feature_importances_
pairs = sorted(zip(cols, importances), key=lambda x: -x[1])
for feat, imp in pairs:
    bar = "#" * int(imp * 200)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

print("\n=== Diagnosis of Failing Cases ===\n")

scaler = joblib.load('standard_scaler.pkl')

FEATURE_COLS     = ['age','sex','cp','trestbps','chol','fbs','restecg',
                    'thalach','exang','oldpeak','slope','ca','thal']
CATEGORICAL_COLS = ['cp','restecg','slope','ca','thal']

cases = [
    ("Case D - High-risk, 3 vessels (FAIL)",
     dict(age=62,sex=1,cp=2,trestbps=130,chol=231,fbs=0,restecg=0,
          thalach=146,exang=0,oldpeak=1.8,slope=1,ca=3,thal=3), 1),
    ("Case H - High-risk, low max HR (FAIL)",
     dict(age=59,sex=1,cp=0,trestbps=140,chol=177,fbs=0,restecg=0,
          thalach=162,exang=1,oldpeak=0.0,slope=2,ca=1,thal=3), 1),
    ("Case E - High-risk elderly (PASS)",
     dict(age=67,sex=1,cp=0,trestbps=160,chol=286,fbs=0,restecg=2,
          thalach=108,exang=1,oldpeak=1.5,slope=1,ca=3,thal=2), 1),
]

for label, raw, expected in cases:
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    df = df.reindex(columns=cols, fill_value=0)
    scaled = scaler.transform(df)
    proba = model.predict_proba(scaled)[0]

    print(f"[{label}]")
    print(f"   Expected          : {'Heart Disease' if expected==1 else 'Healthy'}")
    print(f"   P(Healthy)        : {proba[0]*100:.1f}%")
    print(f"   P(Heart Disease)  : {proba[1]*100:.1f}%")

    top_risk = [(cols[i], df.iloc[0, i], importances[i])
                for i in range(len(cols)) if df.iloc[0, i] != 0]
    top_risk.sort(key=lambda x: -x[2])
    print(f"   Top active features:")
    for feat, val, imp in top_risk[:5]:
        print(f"      {feat:<25} val={val}  importance={imp:.4f}")
    print()
