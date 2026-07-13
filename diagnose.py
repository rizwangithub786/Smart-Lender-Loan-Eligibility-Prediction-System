import os
import pickle
import pandas as pd
import numpy as np

def run_diagnostics():
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset/loan_data.csv')
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
    
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return
        
    df = pd.read_csv(dataset_path)
    
    print("=== 1. Dataset Shape and Target Distribution ===")
    print(f"Total Rows: {len(df)}")
    print(df['Loan_Status'].value_counts())
    
    print("\n=== 2. Credit History vs Loan Status ===")
    ct = pd.crosstab(df['Credit_History'], df['Loan_Status'], margins=True)
    print(ct)
    print("\nApproval Rate by Credit History:")
    print(pd.crosstab(df['Credit_History'], df['Loan_Status'], normalize='index') * 100)
    
    print("\n=== 3. Model Feature Importances ===")
    imp_path = os.path.join(models_dir, 'feature_importance_xgboost.pkl')
    if os.path.exists(imp_path):
        with open(imp_path, 'rb') as f:
            feat_imp = pickle.load(f)
        for idx, feat in enumerate(feat_imp):
            print(f"{idx+1}. {feat['Feature']}: {feat['Importance']:.4f} ({feat['Importance']*100:.2f}%)")
    else:
        print("Feature importance pickle not found!")
        
    print("\n=== 4. Test Preds across different Credit History values ===")
    model_path = os.path.join(models_dir, 'loan_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    encoder_path = os.path.join(models_dir, 'encoder.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            encoder_meta = pickle.load(f)
            
        # Create a sample input representing a strong profile except for Credit History
        # Inputs: Male, Married, 0, Graduate, No, Semiurban, 8000, 2000, 100, 360, Credit_History=1.0 vs 0.0
        features = encoder_meta['features']
        
        # Test Credit_History = 1.0
        sample_good_credit = {
            'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate', 
            'Self_Employed': 'No', 'ApplicantIncome': 8000.0, 'CoapplicantIncome': 2000.0, 
            'LoanAmount': 100.0, 'Loan_Amount_Term': 360.0, 'Credit_History': 1.0, 'Property_Area': 'Semiurban'
        }
        # Test Credit_History = 0.0
        sample_bad_credit = sample_good_credit.copy()
        sample_bad_credit['Credit_History'] = 0.0
        
        for name, sample in [('Good Credit Profile', sample_good_credit), ('Bad Credit Profile', sample_bad_credit)]:
            s_df = pd.DataFrame([sample])
            
            # Feature Engineering: DTI
            total_inc = s_df['ApplicantIncome'] + s_df['CoapplicantIncome']
            monthly_pay = (s_df['LoanAmount'] * 1000) / s_df['Loan_Amount_Term']
            s_df['DTI'] = monthly_pay / total_inc
            s_df['DTI'] = s_df['DTI'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
            
            # Preprocess
            for col in encoder_meta['categorical_cols']:
                le = encoder_meta['label_encoders'][col]
                s_df[col] = le.transform(s_df[col].astype(str))
                
            scale_cols = encoder_meta['scale_cols']
            s_df[scale_cols] = scaler.transform(s_df[scale_cols])
            s_df = s_df[features]
            
            pred = model.predict(s_df)[0]
            prob = model.predict_proba(s_df)[0]
            print(f"{name} -> Prediction: {'Approved' if pred == 1 else 'Rejected'}, Approval Prob: {prob[1]*100:.2f}%, Rejection Prob: {prob[0]*100:.2f}%")

if __name__ == '__main__':
    run_diagnostics()
