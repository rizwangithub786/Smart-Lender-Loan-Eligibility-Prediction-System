import os
import pickle
import pandas as pd
import numpy as np

def load_ml_components():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
    model_path = os.path.join(models_dir, 'loan_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    encoder_path = os.path.join(models_dir, 'encoder.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder_meta = pickle.load(f)
        
    return model, scaler, encoder_meta

def preprocess_profile(profile, scaler, encoder_meta):
    df = pd.DataFrame([profile])
    
    # Feature Engineering
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    
    df['EMI'] = (df['LoanAmount'] * 1000) / df['Loan_Amount_Term']
    df['EMI'] = df['EMI'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    df['DTI'] = df['EMI'] / df['TotalIncome']
    df['DTI'] = df['DTI'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    df['LTI'] = (df['LoanAmount'] * 1000) / df['TotalIncome']
    df['LTI'] = df['LTI'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    dep_map = {'0': 1.0, '1': 0.95, '2': 0.9, '3+': 0.8}
    edu_val = df['Education'].map({'Graduate': 1.2, 'Not Graduate': 0.9}).fillna(0.9)
    emp_val = df['Self_Employed'].map({'Yes': 0.8, 'No': 1.1}).fillna(1.1)
    mar_val = df['Married'].map({'Yes': 1.15, 'No': 0.95}).fillna(0.95)
    dep_val = df['Dependents'].map(dep_map).fillna(1.0)
    df['IncomeStabilityScore'] = edu_val * emp_val * mar_val * dep_val
    
    # Encode Categoricals
    for col in encoder_meta['categorical_cols']:
        le = encoder_meta['label_encoders'][col]
        df[col] = le.transform(df[col].astype(str))
        
    # Scale continuous
    scale_cols = encoder_meta['scale_cols']
    df[scale_cols] = scaler.transform(df[scale_cols])
    
    # Reorder
    df = df[encoder_meta['features']]
    return df

def run_sensitivity():
    model, scaler, encoder_meta = load_ml_components()
    
    # Base profiles
    case_a = {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate', 
        'Self_Employed': 'Yes', 'ApplicantIncome': 1000.0, 'CoapplicantIncome': 0.0, 
        'LoanAmount': 500.0, 'Loan_Amount_Term': 360.0, 'Credit_History': 1.0, 'Property_Area': 'Semiurban'
    }
    
    case_b = {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate', 
        'Self_Employed': 'No', 'ApplicantIncome': 100000.0, 'CoapplicantIncome': 0.0, 
        'LoanAmount': 50.0, 'Loan_Amount_Term': 360.0, 'Credit_History': 1.0, 'Property_Area': 'Semiurban'
    }
    
    print("=== 1. Sensitivity Analysis: Case A vs Case B ===")
    for name, case in [('Case A (Low Income, High Loan, Credit=1)', case_a), ('Case B (High Income, Low Loan, Credit=1)', case_b)]:
        vec = preprocess_profile(case, scaler, encoder_meta)
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        print(f"\n{name}:")
        print(f"  Decision: {'Approved' if pred == 1 else 'Rejected'}")
        print(f"  Approval Probability: {prob[1]*100:.4f}%")
        print(f"  Rejection Probability: {prob[0]*100:.4f}%")
        print("  Processed Feature Vector:")
        for col in vec.columns:
            print(f"    {col}: {vec[col].iloc[0]:.4f}")
            
    print("\n=== 2. Single-Parameter Sweeps (Base profile: Case B) ===")
    base_profile = case_b.copy()
    
    print("\n--- Sweeping ApplicantIncome ---")
    for inc in [1000.0, 3000.0, 5000.0, 10000.0, 100000.0]:
        profile = base_profile.copy()
        profile['ApplicantIncome'] = inc
        vec = preprocess_profile(profile, scaler, encoder_meta)
        prob = model.predict_proba(vec)[0]
        print(f"  Income: {inc:8.1f} -> Approval Prob: {prob[1]*100:6.2f}% | DTI: {vec['DTI'].iloc[0]:.4f}")
        
    print("\n--- Sweeping LoanAmount (Income=5000) ---")
    for amt in [10.0, 50.0, 100.0, 150.0, 300.0, 500.0]:
        profile = base_profile.copy()
        profile['ApplicantIncome'] = 5000.0
        profile['LoanAmount'] = amt
        vec = preprocess_profile(profile, scaler, encoder_meta)
        prob = model.predict_proba(vec)[0]
        print(f"  Loan Amount: {amt:5.1f} -> Approval Prob: {prob[1]*100:6.2f}% | DTI: {vec['DTI'].iloc[0]:.4f}")

    print("\n--- Sweeping Education (Income=3000, Loan=150) ---")
    for edu in ['Graduate', 'Not Graduate']:
        profile = base_profile.copy()
        profile['ApplicantIncome'] = 3000.0
        profile['LoanAmount'] = 150.0
        profile['Education'] = edu
        vec = preprocess_profile(profile, scaler, encoder_meta)
        prob = model.predict_proba(vec)[0]
        print(f"  Education: {edu:15} -> Approval Prob: {prob[1]*100:6.2f}%")

    print("\n--- Sweeping Property Area (Income=3000, Loan=150) ---")
    for area in ['Urban', 'Semiurban', 'Rural']:
        profile = base_profile.copy()
        profile['ApplicantIncome'] = 3000.0
        profile['LoanAmount'] = 150.0
        profile['Property_Area'] = area
        vec = preprocess_profile(profile, scaler, encoder_meta)
        prob = model.predict_proba(vec)[0]
        print(f"  Property Area: {area:12} -> Approval Prob: {prob[1]*100:6.2f}%")

    print("\n--- Sweeping Dependents (Income=3000, Loan=150) ---")
    for deps in ['0', '1', '2', '3+']:
        profile = base_profile.copy()
        profile['ApplicantIncome'] = 3000.0
        profile['LoanAmount'] = 150.0
        profile['Dependents'] = deps
        vec = preprocess_profile(profile, scaler, encoder_meta)
        prob = model.predict_proba(vec)[0]
        print(f"  Dependents: {deps:5} -> Approval Prob: {prob[1]*100:6.2f}%")
        
    print("\n=== 3. Credit History Boundary Tests ===")
    print("\nCan a profile with Credit_History = 0.0 be approved?")
    # Try a hyper-strong profile with Credit_History = 0.0
    super_profile = {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate', 
        'Self_Employed': 'No', 'ApplicantIncome': 500000.0, 'CoapplicantIncome': 500000.0, 
        'LoanAmount': 5.0, 'Loan_Amount_Term': 360.0, 'Credit_History': 0.0, 'Property_Area': 'Semiurban'
    }
    vec = preprocess_profile(super_profile, scaler, encoder_meta)
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    print("  Super Profile (Credit_History = 0):")
    print(f"    Decision: {'Approved' if pred == 1 else 'Rejected'}")
    print(f"    Approval Prob: {prob[1]*100:.4f}%")
    print(f"    Rejection Prob: {prob[0]*100:.4f}%")
    
    print("\nCan a profile with Credit_History = 1.0 be rejected?")
    # Try a profile with Credit_History = 1.0 but bad metrics
    bad_profile = {
        'Gender': 'Male', 'Married': 'No', 'Dependents': '3+', 'Education': 'Not Graduate', 
        'Self_Employed': 'Yes', 'ApplicantIncome': 1000.0, 'CoapplicantIncome': 0.0, 
        'LoanAmount': 600.0, 'Loan_Amount_Term': 120.0, 'Credit_History': 1.0, 'Property_Area': 'Rural'
    }
    vec = preprocess_profile(bad_profile, scaler, encoder_meta)
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    print("  Weak Profile (Credit_History = 1):")
    print(f"    Decision: {'Approved' if pred == 1 else 'Rejected'}")
    print(f"    Approval Prob: {prob[1]*100:.4f}%")
    print(f"    Rejection Prob: {prob[0]*100:.4f}%")

if __name__ == '__main__':
    run_sensitivity()
