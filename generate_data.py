import os
import numpy as np
import pandas as pd

def generate_loan_dataset(output_path, num_samples=600, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Generate core columns
    genders = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.80, 0.20])
    married = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.65, 0.35])
    dependents = np.random.choice(['0', '1', '2', '3+'], size=num_samples, p=[0.57, 0.17, 0.16, 0.10])
    education = np.random.choice(['Graduate', 'Not Graduate'], size=num_samples, p=[0.78, 0.22])
    self_employed = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.14, 0.86])
    
    # ApplicantIncome: log-normal distribution to resemble real income (monthly)
    applicant_income = np.random.lognormal(mean=8.4, sigma=0.6, size=num_samples).astype(int)
    applicant_income = np.clip(applicant_income, 1500, 81000)
    
    # CoapplicantIncome: ~45% have 0 co-applicant income, others have log-normal
    has_coapplicant = np.random.choice([0, 1], size=num_samples, p=[0.45, 0.55])
    coapplicant_income = np.zeros(num_samples)
    coapplicant_income_values = np.random.lognormal(mean=7.3, sigma=0.6, size=num_samples).astype(int)
    coapplicant_income_values = np.clip(coapplicant_income_values, 1000, 41000)
    coapplicant_income[has_coapplicant == 1] = coapplicant_income_values[has_coapplicant == 1]
    
    # LoanAmount: proportional to total income with random multipliers to generate variable DTI ratios, in thousands
    total_income = applicant_income + coapplicant_income
    multipliers = np.random.uniform(0.02, 0.16, size=num_samples)
    loan_amount = (total_income * multipliers + np.random.normal(10, 15, size=num_samples)).astype(int)
    loan_amount = np.clip(loan_amount, 9, 600)
    
    # Loan_Amount_Term
    loan_amount_term = np.random.choice(
        [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], 
        size=num_samples, 
        p=[0.01, 0.01, 0.01, 0.01, 0.01, 0.07, 0.01, 0.02, 0.84, 0.01]
    )
    
    # Credit_History: binary 1.0 or 0.0, heavily biased towards 1.0
    credit_history = np.random.choice([1.0, 0.0], size=num_samples, p=[0.84, 0.16])
    
    # Property_Area: Urban, Semiurban, Rural
    property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], size=num_samples, p=[0.33, 0.38, 0.29])
    
    # 2. Determine Loan_Status probabilistically to prevent Credit_History dominance
    loan_status = []
    dep_map = {'0': 1.0, '1': 0.95, '2': 0.9, '3+': 0.8}
    for i in range(num_samples):
        # Base probability of approval
        p = 0.45
        
        # Credit history effect: strong but not absolute
        if credit_history[i] == 1.0:
            p += 0.15
        else:
            p -= 0.25
            
        # Debt to Income check: Loan amount * 1000 vs total income.
        term = loan_amount_term[i] if loan_amount_term[i] > 0 else 360
        monthly_payment = (loan_amount[i] * 1000) / term
        monthly_income = total_income[i] 
        dti = monthly_payment / monthly_income if monthly_income > 0 else 1.0
        
        # Realistic DTI boundaries
        if dti > 0.45:
            p -= 0.35
        elif dti > 0.35:
            p -= 0.20
        elif dti < 0.20:
            p += 0.15
            
        # Loan-to-Income (LTI) ratio check
        lti = (loan_amount[i] * 1000) / total_income[i] if total_income[i] > 0 else 10.0
        if lti > 6.0:
            p -= 0.25
        elif lti > 4.5:
            p -= 0.15
        elif lti < 3.0:
            p += 0.10
            
        # Income Stability calculation
        edu_val = 1.2 if education[i] == 'Graduate' else 0.9
        emp_val = 0.8 if self_employed[i] == 'Yes' else 1.1
        mar_val = 1.15 if married[i] == 'Yes' else 0.95
        dep_val = dep_map.get(dependents[i], 1.0)
        stability = edu_val * emp_val * mar_val * dep_val
        
        if stability > 1.2:
            p += 0.15
        elif stability < 0.9:
            p -= 0.15
            
        # Joint income scale effect
        if total_income[i] < 3000:
            p -= 0.25
        elif total_income[i] > 15000:
            p += 0.20
            
        # Property Area effect
        if property_area[i] == 'Semiurban':
            p += 0.05
        elif property_area[i] == 'Rural':
            p -= 0.08
            
        # Add random normal noise to probability
        p += np.random.normal(0, 0.05)
        
        # Clamp probability
        p = np.clip(p, 0.01, 0.99)
        
        # Threshold decision
        if p >= 0.50:
            loan_status.append('Y')
        else:
            loan_status.append('N')
            
    loan_status = np.array(loan_status)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Loan_ID': [f'LP00{1000+i}' for i in range(num_samples)],
        'Gender': genders,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
        'Loan_Status': loan_status
    })
    
    # 3. Introduce missing values (NaN) to test our imputation pipeline
    mask_gender = np.random.rand(num_samples) < 0.025
    df.loc[mask_gender, 'Gender'] = np.nan
    
    mask_married = np.random.rand(num_samples) < 0.015
    df.loc[mask_married, 'Married'] = np.nan
    
    mask_dependents = np.random.rand(num_samples) < 0.03
    df.loc[mask_dependents, 'Dependents'] = np.nan
    
    mask_self_employed = np.random.rand(num_samples) < 0.05
    df.loc[mask_self_employed, 'Self_Employed'] = np.nan
    
    mask_loan_amt = np.random.rand(num_samples) < 0.035
    df.loc[mask_loan_amt, 'LoanAmount'] = np.nan
    
    mask_term = np.random.rand(num_samples) < 0.02
    df.loc[mask_term, 'Loan_Amount_Term'] = np.nan
    
    mask_credit = np.random.rand(num_samples) < 0.08
    df.loc[mask_credit, 'Credit_History'] = np.nan
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic loan dataset with {num_samples} samples at {output_path}")
    print(f"Approval rate: {np.mean(loan_status == 'Y') * 100:.2f}%")

if __name__ == '__main__':
    generate_loan_dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset/loan_data.csv'))
