import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app
import json

def run_test():
    # Case 1: Extreme risk (Income = 1000, Loan Amount = 500, Credit History = Yes, Self Employed = Yes)
    case_1 = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'Yes',
        'ApplicantIncome': '1000',
        'CoapplicantIncome': '0',
        'LoanAmount': '500',
        'Loan_Amount_Term': '360',
        'Credit_History': '1.0',
        'Property_Area': 'Semiurban'
    }
    
    # Case 2: Ideal borrower (Income = 100000, Loan Amount = 50, Credit History = Yes, Self Employed = No)
    case_2 = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': '100000',
        'CoapplicantIncome': '0',
        'LoanAmount': '50',
        'Loan_Amount_Term': '360',
        'Credit_History': '1.0',
        'Property_Area': 'Semiurban'
    }
    
    with app.test_client() as client:
        print("=================================================================")
        print("RUNNING PIPELINE TEST FOR CASE 1 (Income = 1000, Loan = 500)")
        print("=================================================================")
        response = client.post('/predict', data=case_1, follow_redirects=True)
        from flask import session
        assert response.status_code == 200
        result = session.get('prediction_result')
        print(f"Outcome: {'APPROVED' if result['approved'] else 'REJECTED'}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Approval Probability: {result['probability']}%")
        print("Key Factors:")
        for factor in result['key_factors']:
            print(f"  - {factor}")
        assert result['approved'] is False, "Case 1 should be Rejected"
        
        print("\n" + "="*65)
        print("RUNNING PIPELINE TEST FOR CASE 2 (Income = 100000, Loan = 50)")
        print("=================================================================")
        response = client.post('/predict', data=case_2, follow_redirects=True)
        assert response.status_code == 200
        result = session.get('prediction_result')
        print(f"Outcome: {'APPROVED' if result['approved'] else 'REJECTED'}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Approval Probability: {result['probability']}%")
        print("Key Factors:")
        for factor in result['key_factors']:
            print(f"  - {factor}")
        assert result['approved'] is True, "Case 2 should be Approved"

if __name__ == '__main__':
    run_test()


