import os
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#cccccc',
    'font.family': 'sans-serif',
    'font.size': 11
})

PRIMARY_PURPLE = '#6d28d9'
SECONDARY_PURPLE = '#8b5cf6'
DARK_PURPLE = '#4c1d95'
LIGHT_PURPLE = '#ddd6fe'
ACCENT_GREEN = '#10b981'
ACCENT_RED = '#ef4444'

PALETTE_MUTED = [PRIMARY_PURPLE, '#a78bfa', '#c084fc', '#f472b6', '#3b82f6']

def run_eda(df, output_dir):
    """
    Perform Exploratory Data Analysis and save plots to output_dir.
    """
    print("--- Starting Exploratory Data Analysis (EDA) ---")
    print(f"Dataset Shape: {df.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Count Plots
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Gender', hue='Loan_Status', palette=[ACCENT_RED, PRIMARY_PURPLE])
    plt.title('Loan Status by Gender', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_vs_loan_status.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Education', hue='Loan_Status', palette=[ACCENT_RED, PRIMARY_PURPLE])
    plt.title('Loan Status by Education', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'education_vs_loan_status.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Property_Area', hue='Loan_Status', palette=[ACCENT_RED, PRIMARY_PURPLE])
    plt.title('Loan Status by Property Area', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'property_area_vs_loan_status.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Credit_History', hue='Loan_Status', palette=[ACCENT_RED, PRIMARY_PURPLE])
    plt.title('Loan Status by Credit History', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'credit_history_vs_loan_status.png'), dpi=150)
    plt.close()
    
    # 2. Distribution Plots
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='ApplicantIncome', kde=True, color=PRIMARY_PURPLE)
    plt.title('Applicant Income Distribution', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'applicant_income_dist.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='CoapplicantIncome', kde=True, color=SECONDARY_PURPLE)
    plt.title('Coapplicant Income Distribution', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coapplicant_income_dist.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='LoanAmount', kde=True, color=DARK_PURPLE)
    plt.title('Loan Amount Distribution', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loan_amount_dist.png'), dpi=150)
    plt.close()
    
    # 3. Bar Charts
    def calculate_approval_rate(df, col):
        temp = df.groupby(col)['Loan_Status'].value_counts(normalize=True).unstack() * 100
        return temp['Y'] if 'Y' in temp.columns else pd.Series(0, index=temp.index)
        
    plt.figure(figsize=(6, 4))
    rate_edu = calculate_approval_rate(df, 'Education')
    sns.barplot(x=rate_edu.index, y=rate_edu.values, palette=[PRIMARY_PURPLE, SECONDARY_PURPLE])
    plt.ylabel('Approval Rate (%)')
    plt.title('Loan Approval Rate by Education', fontsize=12, fontweight='bold', pad=10)
    plt.ylim(0, 100)
    for index, val in enumerate(rate_edu.values):
        plt.text(index, val + 2, f"{val:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'approval_rate_education.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(6, 4))
    rate_prop = calculate_approval_rate(df, 'Property_Area')
    sns.barplot(x=rate_prop.index, y=rate_prop.values, palette=PALETTE_MUTED[:3])
    plt.ylabel('Approval Rate (%)')
    plt.title('Loan Approval Rate by Property Area', fontsize=12, fontweight='bold', pad=10)
    plt.ylim(0, 100)
    for index, val in enumerate(rate_prop.values):
        plt.text(index, val + 2, f"{val:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'approval_rate_property.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(6, 4))
    rate_gender = calculate_approval_rate(df, 'Gender')
    sns.barplot(x=rate_gender.index, y=rate_gender.values, palette=[PRIMARY_PURPLE, SECONDARY_PURPLE])
    plt.ylabel('Approval Rate (%)')
    plt.title('Loan Approval Rate by Gender', fontsize=12, fontweight='bold', pad=10)
    plt.ylim(0, 100)
    for index, val in enumerate(rate_gender.values):
        plt.text(index, val + 2, f"{val:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'approval_rate_gender.png'), dpi=150)
    plt.close()
    
    # 4. Correlation Heatmap
    temp_df = df.copy()
    for col in temp_df.select_dtypes(include=['object']).columns:
        if col != 'Loan_ID':
            temp_df[col] = LabelEncoder().fit_transform(temp_df[col].astype(str))
            
    numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Loan_Status' in temp_df.columns:
        temp_df['Loan_Status'] = temp_df['Loan_Status'].map({'Y': 1, 'N': 0})
        numeric_cols.append('Loan_Status')
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(temp_df[numeric_cols].corr(), annot=True, cmap='Purples', fmt='.2f', linewidths=0.5, cbar=True)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    print("EDA Visualizations successfully saved in:", output_dir)

def preprocess_and_save_pipeline(df, models_dir):
    """
    Impute, encode, scale, balance, and serialize the encoders.
    Also compile dataset analytics stats.
    """
    print("\n--- Starting Data Preprocessing Pipeline ---")
    
    # Save raw dataset stats before processing
    total_records = len(df)
    total_features = df.shape[1] - 2  # Exclude Loan_ID and Loan_Status
    missing_vals_count = df.isnull().sum().sum()
    
    status_counts = df['Loan_Status'].value_counts()
    approval_rate = (status_counts.get('Y', 0) / total_records) * 100
    rejection_rate = (status_counts.get('N', 0) / total_records) * 100
    
    dataset_stats = {
        'total_records': total_records,
        'total_features': total_features,
        'missing_values': missing_vals_count,
        'approval_rate': round(approval_rate, 2),
        'rejection_rate': round(rejection_rate, 2),
        'split_ratio': '80 / 20'
    }
    
    # Serialize stats
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'dataset_stats.pkl'), 'wb') as f:
        pickle.dump(dataset_stats, f)
        
    print("Dataset stats successfully saved:", dataset_stats)

    df = df.copy()
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
        
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'TotalIncome', 'EMI', 'DTI', 'LTI', 'IncomeStabilityScore']
    
    # Missing values mean/mode imputer lists
    modes = {}
    for col in categorical_cols + ['Credit_History']:
        mode_val = df[col].mode()[0]
        modes[col] = mode_val
        df[col] = df[col].fillna(mode_val)
        
    means = {}
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
        mean_val = df[col].mean()
        means[col] = mean_val
        df[col] = df[col].fillna(mean_val)
        
    # Feature Engineering: 5 strong financial metrics
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
    
    # Store means for the engineered features
    for col in ['TotalIncome', 'EMI', 'DTI', 'LTI', 'IncomeStabilityScore']:
        means[col] = df[col].mean()
        
    # Label encode
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    
    # Scaling (all continuous columns including engineered ones)
    scale_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'TotalIncome', 'EMI', 'DTI', 'LTI', 'IncomeStabilityScore']
    scaler = StandardScaler()
    
    X_scaled_part = scaler.fit_transform(X[scale_cols])
    X_scaled = X.copy()
    X_scaled[scale_cols] = X_scaled_part
    
    # Balances using SMOTE (Save params)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    encoder_meta = {
        'label_encoders': label_encoders,
        'modes': modes,
        'means': means,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'scale_cols': scale_cols,
        'features': X.columns.tolist()
    }
    
    with open(os.path.join(models_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder_meta, f)
        
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Preprocessing components serialized.")
    return X_resampled, y_resampled, X_scaled, y

def save_xgb_visualizations(xgb_model, X_test, y_test, features_list, output_dir, models_dir):
    """
    Generate Confusion Matrix plot and Feature Importance plot for XGBoost.
    Save plots to output_dir, and save metric values to models_dir.
    """
    # 1. Predict with XGBoost
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1] if hasattr(xgb_model, 'predict_proba') else None
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title('XGBoost Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Class', fontsize=10, labelpad=8)
    plt.ylabel('True Class', fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_xgboost.png'), dpi=150)
    plt.close()
    
    # Save CM details
    cm_dict = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    with open(os.path.join(models_dir, 'confusion_matrix_xgboost.pkl'), 'wb') as f:
        pickle.dump(cm_dict, f)
        
    # 3. Feature Importance
    importances = xgb_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': features_list,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', color=PRIMARY_PURPLE)
    plt.title('XGBoost Feature Importance', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Relative Importance Score', fontsize=10)
    plt.ylabel('Features', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_xgboost.png'), dpi=150)
    plt.close()
    
    # Save Feature Importance list
    feat_list = feat_imp_df.to_dict(orient='records')
    with open(os.path.join(models_dir, 'feature_importance_xgboost.pkl'), 'wb') as f:
        pickle.dump(feat_list, f)
        
    # 4. Classification Report
    cr = classification_report(y_test, y_pred, output_dict=True)
    with open(os.path.join(models_dir, 'classification_report_xgboost.pkl'), 'wb') as f:
        pickle.dump(cr, f)
        
    print("XGBoost specific analytics generated and saved.")

def train_and_evaluate_models(X_train_res, y_train_res, X_train, y_train, X_test, y_test, features_list, models_dir, eda_dir):
    """
    Train models, measure durations, compare training/testing metrics,
    and save the selected XGBoost model.
    """
    print("\n--- Starting Model Training & Evaluation ---")
    
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    comparison_data = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train_res, y_train_res)
        fit_duration = time.time() - start_time
        
        trained_models[name] = model
        
        # Train Predictions
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        # Test Predictions
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        # Cross Validation Score (using SMOTE balanced training data to estimate)
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        
        comparison_data.append({
            'Model': name,
            'Train Accuracy': f"{train_acc * 100:.2f}%",
            'Test Accuracy': f"{test_acc * 100:.2f}%",
            'Precision': f"{precision * 100:.2f}%",
            'Recall': f"{recall * 100:.2f}%",
            'F1 Score': f"{f1 * 100:.2f}%",
            'CV Score': f"{cv_mean * 100:.2f}%",
            'Training Time': f"{fit_duration:.4f}s"
        })
        print(f"Accuracy: {test_acc:.4f} | F1: {f1:.4f} | Time: {fit_duration:.4f}s")
        
    df_compare = pd.DataFrame(comparison_data)
    print("\n--- Model Performance Comparison Table ---")
    print(df_compare.to_markdown(index=False))
    
    # Save the comparative table metrics
    with open(os.path.join(models_dir, 'metrics_comparison.pkl'), 'wb') as f:
        pickle.dump(comparison_data, f)
        
    # Generate XGBoost Specific Analytics (since XGBoost is selected as production model)
    best_model_name = 'XGBoost'
    best_model_obj = trained_models['XGBoost']
    
    save_xgb_visualizations(best_model_obj, X_test, y_test, features_list, eda_dir, models_dir)
    
    # Save best model info
    best_test_acc = accuracy_score(y_test, best_model_obj.predict(X_test))
    with open(os.path.join(models_dir, 'best_model_info.pkl'), 'wb') as f:
        pickle.dump({
            'model_name': best_model_name,
            'accuracy': best_test_acc
        }, f)
        
    with open(os.path.join(models_dir, 'loan_model.pkl'), 'wb') as f:
        pickle.dump(best_model_obj, f)
        
    print(f"Production model (XGBoost) successfully serialized.")

def main():
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset/loan_data.csv')
    eda_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../frontend/static/images/eda/')
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Run generate_data.py first.")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Phase 2: EDA
    run_eda(df, eda_output_dir)
    
    # Phase 3: Preprocessing
    X_resampled, y_resampled, X_scaled, y = preprocess_and_save_pipeline(df, models_dir)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Balance only the train split for modeling
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Train, evaluate, and save
    features_list = X_scaled.columns.tolist()
    train_and_evaluate_models(
        X_train_res, y_train_res, X_train, y_train, X_test, y_test,
        features_list, models_dir, eda_output_dir
    )

if __name__ == '__main__':
    main()
