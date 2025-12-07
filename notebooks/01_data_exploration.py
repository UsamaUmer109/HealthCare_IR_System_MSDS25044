"""
Step 1: Explore and understand your healthcare datasets
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def explore_dataset(file_path, dataset_name):
    """Explore a dataset and print basic information"""
    print(f"\n{'='*60}")
    print(f"EXPLORING: {dataset_name}")
    print(f"File: {file_path}")
    print('='*60)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Basic information
        print(f"Shape: {df.shape} (rows, columns)")
        print(f"Columns: {list(df.columns)}")
        
        # Data types
        print("\nData Types:")
        print(df.dtypes)
        
        # Missing values
        print("\nMissing Values:")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # First few rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Unique values in first few columns
        print(f"\nUnique counts for important columns:")
        for col in df.columns[:min(5, len(df.columns))]:  # First 5 columns
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count < 10:
                print(f"    Values: {df[col].unique().tolist()}")
        
        return df
    
    except Exception as e:
        print(f" Error loading file: {e}")
        return None

def analyze_symptom_dataset(df):
    """Special analysis for symptom dataset"""
    if df is None:
        return
    
    print(f"\n{'='*60}")
    print("SPECIAL ANALYSIS: Symptom Dataset")
    print('='*60)
    
    # Get all symptom columns
    symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
    print(f"Found {len(symptom_cols)} symptom columns")
    
    # Count non-empty symptoms per disease
    symptom_counts = []
    all_symptoms = []
    
    for idx, row in df.iterrows():
        count = 0
        for col in symptom_cols:
            if pd.notna(row[col]) and str(row[col]).strip():
                count += 1
                symptom_text = str(row[col]).strip()
                all_symptoms.append(symptom_text)
        symptom_counts.append(count)
    
    df['symptom_count'] = symptom_counts
    
    print(f"\nSymptoms per disease:")
    print(f"  Min: {min(symptom_counts)}")
    print(f"  Max: {max(symptom_counts)}")
    print(f"  Average: {np.mean(symptom_counts):.2f}")
    print(f"  Std Dev: {np.std(symptom_counts):.2f}")
    
    # Count unique symptoms
    unique_symptoms = set(all_symptoms)
    print(f"\nTotal symptom mentions: {len(all_symptoms)}")
    print(f"Unique symptoms: {len(unique_symptoms)}")
    
    # Most common symptoms
    from collections import Counter
    symptom_counter = Counter(all_symptoms)
    print(f"\nTop 10 most common symptoms:")
    for symptom, count in symptom_counter.most_common(10):
        print(f"  {symptom}: {count} times")
    
    return df

def analyze_precaution_dataset(df):
    """Special analysis for precaution dataset"""
    if df is None:
        return
    
    print(f"\n{'='*60}")
    print("SPECIAL ANALYSIS: Precaution Dataset")
    print('='*60)
    
    # Get all precaution columns
    precaution_cols = [col for col in df.columns if col.startswith('Precaution_')]
    print(f"Found {len(precaution_cols)} precaution columns")
    
    # Count non-empty precautions per disease
    precaution_counts = []
    all_precautions = []
    
    for idx, row in df.iterrows():
        count = 0
        for col in precaution_cols:
            if pd.notna(row[col]) and str(row[col]).strip():
                count += 1
                precaution_text = str(row[col]).strip()
                all_precautions.append(precaution_text)
        precaution_counts.append(count)
    
    df['precaution_count'] = precaution_counts
    
    print(f"\nPrecautions per disease:")
    print(f"  Min: {min(precaution_counts)}")
    print(f"  Max: {max(precaution_counts)}")
    print(f"  Average: {np.mean(precaution_counts):.2f}")
    
    # Count unique precautions
    unique_precautions = set(all_precautions)
    print(f"\nTotal precaution mentions: {len(all_precautions)}")
    print(f"Unique precautions: {len(unique_precautions)}")
    
    return df

def compare_diseases(symptom_df, precaution_df):
    """Compare diseases in both datasets"""
    print(f"\n{'='*60}")
    print("COMPARING DISEASES IN BOTH DATASETS")
    print('='*60)
    
    if symptom_df is None or precaution_df is None:
        return
    
    # Get disease lists
    symptom_diseases = set(symptom_df['Disease'].str.strip().str.lower())
    precaution_diseases = set(precaution_df['Disease'].str.strip().str.lower())
    
    print(f"Diseases in symptom dataset: {len(symptom_diseases)}")
    print(f"Diseases in precaution dataset: {len(precaution_diseases)}")
    
    # Find common diseases
    common_diseases = symptom_diseases.intersection(precaution_diseases)
    print(f"Diseases in both datasets: {len(common_diseases)}")
    
    # Diseases only in symptoms
    only_in_symptoms = symptom_diseases - precaution_diseases
    print(f"Diseases only in symptom dataset: {len(only_in_symptoms)}")
    
    # Diseases only in precautions
    only_in_precautions = precaution_diseases - symptom_diseases
    print(f"Diseases only in precaution dataset: {len(only_in_precautions)}")
    
    # Show samples
    if common_diseases:
        print(f"\nSample of common diseases (first 10):")
        for disease in list(common_diseases)[:10]:
            print(f"  - {disease.title()}")
    
    if only_in_symptoms:
        print(f"\nSample diseases only in symptom dataset:")
        for disease in list(only_in_symptoms)[:5]:
            print(f"  - {disease.title()}")
    
    return common_diseases, only_in_symptoms, only_in_precautions

def save_analysis_report(symptom_df, precaution_df, common_diseases):
    """Save analysis to a text file"""
    print(f"\n{'='*60}")
    print("SAVING ANALYSIS REPORT")
    print('='*60)
    
    report = f"""HEALTHCARE DATASET ANALYSIS REPORT
{'='*60}

DATASET SUMMARY:
{'='*60}

1. SYMPTOM DATASET:
   - Total diseases: {len(symptom_df)}
   - Columns: {list(symptom_df.columns)}
   - Shape: {symptom_df.shape}
   - Average symptoms per disease: {symptom_df['symptom_count'].mean():.2f}

2. PRECAUTION DATASET:
   - Total diseases: {len(precaution_df)}
   - Columns: {list(precaution_df.columns)}
   - Shape: {precaution_df.shape}
   - Average precautions per disease: {precaution_df['precaution_count'].mean():.2f}

3. COMPARISON:
   - Diseases in both datasets: {len(common_diseases)}
   - Diseases only in symptoms: {len(set(symptom_df['Disease'].str.strip().str.lower()) - common_diseases)}
   - Diseases only in precautions: {len(set(precaution_df['Disease'].str.strip().str.lower()) - common_diseases)}

SAMPLE DATA:
{'='*60}

First 5 diseases from symptom dataset:
"""
    
    for i in range(min(5, len(symptom_df))):
        disease = symptom_df['Disease'].iloc[i]
        symptoms = []
        for col in symptom_df.columns:
            if col.startswith('Symptom_') and pd.notna(symptom_df[col].iloc[i]):
                symptoms.append(str(symptom_df[col].iloc[i]).strip())
        
        report += f"\n{i+1}. {disease}"
        report += f"\n   Symptoms: {', '.join(symptoms[:3])}"
        if len(symptoms) > 3:
            report += f"... (+{len(symptoms)-3} more)"
        report += f"\n   Total symptoms: {len(symptoms)}"
    
    # Save to file
    os.makedirs('data/processed', exist_ok=True)
    report_path = 'data/processed/data_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Analysis report saved to: {report_path}")
    return report_path

def main():
    """Main exploration function"""
    print("🏥 HEALTHCARE DATASET EXPLORATION")
    print("="*60)
    
    # File paths
    symptom_path = 'data/raw/DiseaseAndSymptoms.csv'
    precaution_path = 'data/raw/DiseasePrecautions.csv'
    
    # Explore symptom dataset
    symptom_df = explore_dataset(symptom_path, "Disease and Symptoms")
    if symptom_df is not None:
        symptom_df = analyze_symptom_dataset(symptom_df)
    
    # Explore precaution dataset
    precaution_df = explore_dataset(precaution_path, "Disease and Precautions")
    if precaution_df is not None:
        precaution_df = analyze_precaution_dataset(precaution_df)
    
    # Compare datasets
    if symptom_df is not None and precaution_df is not None:
        common_diseases, only_in_symptoms, only_in_precautions = compare_diseases(
            symptom_df, precaution_df
        )
        
        # Save report
        report_path = save_analysis_report(symptom_df, precaution_df, common_diseases)
        
        print(f"\n{'='*60}")
        print("EXPLORATION COMPLETE!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Check the report at: {report_path}")
        print(f"2. We'll create preprocessing pipeline based on this analysis")
    
    else:
        print("\nCould not load one or both datasets.")
        print("Please check file paths and try again.")

if __name__ == "__main__":
    main()