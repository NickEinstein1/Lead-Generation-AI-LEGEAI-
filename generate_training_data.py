"""
Generate synthetic training data for insurance lead scoring models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_insurance_leads_data(n_samples=5000):
    """Generate synthetic insurance leads training data"""
    
    data = {
        'lead_id': [f'INS_{i:05d}' for i in range(n_samples)],
        'age': np.random.randint(18, 75, n_samples),
        'income': np.random.lognormal(11, 0.5, n_samples).astype(int),
        'policy_type': np.random.choice(['auto', 'home', 'life', 'health'], n_samples),
        'quote_requests_30d': np.random.poisson(2, n_samples),
        'social_engagement_score': np.random.uniform(0, 10, n_samples),
        'location_risk_score': np.random.uniform(1, 10, n_samples),
        'previous_insurance': np.random.choice(['yes', 'no'], n_samples, p=[0.7, 0.3]),
        'credit_score_proxy': np.random.normal(700, 80, n_samples).astype(int),
        'consent_given': np.random.choice([True, False], n_samples, p=[0.95, 0.05]),
        'consent_timestamp': [
            (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat() 
            for _ in range(n_samples)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Generate conversion score based on features (0-100)
    df['conversion_score'] = (
        (100 - df['age']) * 0.3 +
        (df['income'] / 2000) * 0.2 +
        df['social_engagement_score'] * 5 +
        (10 - df['location_risk_score']) * 3 +
        (df['previous_insurance'] == 'yes').astype(int) * 15 +
        (df['credit_score_proxy'] / 10) * 0.15 +
        df['quote_requests_30d'] * 3 +
        np.random.normal(0, 10, n_samples)
    )
    
    # Clip to 0-100 range
    df['conversion_score'] = df['conversion_score'].clip(0, 100)
    
    return df

def generate_life_insurance_leads_data(n_samples=5000):
    """Generate synthetic life insurance leads training data"""
    
    marital_statuses = ['single', 'married', 'divorced', 'widowed', 'partnered']
    employment_statuses = ['employed', 'unemployed', 'self_employed', 'retired']
    health_statuses = ['excellent', 'good', 'fair', 'poor']
    smoking_statuses = ['smoker', 'non_smoker', 'former_smoker']
    education_levels = ['high_school', 'associates', 'bachelors', 'masters', 'doctorate']
    occupation_risks = ['low', 'medium', 'high', 'very_high']
    life_stages = ['young_professional', 'family_building', 'wealth_accumulation', 'estate_planning']
    
    ages = np.random.randint(18, 85, n_samples)
    
    data = {
        'lead_id': [f'LIFE_{i:05d}' for i in range(n_samples)],
        'age': ages,
        'income': np.random.lognormal(11, 0.6, n_samples).astype(int),
        'marital_status': np.random.choice(marital_statuses, n_samples, p=[0.25, 0.45, 0.15, 0.05, 0.10]),
        'dependents_count': np.random.poisson(1.5, n_samples).clip(0, 10),
        'employment_status': np.random.choice(employment_statuses, n_samples, p=[0.70, 0.05, 0.15, 0.10]),
        'health_status': np.random.choice(health_statuses, n_samples, p=[0.25, 0.50, 0.20, 0.05]),
        'smoking_status': np.random.choice(smoking_statuses, n_samples, p=[0.15, 0.70, 0.15]),
        'coverage_amount_requested': np.random.lognormal(12.5, 0.8, n_samples).astype(int),
        'policy_term': np.random.choice([10, 15, 20, 25, 30], n_samples),
        'existing_life_insurance': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),
        'beneficiary_count': np.random.poisson(1.8, n_samples).clip(1, 10),
        'debt_obligations': np.random.lognormal(10, 1.2, n_samples).clip(0, None).astype(int),
        'mortgage_balance': np.random.lognormal(12, 1.0, n_samples).clip(0, None).astype(int),
        'education_level': np.random.choice(education_levels, n_samples, p=[0.20, 0.15, 0.40, 0.20, 0.05]),
        'occupation_risk_level': np.random.choice(occupation_risks, n_samples, p=[0.60, 0.25, 0.10, 0.05]),
        'financial_dependents': np.random.poisson(1.3, n_samples).clip(0, 10),
        'estate_planning_needs': np.random.randint(0, 11, n_samples),
        'consent_given': np.random.choice([True, False], n_samples, p=[0.95, 0.05]),
        'consent_timestamp': [
            (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat() 
            for _ in range(n_samples)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Determine life stage based on age
    df['life_stage'] = pd.cut(ages, bins=[0, 30, 45, 65, 100], 
                               labels=life_stages, 
                               include_lowest=True).astype(str)
    
    # Generate conversion score based on life insurance-specific features
    health_score_map = {'excellent': 20, 'good': 15, 'fair': 10, 'poor': 5}
    smoking_score_map = {'non_smoker': 15, 'former_smoker': 10, 'smoker': 0}
    employment_score_map = {'employed': 20, 'self_employed': 15, 'retired': 10, 'unemployed': 0}
    
    df['conversion_score'] = (
        ((85 - df['age']) / 85 * 20) +  # Younger is better for term life
        (df['income'] / 3000) * 0.15 +
        df['health_status'].map(health_score_map) +
        df['smoking_status'].map(smoking_score_map) +
        df['employment_status'].map(employment_score_map) +
        (df['dependents_count'] * 3).clip(0, 15) +  # More dependents = higher need
        ((df['coverage_amount_requested'] / 100000) * 2).clip(0, 10) +
        (~df['existing_life_insurance']).astype(int) * 10 +
        (df['estate_planning_needs'] * 1.5) +
        np.random.normal(0, 8, n_samples)
    )
    
    # Clip to 0-100 range
    df['conversion_score'] = df['conversion_score'].clip(0, 100)
    
    return df

if __name__ == "__main__":
    print("Generating insurance leads training data...")
    insurance_df = generate_insurance_leads_data(5000)
    insurance_df.to_csv('data/insurance_leads_training.csv', index=False)
    print(f"✓ Generated {len(insurance_df)} insurance leads records")
    print(f"  - Conversion score range: {insurance_df['conversion_score'].min():.1f} - {insurance_df['conversion_score'].max():.1f}")
    print(f"  - Mean conversion score: {insurance_df['conversion_score'].mean():.1f}")
    
    print("\nGenerating life insurance leads training data...")
    life_insurance_df = generate_life_insurance_leads_data(5000)
    life_insurance_df.to_csv('data/life_insurance_leads_training.csv', index=False)
    print(f"✓ Generated {len(life_insurance_df)} life insurance leads records")
    print(f"  - Conversion score range: {life_insurance_df['conversion_score'].min():.1f} - {life_insurance_df['conversion_score'].max():.1f}")
    print(f"  - Mean conversion score: {life_insurance_df['conversion_score'].mean():.1f}")
    
    print("\n✅ Training data generation complete!")

