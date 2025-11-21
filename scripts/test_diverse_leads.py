"""
Test Auto Insurance Ensemble with Diverse Lead Profiles
"""

import requests
import json
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000/v1/auto-insurance/score-lead"

# Diverse test leads with different risk profiles
test_leads = [
    {
        "name": "Premium Lead - Young Professional",
        "lead_id": "AUTO_PREMIUM_001",
        "age": 32,
        "income": 95000,
        "policy_type": "auto",
        "quote_requests_30d": 2,
        "social_engagement_score": 9.2,
        "location_risk_score": 3.5,
        "previous_insurance": "yes",
        "credit_score_proxy": 780,
        "consent_given": True,
        "consent_timestamp": "2024-11-19T10:30:00Z"
    },
    {
        "name": "High Risk - Young Driver, Low Income",
        "lead_id": "AUTO_HIGHRISK_001",
        "age": 21,
        "income": 28000,
        "policy_type": "auto",
        "quote_requests_30d": 8,
        "social_engagement_score": 3.1,
        "location_risk_score": 9.5,
        "previous_insurance": "no",
        "credit_score_proxy": 580,
        "consent_given": True,
        "consent_timestamp": "2024-11-19T10:30:00Z"
    },
    {
        "name": "Senior Citizen - Stable Profile",
        "lead_id": "AUTO_SENIOR_001",
        "age": 68,
        "income": 55000,
        "policy_type": "auto",
        "quote_requests_30d": 1,
        "social_engagement_score": 6.8,
        "location_risk_score": 4.2,
        "previous_insurance": "yes",
        "credit_score_proxy": 740,
        "consent_given": True,
        "consent_timestamp": "2024-11-19T10:30:00Z"
    },
    {
        "name": "Middle-Aged - Average Profile",
        "lead_id": "AUTO_AVERAGE_001",
        "age": 45,
        "income": 62000,
        "policy_type": "auto",
        "quote_requests_30d": 3,
        "social_engagement_score": 6.5,
        "location_risk_score": 6.0,
        "previous_insurance": "yes",
        "credit_score_proxy": 680,
        "consent_given": True,
        "consent_timestamp": "2024-11-19T10:30:00Z"
    },
    {
        "name": "High Earner - Low Engagement",
        "lead_id": "AUTO_HIGHEARNER_001",
        "age": 52,
        "income": 150000,
        "policy_type": "auto",
        "quote_requests_30d": 1,
        "social_engagement_score": 2.5,
        "location_risk_score": 2.8,
        "previous_insurance": "yes",
        "credit_score_proxy": 820,
        "consent_given": True,
        "consent_timestamp": "2024-11-19T10:30:00Z"
    },
    {
        "name": "First-Time Buyer - High Engagement",
        "lead_id": "AUTO_FIRSTTIME_001",
        "age": 26,
        "income": 48000,
        "policy_type": "auto",
        "quote_requests_30d": 6,
        "social_engagement_score": 9.8,
        "location_risk_score": 5.5,
        "previous_insurance": "no",
        "credit_score_proxy": 650,
        "consent_given": True,
        "consent_timestamp": "2024-11-19T10:30:00Z"
    }
]

print("=" * 100)
print("AUTO INSURANCE ENSEMBLE - DIVERSE LEAD TESTING")
print("=" * 100)
print(f"\nTesting {len(test_leads)} leads with different risk profiles...")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

results = []

for i, lead in enumerate(test_leads, 1):
    print(f"\n{'=' * 100}")
    print(f"TEST {i}/{len(test_leads)}: {lead['name']}")
    print(f"{'=' * 100}")
    
    # Extract lead data (remove 'name' field)
    lead_data = {k: v for k, v in lead.items() if k != 'name'}
    
    print(f"Profile: Age {lead['age']}, Income ${lead['income']:,}, Credit {lead['credit_score_proxy']}")
    print(f"         Engagement: {lead['social_engagement_score']}, Risk: {lead['location_risk_score']}")
    print(f"         Previous Insurance: {lead['previous_insurance']}, Quotes: {lead['quote_requests_30d']}")
    
    try:
        response = requests.post(API_URL, json=lead_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            results.append({
                'name': lead['name'],
                'result': result
            })
            
            print(f"\n✅ RESULTS:")
            print(f"   Ensemble Score:     {result.get('score', 'N/A'):>6} (Confidence: {result.get('confidence', 0):.3f})")
            print(f"   XGBoost Score:      {result.get('xgboost_score', 'N/A'):>6} (Confidence: {result.get('xgboost_confidence', 0):.3f})")
            print(f"   Deep Learning:      {result.get('deep_learning_score', 'N/A'):>6} (Confidence: {result.get('deep_learning_confidence', 0):.3f})")
            
            if 'ensemble_weights' in result:
                weights = result['ensemble_weights']
                print(f"   Ensemble Weights:   XGBoost: {weights.get('xgboost', 0):.2f}, DL: {weights.get('deep_learning', 0):.2f}")
            
            # Interpretation
            score = result.get('score', 0)
            if score >= 90:
                print(f"   📊 INTERPRETATION:  🌟 EXCELLENT LEAD - High conversion probability")
            elif score >= 75:
                print(f"   📊 INTERPRETATION:  ✅ GOOD LEAD - Strong potential")
            elif score >= 60:
                print(f"   📊 INTERPRETATION:  ⚠️  MODERATE LEAD - Requires nurturing")
            else:
                print(f"   📊 INTERPRETATION:  ⚡ LOW PRIORITY - High risk/low conversion")
                
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

# Summary
print(f"\n\n{'=' * 100}")
print("SUMMARY - ALL LEADS")
print(f"{'=' * 100}\n")

print(f"{'Lead Profile':<40} {'Ensemble':<10} {'XGBoost':<10} {'Deep Learning':<15} {'Confidence':<12}")
print(f"{'-' * 100}")

for item in results:
    name = item['name']
    result = item['result']
    print(f"{name:<40} {result.get('score', 0):>8.2f}   {result.get('xgboost_score', 0):>8.2f}   "
          f"{result.get('deep_learning_score', 0):>13.2f}   {result.get('confidence', 0):>10.3f}")

print(f"\n{'=' * 100}")
print(f"✅ Testing Complete! Tested {len(results)}/{len(test_leads)} leads successfully.")
print(f"{'=' * 100}\n")

