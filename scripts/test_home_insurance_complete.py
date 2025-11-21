"""
Complete Home Insurance Testing - All Models
Tests XGBoost, Deep Learning, and Ensemble separately
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/v1"

# Test lead for home insurance
test_lead = {
    "lead_id": "HOME_COMPLETE_TEST_001",
    "age": 45,
    "income": 150000,
    "policy_type": "home",
    "quote_requests_30d": 3,
    "social_engagement_score": 8.5,
    "location_risk_score": 3.2,
    "previous_insurance": "yes",
    "credit_score_proxy": 800,
    "consent_given": True,
    "consent_timestamp": "2024-11-19T10:30:00Z"
}

print("\n" + "="*80)
print("🏠 HOME INSURANCE - COMPLETE MODEL TESTING")
print("="*80)

print(f"\n📋 Test Lead:")
print(f"   Lead ID: {test_lead['lead_id']}")
print(f"   Age: {test_lead['age']}, Income: ${test_lead['income']:,}")
print(f"   Credit: {test_lead['credit_score_proxy']}, Engagement: {test_lead['social_engagement_score']}")
print(f"   Previous Insurance: {test_lead['previous_insurance']}")

# Test 1: Health Check
print(f"\n{'='*80}")
print("TEST 1: Health Check")
print(f"{'='*80}")

try:
    response = requests.get(f"{BASE_URL}/home-insurance/health")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Health Check: {result['status']}")
        print(f"   Ensemble Available: {result.get('ensemble_available', False)}")
        print(f"   XGBoost Available: {result.get('xgboost_available', False)}")
    else:
        print(f"❌ Health Check Failed: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ Health Check Exception: {e}")

# Test 2: Model Info
print(f"\n{'='*80}")
print("TEST 2: Model Info")
print(f"{'='*80}")

try:
    response = requests.get(f"{BASE_URL}/home-insurance/model-info")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Model Info Retrieved:")
        print(f"   Model Type: {result.get('model_type', 'N/A')}")
        print(f"   Version: {result.get('version', 'N/A')}")
        print(f"   Features: {len(result.get('features', []))} features")
    else:
        print(f"❌ Model Info Failed: {response.status_code}")
except Exception as e:
    print(f"❌ Model Info Exception: {e}")

# Test 3: Compare Models (XGBoost vs Deep Learning vs Ensemble)
print(f"\n{'='*80}")
print("TEST 3: Compare All Models")
print(f"{'='*80}")

try:
    response = requests.post(
        f"{BASE_URL}/home-insurance/compare-models",
        json=test_lead,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Model Comparison Complete:")
        print(f"\n   Lead ID: {result.get('lead_id', 'N/A')}")
        
        models = result.get('models', {})
        
        # XGBoost
        if 'xgboost' in models and models['xgboost'].get('available'):
            xgb = models['xgboost']
            print(f"\n   📊 XGBoost Model:")
            print(f"      Score: {xgb.get('score', 0):.2f}")
            print(f"      Confidence: {xgb.get('confidence', 0):.1%}")
            print(f"      Status: ✅ Available")
        else:
            print(f"\n   📊 XGBoost Model: ❌ Not Available")
            if 'xgboost' in models:
                print(f"      Error: {models['xgboost'].get('error', 'Unknown')}")
        
        # Deep Learning
        if 'deep_learning' in models and models['deep_learning'].get('available'):
            dl = models['deep_learning']
            print(f"\n   🧠 Deep Learning Model:")
            print(f"      Score: {dl.get('score', 0):.2f}")
            print(f"      Confidence: {dl.get('confidence', 0):.1%}")
            print(f"      Status: ✅ Available")
        else:
            print(f"\n   🧠 Deep Learning Model: ❌ Not Available")
            if 'deep_learning' in models:
                print(f"      Error: {models['deep_learning'].get('error', 'Unknown')}")
        
        # Ensemble
        if 'ensemble' in models and models['ensemble'].get('available'):
            ens = models['ensemble']
            print(f"\n   ⚖️  Ensemble Model:")
            print(f"      Score: {ens.get('score', 0):.2f}")
            print(f"      Confidence: {ens.get('confidence', 0):.1%}")
            print(f"      Status: ✅ Available")
            
            weights = ens.get('ensemble_weights', {})
            if weights:
                print(f"      Weights:")
                print(f"         • XGBoost: {weights.get('xgboost', 0):.1%}")
                print(f"         • Deep Learning: {weights.get('deep_learning', 0):.1%}")
        else:
            print(f"\n   ⚖️  Ensemble Model: ❌ Not Available")
            if 'ensemble' in models:
                print(f"      Error: {models['ensemble'].get('error', 'Unknown')}")
        
        # Recommended
        if 'recommended_score' in result:
            print(f"\n   🎯 Recommended Score: {result['recommended_score']:.2f}")
            print(f"   🎯 Recommended Model: {result.get('recommended_model', 'N/A')}")
            
    else:
        print(f"❌ Model Comparison Failed: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"❌ Model Comparison Exception: {e}")

# Test 4: Score Lead (Ensemble)
print(f"\n{'='*80}")
print("TEST 4: Score Lead with Ensemble")
print(f"{'='*80}")

try:
    response = requests.post(
        f"{BASE_URL}/home-insurance/score-lead",
        json=test_lead,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Lead Scored Successfully:")
        print(f"\n   Final Score: {result.get('score', 0):.2f}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Model Type: {result.get('model_type', 'N/A')}")
        
        if result.get('models_used'):
            print(f"   Models Used: {', '.join(result['models_used'])}")
        
    else:
        print(f"❌ Score Lead Failed: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"❌ Score Lead Exception: {e}")

print(f"\n{'='*80}")
print("✅ HOME INSURANCE TESTING COMPLETE")
print(f"{'='*80}\n")

