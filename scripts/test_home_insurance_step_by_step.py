"""
Step-by-Step Home Insurance Testing
Tests each model individually: XGBoost -> Deep Learning -> Ensemble
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/v1"

# Test lead for home insurance
test_lead = {
    "lead_id": "HOME_STEP_TEST_001",
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
print("🏠 HOME INSURANCE - STEP-BY-STEP MODEL TESTING")
print("="*80)
print(f"\n📋 Test Lead:")
print(f"   Lead ID: {test_lead['lead_id']}")
print(f"   Age: {test_lead['age']}, Income: ${test_lead['income']:,}")
print(f"   Credit: {test_lead['credit_score_proxy']}, Engagement: {test_lead['social_engagement_score']}")
print(f"   Previous Insurance: {test_lead['previous_insurance']}")

# Step 1: Test Health Check
print("\n" + "="*80)
print("STEP 1: Health Check")
print("="*80)

try:
    response = requests.get(f"{BASE_URL}/home-insurance/health")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Health Check Passed")
        print(f"   Status: {result.get('status')}")
        print(f"   Ensemble Available: {result.get('ensemble_available')}")
        print(f"   XGBoost Available: {result.get('xgboost_available')}")
    else:
        print(f"❌ Health Check Failed: {response.status_code}")
        print(f"   Response: {response.text}")
        exit(1)
except Exception as e:
    print(f"❌ Health Check Exception: {e}")
    exit(1)

# Step 2: Test Model Info
print("\n" + "="*80)
print("STEP 2: Model Info")
print("="*80)

try:
    response = requests.get(f"{BASE_URL}/home-insurance/model-info")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Model Info Retrieved")
        print(f"   Model Type: {result.get('model_type')}")
        print(f"   Version: {result.get('version')}")
        print(f"   Features: {len(result.get('features', []))} features")
    else:
        print(f"❌ Model Info Failed: {response.status_code}")
except Exception as e:
    print(f"❌ Model Info Exception: {e}")

# Step 3: Test Compare Models (shows individual model scores)
print("\n" + "="*80)
print("STEP 3: Compare All Models (XGBoost + Deep Learning)")
print("="*80)

try:
    response = requests.post(
        f"{BASE_URL}/home-insurance/compare-models",
        json=test_lead,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Model Comparison Complete")
        
        # XGBoost Results
        if 'xgboost' in result.get('models', {}):
            xgb = result['models']['xgboost']
            if xgb.get('available'):
                print(f"\n   📊 XGBoost Model:")
                print(f"      Score: {xgb.get('score', 0):.2f}")
                print(f"      Confidence: {xgb.get('confidence', 0):.1%}")
                print(f"      Status: ✅ Available")
            else:
                print(f"\n   📊 XGBoost Model:")
                print(f"      Status: ❌ Not Available")
                print(f"      Error: {xgb.get('error', 'Unknown')}")
        
        # Deep Learning Results
        if 'deep_learning' in result.get('models', {}):
            dl = result['models']['deep_learning']
            if dl.get('available'):
                print(f"\n   🧠 Deep Learning Model:")
                print(f"      Score: {dl.get('score', 0):.2f}")
                print(f"      Confidence: {dl.get('confidence', 0):.1%}")
                print(f"      Status: ✅ Available")
            else:
                print(f"\n   🧠 Deep Learning Model:")
                print(f"      Status: ❌ Not Available")
                print(f"      Error: {dl.get('error', 'Unknown')}")
        
        # Ensemble Results
        if 'ensemble' in result.get('models', {}):
            ens = result['models']['ensemble']
            if ens.get('available'):
                print(f"\n   ⚖️  Ensemble Model:")
                print(f"      Score: {ens.get('score', 0):.2f}")
                print(f"      Confidence: {ens.get('confidence', 0):.1%}")
                if ens.get('ensemble_weights'):
                    weights = ens['ensemble_weights']
                    print(f"      Weights: XGBoost={weights.get('xgboost', 0):.1%}, DL={weights.get('deep_learning', 0):.1%}")
                print(f"      Status: ✅ Available")
            else:
                print(f"\n   ⚖️  Ensemble Model:")
                print(f"      Status: ❌ Not Available")
                print(f"      Error: {ens.get('error', 'Unknown')}")
        
        print(f"\n   🎯 Recommended Score: {result.get('recommended_score', 0):.2f}")
        print(f"   🎯 Recommended Model: {result.get('recommended_model', 'N/A')}")
        
    else:
        print(f"❌ Model Comparison Failed: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"❌ Model Comparison Exception: {e}")

# Step 4: Test Ensemble Scoring
print("\n" + "="*80)
print("STEP 4: Ensemble Scoring (Single Lead)")
print("="*80)

try:
    response = requests.post(
        f"{BASE_URL}/home-insurance/score-lead",
        json=test_lead,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Ensemble Scoring Complete")
        print(f"\n   Lead ID: {result.get('lead_id')}")
        print(f"   Final Score: {result.get('score', 0):.2f}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Model Type: {result.get('model_type')}")

        if result.get('xgboost_score') and result.get('deep_learning_score'):
            print(f"\n   📊 Individual Model Scores:")
            print(f"      XGBoost:       {result.get('xgboost_score', 0):.2f} (Confidence: {result.get('xgboost_confidence', 0):.1%})")
            print(f"      Deep Learning: {result.get('deep_learning_score', 0):.2f} (Confidence: {result.get('deep_learning_confidence', 0):.1%})")

            if result.get('ensemble_weights'):
                weights = result['ensemble_weights']
                print(f"\n   ⚖️  Adaptive Weights:")
                print(f"      XGBoost:       {weights.get('xgboost', 0):.1%}")
                print(f"      Deep Learning: {weights.get('deep_learning', 0):.1%}")

        print(f"\n   Models Used: {', '.join(result.get('models_used', []))}")

    else:
        print(f"❌ Ensemble Scoring Failed: {response.status_code}")
        print(f"   Response: {response.text}")

except Exception as e:
    print(f"❌ Ensemble Scoring Exception: {e}")

# Summary
print("\n" + "="*80)
print("✅ HOME INSURANCE TESTING COMPLETE")
print("="*80)
print("\nAll steps completed. Review the results above.")
print("\nNext: You can test with different leads or move to Health Insurance testing.")
print("="*80 + "\n")

