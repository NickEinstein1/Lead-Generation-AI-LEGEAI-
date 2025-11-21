"""
Test script for Auto Insurance Deep Learning Model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.models.auto_insurance_scoring.inference_deep_learning import AutoInsuranceDeepLearningScorer

# Test lead data
test_lead = {
    'age': 35,
    'income': 75000,
    'policy_type': 'auto',
    'quote_requests_30d': 3,
    'social_engagement_score': 8.5,
    'location_risk_score': 6.2,
    'previous_insurance': 'yes',
    'credit_score_proxy': 720,
    'consent_given': True
}

print("=" * 80)
print("TESTING AUTO INSURANCE DEEP LEARNING MODEL")
print("=" * 80)

try:
    # Initialize scorer
    print("\n1. Loading Deep Learning model...")
    scorer = AutoInsuranceDeepLearningScorer()
    print("✅ Model loaded successfully!")
    
    # Score the lead
    print("\n2. Scoring test lead...")
    print(f"Test Lead: {test_lead}")
    
    result = scorer.score_lead(test_lead)
    
    print("\n3. Results:")
    print("=" * 80)
    print(f"Score: {result.get('score', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"Model Type: {result.get('model_type', 'N/A')}")
    print(f"Timestamp: {result.get('timestamp', 'N/A')}")
    print("=" * 80)
    
    if result.get('score', 0) > 0:
        print("\n✅ SUCCESS! Model is working correctly!")
    else:
        print("\n⚠️  WARNING: Score is 0, there may be an issue")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

