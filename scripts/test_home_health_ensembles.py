"""
Test Home and Health Insurance Ensembles
Comprehensive testing with diverse lead profiles
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/v1"

# Test leads for Home Insurance
home_test_leads = [
    {
        "name": "Premium Homeowner - High Value Property",
        "lead": {
            "lead_id": "HOME_TEST_001",
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
    },
    {
        "name": "First-Time Homebuyer - Young Professional",
        "lead": {
            "lead_id": "HOME_TEST_002",
            "age": 28,
            "income": 65000,
            "policy_type": "home",
            "quote_requests_30d": 5,
            "social_engagement_score": 9.2,
            "location_risk_score": 5.5,
            "previous_insurance": "no",
            "credit_score_proxy": 680,
            "consent_given": True,
            "consent_timestamp": "2024-11-19T10:30:00Z"
        }
    },
    {
        "name": "Senior Homeowner - Stable Profile",
        "lead": {
            "lead_id": "HOME_TEST_003",
            "age": 67,
            "income": 75000,
            "policy_type": "home",
            "quote_requests_30d": 1,
            "social_engagement_score": 6.0,
            "location_risk_score": 4.0,
            "previous_insurance": "yes",
            "credit_score_proxy": 760,
            "consent_given": True,
            "consent_timestamp": "2024-11-19T10:30:00Z"
        }
    }
]

# Test leads for Health Insurance
health_test_leads = [
    {
        "name": "Young Professional - Excellent Health",
        "lead": {
            "lead_id": "HEALTH_TEST_001",
            "age": 32,
            "income": 95000,
            "policy_type": "health",
            "quote_requests_30d": 2,
            "social_engagement_score": 8.8,
            "location_risk_score": 3.5,
            "previous_insurance": "yes",
            "credit_score_proxy": 780,
            "consent_given": True,
            "consent_timestamp": "2024-11-19T10:30:00Z"
        }
    },
    {
        "name": "Family Coverage - Middle-Aged",
        "lead": {
            "lead_id": "HEALTH_TEST_002",
            "age": 42,
            "income": 110000,
            "policy_type": "health",
            "quote_requests_30d": 4,
            "social_engagement_score": 7.5,
            "location_risk_score": 4.2,
            "previous_insurance": "yes",
            "credit_score_proxy": 720,
            "consent_given": True,
            "consent_timestamp": "2024-11-19T10:30:00Z"
        }
    },
    {
        "name": "Senior - Medicare Supplement",
        "lead": {
            "lead_id": "HEALTH_TEST_003",
            "age": 68,
            "income": 55000,
            "policy_type": "health",
            "quote_requests_30d": 1,
            "social_engagement_score": 5.5,
            "location_risk_score": 6.0,
            "previous_insurance": "yes",
            "credit_score_proxy": 740,
            "consent_given": True,
            "consent_timestamp": "2024-11-19T10:30:00Z"
        }
    },
    {
        "name": "Young Adult - First Coverage",
        "lead": {
            "lead_id": "HEALTH_TEST_004",
            "age": 24,
            "income": 45000,
            "policy_type": "health",
            "quote_requests_30d": 3,
            "social_engagement_score": 9.5,
            "location_risk_score": 4.5,
            "previous_insurance": "no",
            "credit_score_proxy": 650,
            "consent_given": True,
            "consent_timestamp": "2024-11-19T10:30:00Z"
        }
    }
]


def test_insurance_ensemble(insurance_type, test_leads):
    """Test an insurance ensemble with diverse leads"""
    print(f"\n{'='*80}")
    print(f"🏥 TESTING {insurance_type.upper()} INSURANCE ENSEMBLE")
    print(f"{'='*80}\n")
    
    results = []
    
    for test_case in test_leads:
        print(f"\n📋 {test_case['name']}")
        print(f"   Lead ID: {test_case['lead']['lead_id']}")
        print(f"   Age: {test_case['lead']['age']}, Income: ${test_case['lead']['income']:,}")
        print(f"   Credit: {test_case['lead']['credit_score_proxy']}, Engagement: {test_case['lead']['social_engagement_score']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/{insurance_type}-insurance/score-lead",
                json=test_case['lead'],
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                results.append({
                    'name': test_case['name'],
                    'lead_id': result['lead_id'],
                    'score': result['score'],
                    'confidence': result['confidence'],
                    'xgboost_score': result.get('xgboost_score'),
                    'dl_score': result.get('deep_learning_score'),
                    'weights': result.get('ensemble_weights', {})
                })

                print(f"\n   ✅ ENSEMBLE SCORE: {result['score']:.2f} (Confidence: {result['confidence']:.1%})")

                if result.get('xgboost_score') and result.get('deep_learning_score'):
                    print(f"   📊 Model Breakdown:")
                    print(f"      • XGBoost:       {result['xgboost_score']:.2f} (Confidence: {result.get('xgboost_confidence', 0):.1%})")
                    print(f"      • Deep Learning: {result['deep_learning_score']:.2f} (Confidence: {result.get('deep_learning_confidence', 0):.1%})")

                    if result.get('ensemble_weights'):
                        weights = result['ensemble_weights']
                        print(f"   ⚖️  Adaptive Weights:")
                        print(f"      • XGBoost:       {weights.get('xgboost', 0):.1%}")
                        print(f"      • Deep Learning: {weights.get('deep_learning', 0):.1%}")

            else:
                print(f"   ❌ ERROR: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"   ❌ EXCEPTION: {e}")

    # Summary
    if results:
        print(f"\n{'='*80}")
        print(f"📊 {insurance_type.upper()} INSURANCE SUMMARY")
        print(f"{'='*80}\n")

        avg_score = sum(r['score'] for r in results) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)

        print(f"Total Leads Tested: {len(results)}")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Score Range: {min(r['score'] for r in results):.2f} - {max(r['score'] for r in results):.2f}")

        print(f"\n📈 Detailed Results:")
        print(f"{'Lead':<45} {'Score':>8} {'Confidence':>12}")
        print(f"{'-'*70}")
        for r in results:
            print(f"{r['name']:<45} {r['score']:>8.2f} {r['confidence']:>11.1%}")

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 HOME & HEALTH INSURANCE ENSEMBLE TESTING")
    print("="*80)

    # Test Home Insurance
    home_results = test_insurance_ensemble("home", home_test_leads)

    # Test Health Insurance
    health_results = test_insurance_ensemble("health", health_test_leads)

    # Overall Summary
    print(f"\n{'='*80}")
    print(f"🎯 OVERALL SUMMARY - ALL INSURANCE TYPES")
    print(f"{'='*80}\n")

    total_tested = len(home_results) + len(health_results)
    print(f"✅ Total Leads Tested: {total_tested}")
    print(f"   • Home Insurance:   {len(home_results)} leads")
    print(f"   • Health Insurance: {len(health_results)} leads")

    if home_results:
        home_avg = sum(r['score'] for r in home_results) / len(home_results)
        home_conf = sum(r['confidence'] for r in home_results) / len(home_results)
        print(f"\n📊 Home Insurance:")
        print(f"   Average Score: {home_avg:.2f}")
        print(f"   Average Confidence: {home_conf:.1%}")

    if health_results:
        health_avg = sum(r['score'] for r in health_results) / len(health_results)
        health_conf = sum(r['confidence'] for r in health_results) / len(health_results)
        print(f"\n📊 Health Insurance:")
        print(f"   Average Score: {health_avg:.2f}")
        print(f"   Average Confidence: {health_conf:.1%}")

    print(f"\n{'='*80}")
    print(f"✅ ALL TESTS COMPLETE!")
    print(f"{'='*80}\n")

