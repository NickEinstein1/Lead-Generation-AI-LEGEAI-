#!/usr/bin/env python3
"""
Test script for Life Insurance Policy Type Classification
Demonstrates the new comprehensive policy type system
"""

import sys
import json
from backend.models.insurance_products import (
    LifeInsurancePolicyType,
    get_recommended_policy_types,
    get_policy_display_name,
    get_all_life_insurance_categories,
    LIFE_INSURANCE_PRODUCTS
)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_all_policy_types():
    """Display all available policy types"""
    print_section("ALL LIFE INSURANCE POLICY TYPES")
    
    categories = {}
    for policy_type, info in LIFE_INSURANCE_PRODUCTS.items():
        if info.category not in categories:
            categories[info.category] = []
        categories[info.category].append(info)
    
    for category, products in sorted(categories.items()):
        print(f"\n📁 {category.upper().replace('_', ' ')}")
        print("-" * 80)
        for product in products:
            print(f"  • {product.display_name}")
            print(f"    Type: {product.policy_type.value}")
            print(f"    Age Range: {product.typical_age_range[0]}-{product.typical_age_range[1]}")
            print(f"    Cash Value: {'Yes' if product.cash_value else 'No'}")
            print(f"    Investment: {'Yes' if product.investment_component else 'No'}")
            print()


def test_recommendations():
    """Test policy recommendations for different customer profiles"""
    print_section("POLICY RECOMMENDATIONS BY CUSTOMER PROFILE")
    
    test_cases = [
        {
            "name": "Young Family",
            "age": 32,
            "income": 75000,
            "goal": "income_replacement",
            "description": "32-year-old with $75k income, needs income replacement"
        },
        {
            "name": "High Net Worth Professional",
            "age": 45,
            "income": 250000,
            "goal": "estate_planning",
            "description": "45-year-old with $250k income, estate planning focus"
        },
        {
            "name": "Pre-Retiree",
            "age": 58,
            "income": 120000,
            "goal": "retirement_income",
            "description": "58-year-old with $120k income, seeking retirement income"
        },
        {
            "name": "Wealth Builder",
            "age": 38,
            "income": 150000,
            "goal": "wealth_accumulation",
            "description": "38-year-old with $150k income, wealth accumulation"
        },
        {
            "name": "Senior",
            "age": 72,
            "income": 40000,
            "goal": "final_expense",
            "description": "72-year-old with $40k income, final expense coverage"
        }
    ]
    
    for case in test_cases:
        print(f"\n👤 {case['name']}")
        print(f"   {case['description']}")
        print("-" * 80)
        
        recommendations = get_recommended_policy_types(
            age=case['age'],
            income=case['income'],
            goal=case['goal']
        )
        
        if recommendations:
            print(f"   ✅ Recommended Policies:")
            for i, policy_type in enumerate(recommendations, 1):
                display_name = get_policy_display_name(policy_type)
                print(f"      {i}. {display_name} ({policy_type.value})")
                
                if policy_type in LIFE_INSURANCE_PRODUCTS:
                    info = LIFE_INSURANCE_PRODUCTS[policy_type]
                    print(f"         Best for: {', '.join(info.best_for[:2])}")
        else:
            print("   ⚠️  No specific recommendations")
        print()


def test_category_organization():
    """Test category organization"""
    print_section("POLICY TYPES BY CATEGORY")
    
    categories = get_all_life_insurance_categories()
    
    for category, products in sorted(categories.items()):
        print(f"\n📂 {category.upper()}")
        print(f"   Total Products: {len(products)}")
        print(f"   Products:")
        for product in products:
            print(f"      • {product}")


def test_product_details():
    """Display detailed information for key products"""
    print_section("DETAILED PRODUCT INFORMATION")
    
    key_products = [
        LifeInsurancePolicyType.TERM_LIFE,
        LifeInsurancePolicyType.INDEXED_UNIVERSAL_LIFE,
        LifeInsurancePolicyType.WHOLE_LIFE,
        LifeInsurancePolicyType.FIXED_ANNUITY,
        LifeInsurancePolicyType.FINAL_EXPENSE
    ]
    
    for policy_type in key_products:
        if policy_type in LIFE_INSURANCE_PRODUCTS:
            info = LIFE_INSURANCE_PRODUCTS[policy_type]
            print(f"\n📋 {info.display_name}")
            print("-" * 80)
            print(f"   Category: {info.category}")
            print(f"   Description: {info.description}")
            print(f"   Age Range: {info.typical_age_range[0]}-{info.typical_age_range[1]} years")
            print(f"   Coverage Range: ${info.typical_coverage_range[0]:,} - ${info.typical_coverage_range[1]:,}")
            print(f"   Cash Value: {'Yes' if info.cash_value else 'No'}")
            print(f"   Investment Component: {'Yes' if info.investment_component else 'No'}")
            print(f"   Premium Flexibility: {info.premium_flexibility}")
            print(f"   Underwriting: {info.underwriting_complexity}")
            print(f"   Best For:")
            for item in info.best_for:
                print(f"      • {item}")
            print(f"   Key Features:")
            for feature in info.key_features:
                print(f"      • {feature}")


def main():
    """Run all tests"""
    print("\n" + "🔷" * 40)
    print("  LIFE INSURANCE POLICY TYPE CLASSIFICATION SYSTEM")
    print("  LEGEAI Platform - Comprehensive Testing")
    print("🔷" * 40)
    
    # Run tests
    test_all_policy_types()
    test_recommendations()
    test_category_organization()
    test_product_details()
    
    # Summary
    print_section("SUMMARY")
    print(f"✅ Total Policy Types: {len(LifeInsurancePolicyType)}")
    print(f"✅ Products in Catalog: {len(LIFE_INSURANCE_PRODUCTS)}")
    print(f"✅ Categories: {len(get_all_life_insurance_categories())}")
    print("\n✨ All tests completed successfully!\n")


if __name__ == "__main__":
    main()

