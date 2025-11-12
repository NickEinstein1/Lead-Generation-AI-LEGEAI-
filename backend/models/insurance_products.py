"""
Insurance Product Type Enumerations and Classifications
Defines all insurance product types and their subcategories for the LEGEAI platform
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class InsuranceCategory(Enum):
    """Main insurance product categories"""
    AUTO = "auto"
    HOME = "home"
    LIFE = "life"
    HEALTH = "health"
    BUSINESS = "business"
    DISABILITY = "disability"
    UMBRELLA = "umbrella"
    TRAVEL = "travel"


class LifeInsurancePolicyType(Enum):
    """
    Life Insurance Policy Types - Comprehensive classification
    for underwriting and product recommendation
    """
    # Term Life Insurance
    TERM_LIFE = "term_life"
    LEVEL_TERM = "level_term"
    DECREASING_TERM = "decreasing_term"
    INCREASING_TERM = "increasing_term"
    ANNUAL_RENEWABLE_TERM = "annual_renewable_term"
    RETURN_OF_PREMIUM_TERM = "return_of_premium_term"
    
    # Permanent Life Insurance - Whole Life
    WHOLE_LIFE = "whole_life"
    TRADITIONAL_WHOLE_LIFE = "traditional_whole_life"
    LIMITED_PAY_WHOLE_LIFE = "limited_pay_whole_life"
    SINGLE_PREMIUM_WHOLE_LIFE = "single_premium_whole_life"
    
    # Permanent Life Insurance - Universal Life
    UNIVERSAL_LIFE = "universal_life"
    INDEXED_UNIVERSAL_LIFE = "indexed_universal_life"  # IUL
    VARIABLE_UNIVERSAL_LIFE = "variable_universal_life"  # VUL
    GUARANTEED_UNIVERSAL_LIFE = "guaranteed_universal_life"  # GUL
    
    # Variable Life Insurance
    VARIABLE_LIFE = "variable_life"
    
    # Annuities (Life Insurance Products)
    FIXED_ANNUITY = "fixed_annuity"
    VARIABLE_ANNUITY = "variable_annuity"
    INDEXED_ANNUITY = "indexed_annuity"
    IMMEDIATE_ANNUITY = "immediate_annuity"
    DEFERRED_ANNUITY = "deferred_annuity"
    
    # Specialty Life Insurance
    FINAL_EXPENSE = "final_expense"
    BURIAL_INSURANCE = "burial_insurance"
    GUARANTEED_ISSUE = "guaranteed_issue"
    SIMPLIFIED_ISSUE = "simplified_issue"
    GROUP_LIFE = "group_life"
    KEY_PERSON_INSURANCE = "key_person_insurance"
    SURVIVORSHIP_LIFE = "survivorship_life"  # Second-to-die
    
    # Hybrid Products
    LONG_TERM_CARE_RIDER = "long_term_care_rider"
    CHRONIC_ILLNESS_RIDER = "chronic_illness_rider"


@dataclass
class LifeInsuranceProductInfo:
    """Detailed information about a life insurance product type"""
    policy_type: LifeInsurancePolicyType
    display_name: str
    category: str  # "term", "permanent", "annuity", "specialty", "hybrid"
    description: str
    typical_age_range: tuple  # (min_age, max_age)
    typical_coverage_range: tuple  # (min_coverage, max_coverage)
    cash_value: bool
    investment_component: bool
    premium_flexibility: str  # "fixed", "flexible", "variable"
    best_for: List[str]
    key_features: List[str]
    underwriting_complexity: str  # "simple", "moderate", "complex"


# Life Insurance Product Catalog
LIFE_INSURANCE_PRODUCTS: Dict[LifeInsurancePolicyType, LifeInsuranceProductInfo] = {
    LifeInsurancePolicyType.TERM_LIFE: LifeInsuranceProductInfo(
        policy_type=LifeInsurancePolicyType.TERM_LIFE,
        display_name="Term Life Insurance",
        category="term",
        description="Temporary coverage for a specific period (10, 20, 30 years)",
        typical_age_range=(18, 75),
        typical_coverage_range=(50000, 5000000),
        cash_value=False,
        investment_component=False,
        premium_flexibility="fixed",
        best_for=["Young families", "Mortgage protection", "Income replacement"],
        key_features=["Affordable premiums", "Simple structure", "Temporary coverage"],
        underwriting_complexity="simple"
    ),
    LifeInsurancePolicyType.INDEXED_UNIVERSAL_LIFE: LifeInsuranceProductInfo(
        policy_type=LifeInsurancePolicyType.INDEXED_UNIVERSAL_LIFE,
        display_name="Indexed Universal Life (IUL)",
        category="permanent",
        description="Permanent coverage with cash value tied to market index performance",
        typical_age_range=(25, 70),
        typical_coverage_range=(100000, 10000000),
        cash_value=True,
        investment_component=True,
        premium_flexibility="flexible",
        best_for=["Wealth accumulation", "Tax-advantaged growth", "Estate planning"],
        key_features=["Market-linked returns", "Downside protection", "Flexible premiums", "Tax-deferred growth"],
        underwriting_complexity="complex"
    ),
    LifeInsurancePolicyType.WHOLE_LIFE: LifeInsuranceProductInfo(
        policy_type=LifeInsurancePolicyType.WHOLE_LIFE,
        display_name="Whole Life Insurance",
        category="permanent",
        description="Permanent coverage with guaranteed cash value accumulation",
        typical_age_range=(18, 75),
        typical_coverage_range=(25000, 5000000),
        cash_value=True,
        investment_component=False,
        premium_flexibility="fixed",
        best_for=["Estate planning", "Guaranteed coverage", "Forced savings"],
        key_features=["Lifetime coverage", "Guaranteed cash value", "Fixed premiums", "Dividends (participating policies)"],
        underwriting_complexity="moderate"
    ),
    LifeInsurancePolicyType.VARIABLE_UNIVERSAL_LIFE: LifeInsuranceProductInfo(
        policy_type=LifeInsurancePolicyType.VARIABLE_UNIVERSAL_LIFE,
        display_name="Variable Universal Life (VUL)",
        category="permanent",
        description="Permanent coverage with investment in separate accounts (stocks, bonds)",
        typical_age_range=(25, 70),
        typical_coverage_range=(100000, 10000000),
        cash_value=True,
        investment_component=True,
        premium_flexibility="flexible",
        best_for=["Sophisticated investors", "Maximum growth potential", "Estate planning"],
        key_features=["Investment flexibility", "Potential high returns", "Market risk", "Tax advantages"],
        underwriting_complexity="complex"
    ),
    LifeInsurancePolicyType.FIXED_ANNUITY: LifeInsuranceProductInfo(
        policy_type=LifeInsurancePolicyType.FIXED_ANNUITY,
        display_name="Fixed Annuity",
        category="annuity",
        description="Guaranteed income stream with fixed interest rate",
        typical_age_range=(45, 85),
        typical_coverage_range=(25000, 2000000),
        cash_value=True,
        investment_component=False,
        premium_flexibility="flexible",
        best_for=["Retirement income", "Conservative investors", "Guaranteed returns"],
        key_features=["Guaranteed interest", "Predictable income", "Tax-deferred growth", "Principal protection"],
        underwriting_complexity="simple"
    ),
    LifeInsurancePolicyType.INDEXED_ANNUITY: LifeInsuranceProductInfo(
        policy_type=LifeInsurancePolicyType.INDEXED_ANNUITY,
        display_name="Indexed Annuity (FIA)",
        category="annuity",
        description="Annuity with returns linked to market index with downside protection",
        typical_age_range=(45, 80),
        typical_coverage_range=(50000, 3000000),
        cash_value=True,
        investment_component=True,
        premium_flexibility="flexible",
        best_for=["Moderate risk tolerance", "Market participation with protection", "Retirement planning"],
        key_features=["Index-linked returns", "Principal protection", "Tax deferral", "Income riders available"],
        underwriting_complexity="moderate"
    ),
    LifeInsurancePolicyType.VARIABLE_ANNUITY: LifeInsuranceProductInfo(
        policy_type=LifeInsurancePolicyType.VARIABLE_ANNUITY,
        display_name="Variable Annuity",
        category="annuity",
        description="Annuity with investment in sub-accounts (mutual fund-like)",
        typical_age_range=(40, 75),
        typical_coverage_range=(50000, 5000000),
        cash_value=True,
        investment_component=True,
        premium_flexibility="flexible",
        best_for=["Growth-oriented investors", "Long-term retirement planning", "Tax deferral"],
        key_features=["Investment options", "Potential high returns", "Market risk", "Death benefit options"],
        underwriting_complexity="complex"
    ),
    LifeInsurancePolicyType.FINAL_EXPENSE: LifeInsuranceProductInfo(
        policy_type=LifeInsurancePolicyType.FINAL_EXPENSE,
        display_name="Final Expense Insurance",
        category="specialty",
        description="Small whole life policy to cover funeral and burial costs",
        typical_age_range=(50, 85),
        typical_coverage_range=(5000, 50000),
        cash_value=True,
        investment_component=False,
        premium_flexibility="fixed",
        best_for=["Seniors", "Funeral cost coverage", "Simplified underwriting"],
        key_features=["Small coverage amounts", "Guaranteed acceptance options", "Affordable premiums", "Quick approval"],
        underwriting_complexity="simple"
    ),
    LifeInsurancePolicyType.GUARANTEED_UNIVERSAL_LIFE: LifeInsuranceProductInfo(
        policy_type=LifeInsurancePolicyType.GUARANTEED_UNIVERSAL_LIFE,
        display_name="Guaranteed Universal Life (GUL)",
        category="permanent",
        description="Permanent coverage with guaranteed death benefit and minimal cash value",
        typical_age_range=(25, 75),
        typical_coverage_range=(100000, 5000000),
        cash_value=False,
        investment_component=False,
        premium_flexibility="fixed",
        best_for=["Estate planning", "Guaranteed coverage", "Lower cost permanent insurance"],
        key_features=["Guaranteed death benefit", "Lower premiums than whole life", "Minimal cash value", "Lifetime coverage"],
        underwriting_complexity="moderate"
    ),
}


def get_policy_types_by_category(category: str) -> List[LifeInsurancePolicyType]:
    """Get all policy types for a specific category"""
    return [
        info.policy_type
        for info in LIFE_INSURANCE_PRODUCTS.values()
        if info.category == category
    ]


def get_recommended_policy_types(age: int, income: float, goal: str) -> List[LifeInsurancePolicyType]:
    """
    Recommend policy types based on customer profile

    Args:
        age: Customer age
        income: Annual income
        goal: Primary goal (e.g., "income_replacement", "estate_planning", "retirement_income")

    Returns:
        List of recommended policy types
    """
    recommendations = []

    if goal == "income_replacement" and age < 45:
        recommendations.extend([
            LifeInsurancePolicyType.TERM_LIFE,
            LifeInsurancePolicyType.LEVEL_TERM
        ])

    elif goal == "estate_planning" and income > 150000:
        recommendations.extend([
            LifeInsurancePolicyType.INDEXED_UNIVERSAL_LIFE,
            LifeInsurancePolicyType.WHOLE_LIFE,
            LifeInsurancePolicyType.VARIABLE_UNIVERSAL_LIFE
        ])

    elif goal == "retirement_income" and age >= 45:
        recommendations.extend([
            LifeInsurancePolicyType.FIXED_ANNUITY,
            LifeInsurancePolicyType.INDEXED_ANNUITY,
            LifeInsurancePolicyType.VARIABLE_ANNUITY
        ])

    elif goal == "wealth_accumulation" and income > 100000:
        recommendations.extend([
            LifeInsurancePolicyType.INDEXED_UNIVERSAL_LIFE,
            LifeInsurancePolicyType.VARIABLE_UNIVERSAL_LIFE
        ])

    elif goal == "final_expense" and age >= 50:
        recommendations.extend([
            LifeInsurancePolicyType.FINAL_EXPENSE,
            LifeInsurancePolicyType.GUARANTEED_ISSUE
        ])

    else:
        # Default recommendations
        if age < 40:
            recommendations.append(LifeInsurancePolicyType.TERM_LIFE)
        else:
            recommendations.append(LifeInsurancePolicyType.WHOLE_LIFE)

    return recommendations


def get_policy_display_name(policy_type: LifeInsurancePolicyType) -> str:
    """Get the display name for a policy type"""
    if policy_type in LIFE_INSURANCE_PRODUCTS:
        return LIFE_INSURANCE_PRODUCTS[policy_type].display_name
    return policy_type.value.replace("_", " ").title()


def get_all_life_insurance_categories() -> Dict[str, List[str]]:
    """Get all life insurance policy types organized by category"""
    categories = {}
    for info in LIFE_INSURANCE_PRODUCTS.values():
        if info.category not in categories:
            categories[info.category] = []
        categories[info.category].append(info.display_name)
    return categories

