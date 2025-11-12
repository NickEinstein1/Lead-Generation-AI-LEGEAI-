# Life Insurance Policy Types - Comprehensive Classification

## Overview

The LEGEAI platform now includes a comprehensive classification system for life insurance policy types, covering all major categories used in the underwriting business. This system enables accurate product recommendations, better lead scoring, and improved customer matching.

## Policy Categories

### 1. **Term Life Insurance**
Temporary coverage for a specific period (typically 10, 20, or 30 years).

#### Subcategories:
- **Term Life** (`term_life`) - Standard term life insurance
- **Level Term** (`level_term`) - Death benefit and premiums remain constant
- **Decreasing Term** (`decreasing_term`) - Death benefit decreases over time (mortgage protection)
- **Increasing Term** (`increasing_term`) - Death benefit increases over time
- **Annual Renewable Term** (`annual_renewable_term`) - Renews annually with increasing premiums
- **Return of Premium Term** (`return_of_premium_term`) - Returns premiums if policyholder survives term

**Best For:** Young families, mortgage protection, income replacement, temporary needs

---

### 2. **Permanent Life Insurance - Whole Life**
Lifetime coverage with guaranteed cash value accumulation.

#### Subcategories:
- **Whole Life** (`whole_life`) - Traditional whole life insurance
- **Traditional Whole Life** (`traditional_whole_life`) - Standard whole life with fixed premiums
- **Limited Pay Whole Life** (`limited_pay_whole_life`) - Premiums paid for limited period (e.g., 20-pay life)
- **Single Premium Whole Life** (`single_premium_whole_life`) - One-time premium payment

**Best For:** Estate planning, guaranteed coverage, forced savings, lifetime protection

---

### 3. **Permanent Life Insurance - Universal Life**
Flexible permanent coverage with adjustable premiums and death benefits.

#### Subcategories:
- **Universal Life** (`universal_life`) - Standard universal life with flexible premiums
- **Indexed Universal Life (IUL)** (`indexed_universal_life`) - Cash value tied to market index performance
- **Variable Universal Life (VUL)** (`variable_universal_life`) - Investment in separate accounts (stocks, bonds)
- **Guaranteed Universal Life (GUL)** (`guaranteed_universal_life`) - Guaranteed death benefit with minimal cash value

**Best For:** Wealth accumulation, tax-advantaged growth, estate planning, flexible premium needs

---

### 4. **Variable Life Insurance**
Permanent coverage with investment component and market risk.

#### Subcategories:
- **Variable Life** (`variable_life`) - Fixed premiums with investment options

**Best For:** Sophisticated investors, maximum growth potential, long-term wealth building

---

### 5. **Annuities**
Life insurance products designed to provide retirement income.

#### Subcategories:
- **Fixed Annuity** (`fixed_annuity`) - Guaranteed income with fixed interest rate
- **Variable Annuity** (`variable_annuity`) - Investment in sub-accounts with market exposure
- **Indexed Annuity** (`indexed_annuity`) - Returns linked to market index with downside protection
- **Immediate Annuity** (`immediate_annuity`) - Income starts immediately after purchase
- **Deferred Annuity** (`deferred_annuity`) - Income starts at future date

**Best For:** Retirement income, conservative investors, guaranteed returns, tax deferral

---

### 6. **Specialty Life Insurance**
Specialized products for specific needs and demographics.

#### Subcategories:
- **Final Expense** (`final_expense`) - Small whole life policy for funeral costs
- **Burial Insurance** (`burial_insurance`) - Specifically for burial expenses
- **Guaranteed Issue** (`guaranteed_issue`) - No medical exam required
- **Simplified Issue** (`simplified_issue`) - Simplified underwriting process
- **Group Life** (`group_life`) - Employer-sponsored coverage
- **Key Person Insurance** (`key_person_insurance`) - Business coverage for key employees
- **Survivorship Life** (`survivorship_life`) - Second-to-die policy for couples

**Best For:** Seniors, simplified underwriting, business needs, estate planning

---

### 7. **Hybrid Products**
Products combining life insurance with additional benefits.

#### Subcategories:
- **Long-Term Care Rider** (`long_term_care_rider`) - Life insurance with LTC benefits
- **Chronic Illness Rider** (`chronic_illness_rider`) - Accelerated death benefit for chronic illness

**Best For:** Comprehensive protection, long-term care planning, multi-benefit needs

---

## API Endpoints

### Get All Policy Types
```http
GET /v1/life-insurance/policy-types
```

Returns all available policy types organized by category.

### Get Policy Types by Category
```http
GET /v1/life-insurance/policy-types/{category}
```

Categories: `term`, `permanent`, `annuity`, `specialty`, `hybrid`

### Score Lead with Policy Recommendations
```http
POST /v1/life-insurance/score-life-insurance-lead
```

Returns lead score with detailed policy recommendations based on customer profile.

---

## Usage Examples

### Example 1: Young Family (Income Replacement)
**Profile:** Age 32, $75,000 income, 2 dependents
**Recommended:** Term Life, Level Term
**Reason:** Affordable coverage for temporary income replacement needs

### Example 2: High Net Worth (Estate Planning)
**Profile:** Age 55, $250,000 income, estate planning goals
**Recommended:** Indexed Universal Life, Whole Life, Variable Universal Life
**Reason:** Tax-advantaged wealth accumulation and estate transfer

### Example 3: Retiree (Retirement Income)
**Profile:** Age 65, $100,000 savings, seeking guaranteed income
**Recommended:** Fixed Annuity, Indexed Annuity
**Reason:** Guaranteed retirement income with principal protection

### Example 4: Senior (Final Expense)
**Profile:** Age 72, limited income, funeral cost coverage
**Recommended:** Final Expense, Guaranteed Issue
**Reason:** Simplified underwriting, affordable coverage for burial costs

---

## Integration with Lead Scoring

The policy type classification is fully integrated with the life insurance lead scoring model:

1. **Automatic Recommendations:** Based on age, income, life stage, and goals
2. **Multiple Options:** Top 3 policy recommendations with detailed information
3. **Premium Estimates:** Calculated based on coverage amount and customer profile
4. **Underwriting Complexity:** Indicates application difficulty (simple, moderate, complex)

---

## Technical Implementation

### Database Schema
Policy types are stored in the `product_interest` field of the `leads` table and in the `metadata` JSON field for detailed tracking.

### Enum Definition
```python
from backend.models.insurance_products import LifeInsurancePolicyType

# Access policy types
policy = LifeInsurancePolicyType.INDEXED_UNIVERSAL_LIFE
```

### Get Recommendations
```python
from backend.models.insurance_products import get_recommended_policy_types

recommendations = get_recommended_policy_types(
    age=45,
    income=150000,
    goal="estate_planning"
)
```

---

## Compliance & Regulations

All policy types comply with:
- State insurance regulations
- NAIC model laws
- FINRA rules (for variable products)
- SEC regulations (for securities-based products)
- Tax code requirements (IRC Section 7702)

