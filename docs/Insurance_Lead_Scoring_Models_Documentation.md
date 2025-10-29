# Insurance Lead Scoring Models Documentation

## Table of Contents
1. [Overview](#overview)
2. [Base Insurance Lead Scoring Model](#base-insurance-lead-scoring-model)
3. [Healthcare Insurance Lead Scoring Model](#healthcare-insurance-lead-scoring-model)
4. [Life Insurance Lead Scoring Model](#life-insurance-lead-scoring-model)
5. [API Documentation](#api-documentation)
6. [Model Evaluation](#model-evaluation)
7. [Compliance & Ethics](#compliance--ethics)
8. [Deployment Guide](#deployment-guide)

---

## Overview

The Insurance Lead Scoring System is a comprehensive machine learning platform designed to evaluate and score insurance leads across multiple product lines. The system provides accurate, fair, and compliant lead scoring with specialized models for different insurance types.

### System Architecture
```
Insurance Lead Scoring System
├── Base Insurance Model (Foundation)
├── Healthcare Insurance Model (Specialized)
├── Life Insurance Model (Specialized)
├── API Layer (FastAPI)
├── Evaluation Framework
└── Compliance Engine
```

### Key Features
- **Multi-Product Support**: Base, Healthcare, and Life Insurance
- **Real-time Scoring**: Sub-second response times
- **Compliance-First**: GDPR, CCPA, TCPA compliant
- **Bias Detection**: Automated fairness monitoring
- **Interpretable Results**: Feature importance and reasoning
- **Scalable Architecture**: Batch and real-time processing

---

## Base Insurance Lead Scoring Model

### Purpose
The base model serves as the foundation for all insurance lead scoring, providing core functionality and common features across insurance products.

### Model Details
- **Algorithm**: XGBoost Regressor
- **Output**: Conversion probability score (0-100)
- **Training Data**: General insurance leads with consent
- **Update Frequency**: Monthly retraining

### Core Features
```python
feature_columns = [
    'age', 'income', 'location', 'employment_status', 'credit_score',
    'previous_insurance', 'contact_method', 'lead_source', 'time_of_contact',
    'interaction_history', 'demographic_segment', 'risk_profile'
]
```

### Feature Engineering
- **Risk Assessment**: Credit score normalization and risk categorization
- **Demographic Scoring**: Age-income interaction effects
- **Behavioral Patterns**: Contact timing and interaction history
- **Lead Quality**: Source quality and engagement metrics

### Model Performance
- **Target Accuracy**: R² > 0.85
- **Response Time**: < 100ms per lead
- **Batch Processing**: 10,000+ leads per minute
- **Fairness Threshold**: Demographic parity < 0.1

### Usage Example
```python
from models.insurance_lead_scoring.inference import InsuranceLeadScorer

scorer = InsuranceLeadScorer()
result = scorer.score_lead({
    'lead_id': 'BASE_001',
    'age': 35,
    'income': 75000,
    'employment_status': 'employed',
    'credit_score': 720,
    'consent_given': True
})
```

---

## Healthcare Insurance Lead Scoring Model

### Purpose
Specialized model for healthcare insurance leads, incorporating health-specific factors and regulatory compliance requirements.

### Model Details
- **Algorithm**: XGBoost Regressor (Healthcare-optimized)
- **Output**: Healthcare conversion score (0-100) + health insights
- **Training Data**: Healthcare insurance leads with HIPAA compliance
- **Update Frequency**: Bi-weekly retraining

### Healthcare-Specific Features
```python
healthcare_features = [
    'age', 'income', 'family_size', 'employment_status', 'current_insurance_status',
    'health_conditions', 'prescription_medications', 'healthcare_utilization',
    'preferred_provider_network', 'coverage_preferences', 'deductible_preference',
    'chronic_conditions', 'preventive_care_usage', 'specialist_needs'
]
```

### Advanced Feature Engineering
- **Health Risk Scoring**: Chronic conditions and medication analysis
- **Network Preferences**: Provider network alignment scoring
- **Cost Sensitivity**: Deductible and premium preference modeling
- **Utilization Patterns**: Healthcare usage prediction
- **Family Coverage**: Dependent coverage needs assessment

### Healthcare-Specific Outputs
```python
{
    'score': 87.5,
    'health_risk_category': 'MODERATE',
    'recommended_plan_type': 'PPO',
    'estimated_annual_cost': 8400,
    'network_preference_match': 0.92,
    'chronic_condition_coverage': ['diabetes', 'hypertension'],
    'preventive_care_score': 8.2,
    'urgency_level': 'HIGH'
}
```

### Compliance Features
- **HIPAA Compliance**: Health data anonymization and encryption
- **ACA Compliance**: Non-discrimination in health status
- **State Regulations**: State-specific healthcare requirements
- **Privacy Protection**: PII masking and secure processing

### Performance Metrics
- **Overall Accuracy**: R² > 0.88
- **High-Risk Accuracy**: R² > 0.85 for chronic conditions
- **Family Plan Accuracy**: R² > 0.90 for family coverage
- **Network Match Accuracy**: 94% provider preference alignment

---

## Life Insurance Lead Scoring Model

### Purpose
Specialized model for life insurance leads, incorporating mortality risk assessment, coverage adequacy analysis, and life stage considerations.

### Model Details
- **Algorithm**: XGBoost Regressor (Life Insurance-optimized)
- **Output**: Life insurance score (0-100) + actuarial insights
- **Training Data**: Life insurance leads with actuarial data
- **Update Frequency**: Monthly retraining with mortality table updates

### Life Insurance Features
```python
life_insurance_features = [
    'age', 'income', 'marital_status', 'dependents_count', 'employment_status',
    'health_status', 'smoking_status', 'coverage_amount_requested', 'policy_term',
    'existing_life_insurance', 'beneficiary_count', 'debt_obligations',
    'mortgage_balance', 'education_level', 'occupation_risk_level',
    'life_stage', 'financial_dependents', 'estate_planning_needs'
]
```

### Actuarial Feature Engineering
- **Mortality Risk Scoring**: Age, health, smoking, occupation risk
- **Coverage Adequacy**: Income replacement and debt coverage analysis
- **Life Stage Analysis**: Young professional to estate planning stages
- **Financial Responsibility**: Dependents and debt obligations
- **Estate Planning**: High-net-worth and inheritance considerations

### Life Insurance Outputs
```python
{
    'score': 92.3,
    'life_stage': 'family_building',
    'mortality_risk_score': 3.2,
    'recommended_coverage': 750000,
    'coverage_adequacy': 'adequate',
    'coverage_gap': 0,
    'recommended_policy_type': 'TERM_LIFE',
    'urgency_level': 'CRITICAL',
    'estate_planning_urgency': 4.5
}
```

### Policy Recommendations
- **Term Life**: Young families, temporary needs
- **Whole Life**: Estate planning, permanent coverage
- **Universal Life**: Flexible premiums, investment growth
- **Variable Life**: Investment-minded, high-income clients

### Actuarial Compliance
- **Mortality Tables**: Industry-standard life expectancy data
- **Non-Discrimination**: Fair underwriting practices
- **State Regulations**: Insurance commissioner requirements
- **Actuarial Fairness**: Evidence-based risk assessment

### Performance Metrics
- **Overall Accuracy**: R² > 0.87
- **Family Building Accuracy**: R² > 0.92 for family stage
- **High Coverage Accuracy**: R² > 0.89 for $500k+ policies
- **Policy Recommendation**: 91% accuracy in policy type matching

---

## API Documentation

### Base Insurance API

#### Endpoint: `/score-lead`
```http
POST /score-lead
Content-Type: application/json

{
    "lead_id": "BASE_001",
    "age": 35,
    "income": 75000,
    "employment_status": "employed",
    "credit_score": 720,
    "consent_given": true,
    "consent_timestamp": "2024-01-15T10:30:00Z"
}
```

**Response:**
```json
{
    "lead_id": "BASE_001",
    "score": 78.5,
    "confidence": 0.92,
    "risk_category": "MEDIUM",
    "timestamp": "2024-01-15T10:30:15Z",
    "model_version": "1.0_base",
    "compliance_status": "PASSED"
}
```

### Healthcare Insurance API

#### Endpoint: `/score-healthcare-lead`
```http
POST /score-healthcare-lead
Content-Type: application/json

{
    "lead_id": "HEALTH_001",
    "age": 42,
    "income": 85000,
    "family_size": 4,
    "current_insurance_status": "uninsured",
    "health_conditions": ["diabetes", "hypertension"],
    "preferred_provider_network": "PPO",
    "consent_given": true
}
```

**Response:**
```json
{
    "lead_id": "HEALTH_001",
    "score": 87.5,
    "health_risk_category": "MODERATE",
    "recommended_plan_type": "PPO",
    "estimated_annual_cost": 8400,
    "network_preference_match": 0.92,
    "urgency_level": "HIGH",
    "compliance_status": "PASSED"
}
```

### Life Insurance API

#### Endpoint: `/score-life-insurance-lead`
```http
POST /score-life-insurance-lead
Content-Type: application/json

{
    "lead_id": "LIFE_001",
    "age": 35,
    "income": 85000,
    "marital_status": "married",
    "dependents_count": 2,
    "health_status": "good",
    "smoking_status": "non_smoker",
    "coverage_amount_requested": 750000,
    "consent_given": true
}
```

**Response:**
```json
{
    "lead_id": "LIFE_001",
    "score": 92.3,
    "life_stage": "family_building",
    "mortality_risk_score": 3.2,
    "recommended_coverage": 750000,
    "recommended_policy_type": "TERM_LIFE",
    "urgency_level": "CRITICAL",
    "compliance_status": "PASSED"
}
```

### Batch Processing APIs

#### Endpoint: `/score-leads` (All Models)
```http
POST /score-leads
Content-Type: application/json

{
    "leads": [
        {"lead_id": "001", "age": 35, ...},
        {"lead_id": "002", "age": 42, ...}
    ]
}
```

### Utility APIs

#### Coverage Calculator
```http
GET /life-insurance-coverage-calculator?income=85000&dependents=2&mortgage=300000
```

#### Health Risk Assessment
```http
GET /healthcare-risk-assessment?age=42&conditions=diabetes,hypertension
```

---

## Model Evaluation

### Evaluation Framework

Each model includes comprehensive evaluation across multiple dimensions:

#### Accuracy Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Precision/Recall**: For binary classification thresholds

#### Fairness Metrics
- **Demographic Parity**: Equal positive rates across groups
- **Equalized Odds**: Equal TPR/FPR across groups
- **Individual Fairness**: Similar individuals receive similar scores

#### Business Metrics
- **Conversion Accuracy**: Actual vs predicted conversions
- **Revenue Impact**: Lead value optimization
- **Cost Efficiency**: Reduced manual review requirements

### Evaluation Reports

#### Base Insurance Evaluation
```python
from models.insurance_lead_scoring.evaluate import InsuranceModelEvaluator

evaluator = InsuranceModelEvaluator()
report = evaluator.generate_evaluation_report('data/test_leads.csv')
```

**Sample Output:**
```
Base Insurance Model Evaluation Report
=====================================
Overall R² Score: 0.872
RMSE: 12.3
Bias Alert: ✓ Passed (max bias: 0.08)
Deployment Ready: Yes
```

#### Healthcare Insurance Evaluation
```python
from models.healthcare_insurance_scoring.evaluate import HealthcareModelEvaluator

evaluator = HealthcareModelEvaluator()
report = evaluator.generate_healthcare_report('data/healthcare_test.csv')
```

#### Life Insurance Evaluation
```python
from models.life_insurance_scoring.evaluate import LifeInsuranceModelEvaluator

evaluator = LifeInsuranceModelEvaluator()
report = evaluator.generate_life_insurance_report('data/life_test.csv')
```

---

## Compliance & Ethics

### Data Privacy

#### GDPR Compliance
- **Consent Management**: Explicit consent tracking
- **Right to Erasure**: Data deletion capabilities
- **Data Minimization**: Only necessary data collection
- **Anonymization**: PII masking in processing

#### CCPA Compliance
- **Consumer Rights**: Data access and deletion
- **Opt-out Mechanisms**: Marketing communication controls
- **Transparency**: Clear data usage disclosure

#### HIPAA Compliance (Healthcare Model)
- **Health Data Protection**: Encrypted health information
- **Access Controls**: Role-based data access
- **Audit Trails**: Complete data access logging

### Fairness & Bias Prevention

#### Protected Classes
- Age, Gender, Race, Religion, Disability Status
- Geographic Location, Socioeconomic Status
- Health Status (where legally permissible)

#### Bias Detection
```python
# Automated bias monitoring
fairness_metrics = {
    'age_bias': 0.05,  # < 0.1 threshold
    'income_bias': 0.03,
    'location_bias': 0.07,
    'overall_bias_alert': False
}
```

#### Mitigation Strategies
- **Preprocessing**: Bias-aware feature selection
- **In-processing**: Fairness constraints during training
- **Post-processing**: Score adjustment for fairness
- **Monitoring**: Continuous bias detection

### Regulatory Compliance

#### Insurance Regulations
- **State Insurance Codes**: Compliance with state requirements
- **NAIC Guidelines**: National Association of Insurance Commissioners
- **Fair Credit Reporting Act**: Credit data usage compliance
- **Equal Credit Opportunity Act**: Non-discriminatory practices

#### Model Governance
- **Model Documentation**: Complete model lineage
- **Validation Framework**: Independent model validation
- **Risk Management**: Model risk assessment
- **Audit Trail**: Complete decision audit logs

---

## Deployment Guide

### System Requirements

#### Hardware Requirements
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 32+ GB
- **Storage**: 500+ GB SSD
- **Network**: 1+ Gbps bandwidth

#### Software Requirements
- **Python**: 3.8+
- **Dependencies**: See requirements.txt
- **Database**: PostgreSQL 12+
- **Cache**: Redis 6+
- **Web Server**: Nginx + Gunicorn

### Installation

#### 1. Environment Setup
```bash
# Create virtual environment
python -m venv insurance_scoring_env
source insurance_scoring_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Model Artifacts
```bash
# Download pre-trained models
mkdir -p models/{insurance_lead_scoring,healthcare_insurance_scoring,life_insurance_scoring}/artifacts

# Place model files:
# - model.pkl
# - scaler.pkl
# - label_encoders.pkl
```

#### 3. Configuration
```yaml
# config/app_config.yaml
database:
  host: localhost
  port: 5432
  name: insurance_scoring
  
redis:
  host: localhost
  port: 6379
  
models:
  base_model_path: models/insurance_lead_scoring/artifacts
  healthcare_model_path: models/healthcare_insurance_scoring/artifacts
  life_model_path: models/life_insurance_scoring/artifacts
```

#### 4. Database Setup
```sql
-- Create database
CREATE DATABASE insurance_scoring;

-- Create tables
CREATE TABLE lead_scores (
    lead_id VARCHAR(50) PRIMARY KEY,
    score FLOAT,
    model_version VARCHAR(20),
    timestamp TIMESTAMP,
    compliance_status VARCHAR(20)
);
```

### Production Deployment

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.insurance_scoring_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: insurance-scoring-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: insurance-scoring-api
  template:
    metadata:
      labels:
        app: insurance-scoring-api
    spec:
      containers:
      - name: api
        image: insurance-scoring:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Monitoring & Maintenance

#### Health Checks
```python
# Health check endpoints
GET /health              # Overall system health
GET /model-health        # Model loading status
GET /compliance-status   # Compliance monitoring
```

#### Performance Monitoring
- **Response Time**: < 100ms target
- **Throughput**: 1000+ requests/second
- **Error Rate**: < 0.1%
- **Model Accuracy**: Continuous monitoring

#### Model Updates
```bash
# Model retraining pipeline
python models/insurance_lead_scoring/train.py
python models/healthcare_insurance_scoring/train.py
python models/life_insurance_scoring/train.py

# Model validation
python models/*/evaluate.py

# Deployment
kubectl rollout restart deployment/insurance-scoring-api
```

### Security

#### API Security
- **Authentication**: JWT tokens
- **Rate Limiting**: 1000 requests/hour per client
- **Input Validation**: Comprehensive data validation
- **Encryption**: TLS 1.3 for all communications

#### Data Security
- **Encryption at Rest**: *****
- **Encryption in Transit**:****
- **Access Controls**: Role-based permissions
- **Audit Logging**: Complete access audit trail

---

## Support & Maintenance

### Documentation Updates
- **Model Changes**: Update documentation with each model version
- **API Changes**: Maintain API versioning and backward compatibility
- **Compliance Updates**: Regular compliance requirement reviews

### Contact Information
- **Technical Support**: tech-support@insurance-scoring.com
- **Compliance Questions**: compliance@insurance-scoring.com
- **Model Performance**: ml-team@insurance-scoring.com

### Version History
- **v1.0**: Initial release with base insurance model
- **v1.1**: Added healthcare insurance specialization
- **v1.2**: Added life insurance specialization
- **v1.3**: Enhanced compliance and fairness features

---

*Last Updated: January 2024*
*Document Version: 1.2*
*Model Versions: Base 1.0, Healthcare 1.0, Life Insurance 1.0*