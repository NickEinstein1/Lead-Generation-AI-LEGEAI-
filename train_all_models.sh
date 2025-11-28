#!/bin/bash

# Train All ML Models - macOS/Linux Bash Script
# This script trains all insurance lead scoring models

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}🤖 LEGEAI - Train All ML Models${NC}"
echo -e "${CYAN}=================================${NC}"
echo ""

# Set PYTHONPATH
export PYTHONPATH=.
echo -e "${GREEN}✅ PYTHONPATH set to current directory${NC}"
echo ""

# Check if training data exists
echo -e "${YELLOW}📊 Checking training data...${NC}"
if [ ! -f "data/insurance_leads_training.csv" ]; then
    echo -e "${YELLOW}⚠️  Training data not found. Generating...${NC}"
    python generate_training_data.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to generate training data${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Training data generated${NC}"
else
    echo -e "${GREEN}✅ Training data found${NC}"
fi
echo ""

# Train Insurance Lead Scoring (XGBoost)
echo -e "${CYAN}1️⃣  Training Insurance Lead Scoring (XGBoost)...${NC}"
python backend/models/insurance_lead_scoring/train.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Insurance Lead Scoring model trained${NC}"
else
    echo -e "${YELLOW}⚠️  Insurance Lead Scoring training failed (continuing...)${NC}"
fi
echo ""

# Train Life Insurance XGBoost
echo -e "${CYAN}2️⃣  Training Life Insurance (XGBoost)...${NC}"
python backend/models/life_insurance_scoring/train.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Life Insurance XGBoost model trained${NC}"
else
    echo -e "${YELLOW}⚠️  Life Insurance XGBoost training failed (continuing...)${NC}"
fi
echo ""

# Train Life Insurance Deep Learning
echo -e "${CYAN}3️⃣  Training Life Insurance (Deep Learning)...${NC}"
python backend/models/life_insurance_scoring/train_deep_learning.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Life Insurance Deep Learning model trained${NC}"
else
    echo -e "${YELLOW}⚠️  Life Insurance Deep Learning training failed (continuing...)${NC}"
fi
echo ""

# Train Auto Insurance Deep Learning
echo -e "${CYAN}4️⃣  Training Auto Insurance (Deep Learning)...${NC}"
python backend/models/auto_insurance_scoring/train_deep_learning.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Auto Insurance Deep Learning model trained${NC}"
else
    echo -e "${YELLOW}⚠️  Auto Insurance Deep Learning training failed (continuing...)${NC}"
fi
echo ""

# Train Home Insurance Deep Learning
echo -e "${CYAN}5️⃣  Training Home Insurance (Deep Learning)...${NC}"
python backend/models/home_insurance_scoring/train_deep_learning.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Home Insurance Deep Learning model trained${NC}"
else
    echo -e "${YELLOW}⚠️  Home Insurance Deep Learning training failed (continuing...)${NC}"
fi
echo ""

# Train Health Insurance Deep Learning
echo -e "${CYAN}6️⃣  Training Health Insurance (Deep Learning)...${NC}"
python backend/models/healthcare_insurance_scoring/train_deep_learning.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Health Insurance Deep Learning model trained${NC}"
else
    echo -e "${YELLOW}⚠️  Health Insurance Deep Learning training failed (continuing...)${NC}"
fi
echo ""

# Summary
echo -e "${CYAN}=================================${NC}"
echo -e "${GREEN}🎉 Training Complete!${NC}"
echo -e "${CYAN}=================================${NC}"
echo ""
echo -e "${YELLOW}📁 Model files saved to:${NC}"
echo "   - models/insurance_lead_scoring/artifacts/"
echo "   - models/life_insurance_scoring/artifacts/"
echo "   - models/life_insurance_scoring/deep_learning_artifacts/"
echo "   - backend/models/auto_insurance_scoring/saved_models/"
echo "   - backend/models/home_insurance_scoring/saved_models/"
echo "   - backend/models/healthcare_insurance_scoring/saved_models/"
echo ""
echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo "   1. Start the backend: python -m uvicorn backend.api.main:app --reload"
echo "   2. Verify no model loading errors in logs"
echo "   3. Test lead scoring endpoints"
echo ""

