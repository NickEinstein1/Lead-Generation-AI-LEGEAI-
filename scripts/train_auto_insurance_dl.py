"""
Script to train the Auto Insurance Deep Learning Model

This script:
1. Filters auto insurance leads from the general insurance dataset
2. Trains the PyTorch deep learning model
3. Saves the trained model and preprocessors
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from backend.models.auto_insurance_scoring.train_deep_learning import AutoInsuranceDeepLearningTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_auto_insurance_data(input_path: str, output_path: str):
    """Filter and prepare auto insurance data"""
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Filter only auto insurance leads
    auto_df = df[df['policy_type'] == 'auto'].copy()
    logger.info(f"Found {len(auto_df)} auto insurance leads")
    
    # Ensure required columns exist
    required_columns = [
        'lead_id', 'age', 'income', 'policy_type', 'quote_requests_30d',
        'social_engagement_score', 'location_risk_score', 'previous_insurance',
        'credit_score_proxy', 'consent_given', 'conversion_score'
    ]
    
    missing_columns = [col for col in required_columns if col not in auto_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Save filtered data
    auto_df.to_csv(output_path, index=False)
    logger.info(f"Auto insurance data saved to {output_path}")
    
    return output_path


def main():
    """Main training function"""
    logger.info("=" * 80)
    logger.info("AUTO INSURANCE DEEP LEARNING MODEL TRAINING")
    logger.info("=" * 80)
    
    # Paths
    input_data_path = "data/insurance_leads_training.csv"
    auto_data_path = "data/auto_insurance_leads_training.csv"
    
    # Step 1: Prepare auto insurance data
    logger.info("\nStep 1: Preparing auto insurance data...")
    try:
        data_path = prepare_auto_insurance_data(input_data_path, auto_data_path)
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        return
    
    # Step 2: Initialize trainer
    logger.info("\nStep 2: Initializing Deep Learning trainer...")
    trainer = AutoInsuranceDeepLearningTrainer()
    
    # Step 3: Train model
    logger.info("\nStep 3: Training Deep Learning model...")
    logger.info("This may take several minutes...")
    
    try:
        results = trainer.train(
            data_path=data_path,
            epochs=100,
            batch_size=32,
            learning_rate=0.001
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Mean Squared Error (MSE): {results['mse']:.2f}")
        logger.info(f"R² Score: {results['r2_score']:.3f}")
        logger.info(f"Mean Absolute Error (MAE): {results['mae']:.2f}")
        logger.info("=" * 80)
        
        # Interpret R² score
        r2 = results['r2_score']
        if r2 > 0.7:
            logger.info("✅ Excellent model performance!")
        elif r2 > 0.5:
            logger.info("✅ Good model performance!")
        elif r2 > 0.3:
            logger.info("⚠️  Moderate model performance")
        else:
            logger.info("⚠️  Model may need improvement")
        
        logger.info("\nModel saved to: backend/models/auto_insurance_scoring/saved_models/")
        logger.info("\nYou can now use the ensemble scorer with both XGBoost and Deep Learning!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

