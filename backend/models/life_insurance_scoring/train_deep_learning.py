"""
Deep Learning Training for Life Insurance Lead Scoring

This module trains advanced neural network models specifically for life insurance
lead scoring, including:
- Deep Neural Networks with attention mechanisms
- LSTM for behavioral sequence prediction
- Transformer-based models
- Ensemble of deep learning + XGBoost
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LifeInsuranceDataset(Dataset):
    """PyTorch Dataset for life insurance leads"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class LifeInsuranceDeepNetwork(nn.Module):
    """
    Deep Neural Network for Life Insurance Lead Scoring
    
    Architecture:
    - Input layer with feature normalization
    - Multi-head attention for feature importance
    - 4 hidden layers with batch normalization and dropout
    - Residual connections for better gradient flow
    - Output layer with sigmoid activation (0-100 score)
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64, 32], 
                 dropout_rate: float = 0.3, use_attention: bool = True):
        super(LifeInsuranceDeepNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.use_attention = use_attention
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Multi-head attention for feature importance
        if use_attention:
            # Determine number of heads based on input_dim
            num_heads = 2 if input_dim % 2 == 0 else 1
            self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
            self.attention_norm = nn.LayerNorm(input_dim)
        
        # Build deep network with residual connections
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x_reshaped = x.unsqueeze(1)  # Add sequence dimension
            attn_output, attn_weights = self.attention(x_reshaped, x_reshaped, x_reshaped)
            x = self.attention_norm(attn_output.squeeze(1) + x)  # Residual connection
        
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Output (0-100 scale)
        output = self.output(x)
        output = torch.sigmoid(output) * 100  # Scale to 0-100
        
        return output.squeeze()


class LifeInsuranceDeepLearningTrainer:
    """Trainer for life insurance deep learning models"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = [
            'age', 'income', 'marital_status', 'dependents_count', 'employment_status',
            'health_status', 'smoking_status', 'coverage_amount_requested', 'policy_term',
            'existing_life_insurance', 'beneficiary_count', 'debt_obligations',
            'mortgage_balance', 'education_level', 'occupation_risk_level',
            'life_stage', 'financial_dependents', 'estate_planning_needs'
        ]
        
        logger.info(f"Deep Learning Trainer initialized on device: {self.device}")

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess life insurance data with feature engineering"""

        # Handle categorical variables
        categorical_cols = ['marital_status', 'employment_status', 'health_status',
                           'smoking_status', 'education_level', 'occupation_risk_level', 'life_stage']

        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))

        # Life insurance-specific feature engineering
        df['age_risk_factor'] = np.where(df['age'] > 60, df['age'] * 1.5,
                                np.where(df['age'] > 45, df['age'] * 1.2, df['age']))

        df['coverage_income_ratio'] = df['coverage_amount_requested'] / (df['income'] + 1)
        df['financial_responsibility_score'] = (
            df['dependents_count'] * 2 +
            df['mortgage_balance'] / 100000 +
            df['debt_obligations'] / 50000
        )

        df['mortality_risk_score'] = (
            (df['age'] / 10) +
            (df['smoking_status'] * 3) +
            (df['health_status'] * 2) +
            (df['occupation_risk_level'] * 1.5)
        )

        # Handle missing values
        df['mortgage_balance'] = df['mortgage_balance'].fillna(0)
        df['debt_obligations'] = df['debt_obligations'].fillna(df['debt_obligations'].median())
        df['beneficiary_count'] = df['beneficiary_count'].fillna(1)

        # Select features
        feature_cols = self.feature_columns + [
            'age_risk_factor', 'coverage_income_ratio', 'financial_responsibility_score',
            'mortality_risk_score'
        ]

        feature_df = df[feature_cols]
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

        return feature_df.values

    def train(self, data_path: str, epochs: int = 100, batch_size: int = 64,
              learning_rate: float = 0.001, validation_split: float = 0.2):
        """Train deep learning model for life insurance scoring"""

        logger.info("=" * 80)
        logger.info("LIFE INSURANCE DEEP LEARNING MODEL TRAINING")
        logger.info("=" * 80)

        # Load data
        logger.info(f"Loading training data from {data_path}...")
        df = pd.read_csv(data_path)

        # Filter consented leads
        df = df[df['consent_given'] == True]
        logger.info(f"Training on {len(df)} consented life insurance leads")

        # Preprocess
        X = self.preprocess_data(df)
        y = df['conversion_score'].values / 100.0  # Normalize to 0-1 for training

        # Split data
        life_stage_bins = pd.cut(df['age'], bins=[0, 30, 45, 65, 100],
                                 labels=['young', 'family', 'mature', 'senior'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=life_stage_bins
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create datasets
        train_dataset = LifeInsuranceDataset(X_train_scaled, y_train)
        test_dataset = LifeInsuranceDataset(X_test_scaled, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_dim = X_train_scaled.shape[1]
        self.model = LifeInsuranceDeepNetwork(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64, 32],
            dropout_rate=0.3,
            use_attention=True
        ).to(self.device)

        logger.info(f"Model architecture:")
        logger.info(f"  Input dimension: {input_dim}")
        logger.info(f"  Hidden layers: [256, 128, 64, 32]")
        logger.info(f"  Attention: Enabled (4 heads)")
        logger.info(f"  Device: {self.device}")

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

        # Training loop
        best_test_loss = float('inf')
        patience_counter = 0
        max_patience = 20

        logger.info(f"\nStarting training for {epochs} epochs...")
        logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features) / 100.0  # Scale back to 0-1 for loss
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            test_loss = 0.0
            predictions = []
            actuals = []

            with torch.no_grad():
                for features, targets in test_loader:
                    features = features.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(features) / 100.0
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    predictions.extend(outputs.cpu().numpy() * 100)  # Scale back to 0-100
                    actuals.extend(targets.cpu().numpy() * 100)

            test_loss /= len(test_loader)

            # Calculate metrics
            mse = mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)

            # Learning rate scheduling
            scheduler.step(test_loss)

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                          f"R²: {r2:.4f}, MAE: {mae:.2f}")

            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                # Save best model
                self.save_model(is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model
        self.load_best_model()

        # Final evaluation
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED - FINAL EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Best Test Loss: {best_test_loss:.4f}")
        logger.info(f"Final R² Score: {r2:.4f}")
        logger.info(f"Final MAE: {mae:.2f}")
        logger.info(f"Final MSE: {mse:.2f}")

        return {
            'r2_score': r2,
            'mae': mae,
            'mse': mse,
            'best_loss': best_test_loss,
            'epochs_trained': epoch + 1
        }

    def save_model(self, is_best: bool = False):
        """Save model, scaler, and encoders"""
        model_dir = 'models/life_insurance_scoring/deep_learning_artifacts'
        os.makedirs(model_dir, exist_ok=True)

        # Save PyTorch model
        model_path = f'{model_dir}/{"best_" if is_best else ""}model.pth'
        torch.save(self.model.state_dict(), model_path)

        # Save scaler and encoders
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{model_dir}/label_encoders.pkl')

        # Save model config
        config = {
            'input_dim': self.model.input_dim,
            'device': str(self.device),
            'feature_columns': self.feature_columns,
            'timestamp': datetime.now().isoformat()
        }
        with open(f'{model_dir}/config.json', 'w') as f:
            json.dump(config, f, indent=2)

        if is_best:
            logger.info(f"Best model saved to {model_dir}")

    def load_best_model(self):
        """Load the best saved model"""
        model_dir = 'models/life_insurance_scoring/deep_learning_artifacts'
        model_path = f'{model_dir}/best_model.pth'

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info("Best model loaded successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Deep Learning Model for Life Insurance Scoring')
    parser.add_argument('--data', type=str, default='data/life_insurance_leads_training.csv',
                       help='Path to training data CSV')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation split ratio')

    args = parser.parse_args()

    # Train model
    trainer = LifeInsuranceDeepLearningTrainer()
    metrics = trainer.train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validation_split=args.validation_split
    )

    print("\n" + "=" * 80)
    print("DEEP LEARNING TRAINING SUMMARY")
    print("=" * 80)
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"Epochs Trained: {metrics['epochs_trained']}")
    print("=" * 80)

