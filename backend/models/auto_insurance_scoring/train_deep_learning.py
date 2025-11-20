"""
Auto Insurance Deep Learning Model Training

This module trains a PyTorch neural network for auto insurance lead scoring.
Features:
- Multi-layer neural network with attention mechanism
- Dropout for regularization
- Batch normalization for stability
- Early stopping to prevent overfitting
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoInsuranceDataset(Dataset):
    """PyTorch Dataset for auto insurance leads"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AutoInsuranceNeuralNetwork(nn.Module):
    """
    Deep Learning model for auto insurance lead scoring
    
    Architecture:
    - Input layer (auto insurance features)
    - 3 hidden layers with batch normalization and dropout
    - Attention mechanism for feature importance
    - Output layer (conversion score 0-100)
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(AutoInsuranceNeuralNetwork, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        
        # Hidden layers
        self.hidden1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        
        self.hidden2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_sizes[2], 1)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[2], 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Hidden layer 1
        x = self.hidden1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Hidden layer 2
        x = self.hidden2(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Attention weights
        attention_weights = self.sigmoid(self.attention(x))
        x = x * attention_weights
        
        # Output (0-100 range)
        x = self.output_layer(x)
        x = torch.clamp(x, 0, 100)  # Ensure output is in valid range
        
        return x.squeeze()


class AutoInsuranceDeepLearningTrainer:
    """Trainer for auto insurance deep learning model"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_columns = [
            'age', 'income', 'policy_type', 'quote_requests_30d',
            'social_engagement_score', 'location_risk_score',
            'previous_insurance', 'credit_score_proxy'
        ]
        logger.info(f"Using device: {self.device}")
    
    def preprocess_data(self, df):
        """Preprocess auto insurance data"""
        # Handle categorical variables
        categorical_cols = ['policy_type', 'previous_insurance']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))

        # Feature engineering
        df['age_income_ratio'] = df['age'] / (df['income'] / 1000)
        df['engagement_per_request'] = df['social_engagement_score'] / (df['quote_requests_30d'] + 1)
        df['risk_engagement_product'] = df['location_risk_score'] * df['social_engagement_score']

        # Select features
        feature_df = df[self.feature_columns + ['age_income_ratio', 'engagement_per_request', 'risk_engagement_product']]
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

        return feature_df

    def train(self, data_path, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the deep learning model"""
        logger.info("Loading auto insurance training data...")
        df = pd.read_csv(data_path)

        # Ensure consent compliance
        df = df[df['consent_given'] == True]
        logger.info(f"Training on {len(df)} consented auto insurance leads")

        # Preprocess
        X = self.preprocess_data(df)
        y = df['conversion_score'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create datasets
        train_dataset = AutoInsuranceDataset(X_train_scaled, y_train)
        test_dataset = AutoInsuranceDataset(X_test_scaled, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # Initialize model
        input_size = X_train_scaled.shape[1]
        self.model = AutoInsuranceNeuralNetwork(input_size).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        # Training loop
        logger.info("Training deep learning model...")
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(test_loader)

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            y_pred = self.model(X_test_tensor).cpu().numpy()

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"Auto Insurance Deep Learning Model Performance:")
        logger.info(f"MSE: {mse:.2f}")
        logger.info(f"R² Score: {r2:.3f}")
        logger.info(f"MAE: {mae:.2f}")

        return {'mse': mse, 'r2_score': r2, 'mae': mae}

    def save_model(self):
        """Save model and preprocessors"""
        model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        os.makedirs(model_dir, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(model_dir, 'auto_insurance_dl_model.pth'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'auto_insurance_dl_scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'auto_insurance_dl_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(model_dir, 'auto_insurance_dl_features.pkl'))

        logger.info(f"Auto insurance deep learning model saved to {model_dir}")


if __name__ == "__main__":
    trainer = AutoInsuranceDeepLearningTrainer()

    # Train model (you'll need to provide the data path)
    data_path = "data/auto_insurance_leads.csv"
    if os.path.exists(data_path):
        results = trainer.train(data_path, epochs=100, batch_size=32, learning_rate=0.001)
        print(f"\nTraining Results: {results}")
    else:
        logger.warning(f"Training data not found at {data_path}")
        logger.info("Model structure created. Provide training data to train the model.")

