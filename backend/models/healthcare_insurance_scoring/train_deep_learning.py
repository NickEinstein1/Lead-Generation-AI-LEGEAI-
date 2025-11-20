"""
Deep Learning Model Training for Health Insurance Lead Scoring
PyTorch-based neural network with batch normalization and dropout
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthInsuranceDataset(Dataset):
    """PyTorch Dataset for health insurance leads"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HealthInsuranceNeuralNetwork(nn.Module):
    """Neural Network for Health Insurance Lead Scoring"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(HealthInsuranceNeuralNetwork, self).__init__()
        
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Hidden layer 1
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Hidden layer 2
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_sizes[2], 1)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Handle single sample case for batch normalization
        single_sample = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        elif x.size(0) == 1 and self.training:
            pass
        
        # Layer 1
        x = self.fc1(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc4(x)
        
        if single_sample:
            return x.squeeze()
        return x.squeeze(-1)


class HealthInsuranceDeepLearningTrainer:
    """Trainer for Health Insurance Deep Learning Model"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        
        # Feature columns for health insurance
        self.feature_columns = [
            'age', 'income', 'policy_type', 'quote_requests_30d',
            'social_engagement_score', 'location_risk_score',
            'previous_insurance', 'credit_score_proxy'
        ]
        
    def prepare_health_insurance_data(self, input_path: str, output_path: str):
        """Filter health insurance data from general dataset"""
        logger.info(f"Filtering health insurance data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Filter only health insurance leads
        health_df = df[df['policy_type'] == 'health'].copy()
        logger.info(f"Found {len(health_df)} health insurance leads")
        
        # Save filtered data
        health_df.to_csv(output_path, index=False)
        logger.info(f"Saved health insurance data to {output_path}")
        
        return output_path
    
    def preprocess_data(self, df):
        """Preprocess health insurance data"""
        # Handle categorical variables
        categorical_cols = ['policy_type', 'previous_insurance']
        for col in categorical_cols:
            if col in df.columns:
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
        feature_cols = self.feature_columns + ['age_income_ratio', 'engagement_per_request', 'risk_engagement_product']
        X = df[feature_cols].values

        return X

    def train(self, data_path: str, epochs=200, batch_size=32, learning_rate=0.001):
        """Train the deep learning model"""
        logger.info("Loading health insurance training data...")
        df = pd.read_csv(data_path)

        # Ensure consent compliance
        df = df[df['consent_given'] == True]
        logger.info(f"Training on {len(df)} consented health insurance leads")

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
        train_dataset = HealthInsuranceDataset(X_train_scaled, y_train)
        test_dataset = HealthInsuranceDataset(X_test_scaled, y_test)

        # Create data loaders with drop_last=True to avoid batch norm issues
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # Initialize model
        input_size = X_train_scaled.shape[1]
        self.model = HealthInsuranceNeuralNetwork(input_size).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        patience = 20

        logger.info("Starting training...")
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
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
            test_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)

            scheduler.step(avg_test_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

            # Early stopping
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            predictions = self.model(X_test_tensor).cpu().numpy()

        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

        logger.info(f"\nHealth Insurance Deep Learning Model Performance:")
        logger.info(f"MSE: {mse:.2f}")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"R² Score: {r2:.3f}")

        # Save model
        self.save_model()

        return {'mse': mse, 'mae': mae, 'r2_score': r2}

    def save_model(self):
        """Save the trained model"""
        model_dir = 'backend/models/healthcare_insurance_scoring/saved_models'
        os.makedirs(model_dir, exist_ok=True)

        # Save PyTorch model
        torch.save(self.model.state_dict(), f'{model_dir}/health_insurance_dl_model.pth')

        # Save scaler and encoders
        joblib.dump(self.scaler, f'{model_dir}/health_insurance_dl_scaler.pkl')
        joblib.dump(self.label_encoders, f'{model_dir}/health_insurance_dl_label_encoders.pkl')
        joblib.dump(self.feature_columns, f'{model_dir}/health_insurance_dl_features.pkl')

        logger.info(f"Model saved to {model_dir}")


if __name__ == "__main__":
    trainer = HealthInsuranceDeepLearningTrainer()

    # Prepare data
    input_path = 'data/insurance_leads_training.csv'
    output_path = 'data/health_insurance_leads_training.csv'
    trainer.prepare_health_insurance_data(input_path, output_path)

    # Train model
    metrics = trainer.train(output_path)
    print(f"\nHealth Insurance Deep Learning Training Complete!")
    print(f"R² Score: {metrics['r2_score']:.3f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")

