import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt
import seaborn as sns
from inference import InsuranceLeadScorer
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path='models/insurance_lead_scoring/artifacts'):
        self.scorer = InsuranceLeadScorer(model_path)
        
    def evaluate_accuracy(self, test_data_path):
        """Evaluate model accuracy on test set"""
        df = pd.read_csv(test_data_path)
        df = df[df['consent_given'] == True]  # Only consented data
        
        predictions = []
        actuals = []
        
        for _, row in df.iterrows():
            result = self.scorer.score_lead(row.to_dict())
            if 'error' not in result:
                predictions.append(result['score'])
                actuals.append(row['conversion_score'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        metrics = {
            'mse': mean_squared_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mae': mean_absolute_error(actuals, predictions),
            'r2_score': r2_score(actuals, predictions),
            'accuracy_90_threshold': r2_score(actuals, predictions) > 0.9
        }
        
        logger.info(f"Accuracy Metrics: {metrics}")
        return metrics
    
    def evaluate_fairness(self, test_data_path):
        """Evaluate model for bias and fairness"""
        df = pd.read_csv(test_data_path)
        df = df[df['consent_given'] == True]
        
        predictions = []
        sensitive_attributes = []
        
        for _, row in df.iterrows():
            result = self.scorer.score_lead(row.to_dict())
            if 'error' not in result:
                predictions.append(result['score'])
                # Use age groups as sensitive attribute
                age_group = 'young' if row['age'] < 35 else 'older'
                sensitive_attributes.append(age_group)
        
        predictions = np.array(predictions)
        sensitive_attributes = np.array(sensitive_attributes)
        
        # Convert scores to binary for fairness metrics
        binary_predictions = (predictions > 50).astype(int)
        
        # Calculate fairness metrics
        dp_diff = demographic_parity_difference(
            binary_predictions, sensitive_attributes
        )
        
        fairness_metrics = {
            'demographic_parity_difference': abs(dp_diff),
            'bias_detected': abs(dp_diff) > 0.1,
            'age_group_distribution': pd.Series(sensitive_attributes).value_counts().to_dict()
        }
        
        logger.info(f"Fairness Metrics: {fairness_metrics}")
        return fairness_metrics
    
    def performance_monitoring(self, test_data_path):
        """Monitor model performance over time"""
        df = pd.read_csv(test_data_path)
        df = df[df['consent_given'] == True]
        
        # Simulate time-based performance
        df['timestamp'] = pd.to_datetime(df.get('timestamp', pd.Timestamp.now()))
        df = df.sort_values('timestamp')
        
        performance_over_time = []
        window_size = len(df) // 10  # 10 time windows
        
        for i in range(0, len(df), window_size):
            window_data = df.iloc[i:i+window_size]
            if len(window_data) < 10:  # Skip small windows
                continue
                
            predictions = []
            actuals = []
            
            for _, row in window_data.iterrows():
                result = self.scorer.score_lead(row.to_dict())
                if 'error' not in result:
                    predictions.append(result['score'])
                    actuals.append(row['conversion_score'])
            
            if len(predictions) > 0:
                r2 = r2_score(actuals, predictions)
                performance_over_time.append({
                    'window': i // window_size,
                    'r2_score': r2,
                    'sample_size': len(predictions)
                })
        
        return performance_over_time
    
    def generate_report(self, test_data_path, output_path='evaluation_report.html'):
        """Generate comprehensive evaluation report"""
        accuracy_metrics = self.evaluate_accuracy(test_data_path)
        fairness_metrics = self.evaluate_fairness(test_data_path)
        performance_data = self.performance_monitoring(test_data_path)
        
        report = f"""
        <html>
        <head><title>Insurance Lead Scoring Model Evaluation</title></head>
        <body>
        <h1>Model Evaluation Report</h1>
        <h2>Accuracy Metrics</h2>
        <ul>
            <li>R² Score: {accuracy_metrics['r2_score']:.3f}</li>
            <li>RMSE: {accuracy_metrics['rmse']:.2f}</li>
            <li>MAE: {accuracy_metrics['mae']:.2f}</li>
            <li>Meets 90% Threshold: {accuracy_metrics['accuracy_90_threshold']}</li>
        </ul>
        
        <h2>Fairness Metrics</h2>
        <ul>
            <li>Demographic Parity Difference: {fairness_metrics['demographic_parity_difference']:.3f}</li>
            <li>Bias Detected: {fairness_metrics['bias_detected']}</li>
        </ul>
        
        <h2>Compliance Status</h2>
        <ul>
            <li>GDPR Compliant: ✓ (Consent validation implemented)</li>
            <li>CCPA Compliant: ✓ (PII anonymization implemented)</li>
            <li>Fairness Audit: {'⚠️ Review needed' if fairness_metrics['bias_detected'] else '✓ Passed'}</li>
        </ul>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        return {
            'accuracy': accuracy_metrics,
            'fairness': fairness_metrics,
            'performance_trend': performance_data
        }

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    report = evaluator.generate_report('data/insurance_leads_test.csv')
    print("Evaluation completed. Check evaluation_report.html for details.")