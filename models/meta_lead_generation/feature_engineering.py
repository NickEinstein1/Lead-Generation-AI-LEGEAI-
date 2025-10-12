import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for lead scoring"""
    
    def __init__(self):
        self.feature_transformers = {}
        self.feature_stats = {}
    
    def create_polynomial_features(self, df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for specified columns"""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        
        for col in columns:
            if col in df.columns:
                poly_features = poly.fit_transform(df[[col]])
                feature_names = [f"{col}_poly_{i}" for i in range(1, poly_features.shape[1])]
                
                poly_df = pd.DataFrame(poly_features[:, 1:], columns=feature_names, index=df.index)
                df = pd.concat([df, poly_df], axis=1)
        
        return df
    
    def create_binned_features(self, df: pd.DataFrame, binning_config: Dict[str, Dict]) -> pd.DataFrame:
        """Create binned categorical features from continuous variables"""
        for col, config in binning_config.items():
            if col in df.columns:
                bins = config.get('bins', 5)
                labels = config.get('labels', None)
                strategy = config.get('strategy', 'uniform')
                
                if strategy == 'quantile':
                    df[f"{col}_binned"] = pd.qcut(df[col], q=bins, labels=labels, duplicates='drop')
                else:
                    df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=labels)
        
        return df
    
    def create_ratio_features(self, df: pd.DataFrame, ratio_config: List[Dict]) -> pd.DataFrame:
        """Create ratio features between different columns"""
        for config in ratio_config:
            numerator = config['numerator']
            denominator = config['denominator']
            name = config.get('name', f"{numerator}_to_{denominator}_ratio")
            
            if numerator in df.columns and denominator in df.columns:
                df[name] = df[numerator] / (df[denominator] + 1e-8)  # Add small epsilon to avoid division by zero
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, time_col: str, value_cols: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lag features for time series data"""
        if time_col in df.columns:
            df_sorted = df.sort_values(time_col)
            
            for col in value_cols:
                if col in df_sorted.columns:
                    for lag in lags:
                        df_sorted[f"{col}_lag_{lag}"] = df_sorted[col].shift(lag)
            
            return df_sorted
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, time_col: str, value_cols: List[str], windows: List[int]) -> pd.DataFrame:
        """Create rolling window features"""
        if time_col in df.columns:
            df_sorted = df.sort_values(time_col)
            
            for col in value_cols:
                if col in df_sorted.columns:
                    for window in windows:
                        df_sorted[f"{col}_rolling_mean_{window}"] = df_sorted[col].rolling(window=window).mean()
                        df_sorted[f"{col}_rolling_std_{window}"] = df_sorted[col].rolling(window=window).std()
                        df_sorted[f"{col}_rolling_max_{window}"] = df_sorted[col].rolling(window=window).max()
                        df_sorted[f"{col}_rolling_min_{window}"] = df_sorted[col].rolling(window=window).min()
            
            return df_sorted
        
        return df
    
    def create_text_features(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """Create features from text columns"""
        for col in text_columns:
            if col in df.columns:
                # Basic text features
                df[f"{col}_length"] = df[col].astype(str).str.len()
                df[f"{col}_word_count"] = df[col].astype(str).str.split().str.len()
                df[f"{col}_char_count"] = df[col].astype(str).str.replace(' ', '').str.len()
                
                # Sentiment analysis (simplified)
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing']
                
                df[f"{col}_positive_sentiment"] = df[col].astype(str).str.lower().apply(
                    lambda x: sum(word in x for word in positive_words)
                )
                df[f"{col}_negative_sentiment"] = df[col].astype(str).str.lower().apply(
                    lambda x: sum(word in x for word in negative_words)
                )
                df[f"{col}_sentiment_score"] = df[f"{col}_positive_sentiment"] - df[f"{col}_negative_sentiment"]
        
        return df
    
    def create_clustering_features(self, df: pd.DataFrame, cluster_columns: List[str], n_clusters: int = 5) -> pd.DataFrame:
        """Create clustering-based features"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        if all(col in df.columns for col in cluster_columns):
            # Prepare data for clustering
            cluster_data = df[cluster_columns].fillna(df[cluster_columns].median())
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add cluster features
            df['customer_cluster'] = clusters
            
            # Add distance to cluster centers
            distances = kmeans.transform(scaled_data)
            for i in range(n_clusters):
                df[f'distance_to_cluster_{i}'] = distances[:, i]
            
            # Add cluster-based statistics
            cluster_stats = df.groupby('customer_cluster')[cluster_columns].agg(['mean', 'std']).fillna(0)
            
            for col in cluster_columns:
                df[f'{col}_cluster_mean'] = df['customer_cluster'].map(cluster_stats[(col, 'mean')])
                df[f'{col}_cluster_std'] = df['customer_cluster'].map(cluster_stats[(col, 'std')])
        
        return df
    
    def create_anomaly_features(self, df: pd.DataFrame, anomaly_columns: List[str]) -> pd.DataFrame:
        """Create anomaly detection features"""
        from sklearn.ensemble import IsolationForest
        
        if all(col in df.columns for col in anomaly_columns):
            # Prepare data
            anomaly_data = df[anomaly_columns].fillna(df[anomaly_columns].median())
            
            # Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.fit_predict(anomaly_data)
            anomaly_scores_continuous = iso_forest.decision_function(anomaly_data)
            
            df['is_anomaly'] = (anomaly_scores == -1).astype(int)
            df['anomaly_score'] = anomaly_scores_continuous
            
            # Statistical outliers
            for col in anomaly_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df[f'{col}_is_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
                df[f'{col}_outlier_distance'] = np.maximum(
                    lower_bound - df[col], 
                    df[col] - upper_bound
                ).clip(lower=0)
        
        return df
    
    def create_frequency_encoding(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Create frequency encoding for categorical variables"""
        for col in categorical_columns:
            if col in df.columns:
                freq_map = df[col].value_counts().to_dict()
                df[f'{col}_frequency'] = df[col].map(freq_map)
                df[f'{col}_frequency_normalized'] = df[f'{col}_frequency'] / len(df)
        
        return df
    
    def create_target_encoding(self, df: pd.DataFrame, categorical_columns: List[str], 
                             target_column: str, smoothing: float = 1.0) -> pd.DataFrame:
        """Create target encoding for categorical variables"""
        if target_column in df.columns:
            global_mean = df[target_column].mean()
            
            for col in categorical_columns:
                if col in df.columns:
                    # Calculate target statistics by category
                    target_stats = df.groupby(col)[target_column].agg(['count', 'mean']).reset_index()
                    target_stats.columns = [col, 'count', 'target_mean']
                    
                    # Apply smoothing
                    target_stats['target_encoded'] = (
                        (target_stats['target_mean'] * target_stats['count'] + global_mean * smoothing) /
                        (target_stats['count'] + smoothing)
                    )
                    
                    # Map back to original dataframe
                    encoding_map = target_stats.set_index(col)['target_encoded'].to_dict()
                    df[f'{col}_target_encoded'] = df[col].map(encoding_map).fillna(global_mean)
        
        return df

# Configuration for feature engineering
FEATURE_ENGINEERING_CONFIG = {
    'polynomial_features': {
        'columns': ['age', 'income', 'credit_score'],
        'degree': 2
    },
    'binning_config': {
        'age': {
            'bins': [0, 25, 35, 45, 55, 65, 100],
            'labels': ['young', 'young_adult', 'adult', 'middle_age', 'senior', 'elderly'],
            'strategy': 'custom'
        },
        'income': {
            'bins': 5,
            'strategy': 'quantile'
        },
        'credit_score': {
            'bins': [300, 580, 670, 740, 800, 850],
            'labels': ['poor', 'fair', 'good', 'very_good', 'excellent'],
            'strategy': 'custom'
        }
    },
    'ratio_features': [
        {'numerator': 'income', 'denominator': 'age', 'name': 'income_per_age'},
        {'numerator': 'quote_requests_30d', 'denominator': 'website_visits_30d', 'name': 'conversion_intent'},
        {'numerator': 'dependents_count', 'denominator': 'income', 'name': 'dependency_burden'}
    ],
    'clustering_features': {
        'columns': ['age', 'income', 'credit_score', 'website_visits_30d'],
        'n_clusters': 5
    },
    'anomaly_features': {
        'columns': ['age', 'income', 'credit_score', 'quote_requests_30d']
    },
    'text_features': {
        'columns': ['comments', 'feedback', 'notes']
    }
}