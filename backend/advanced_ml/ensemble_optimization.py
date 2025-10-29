"""
Ensemble Model Optimization System

Advanced ensemble methods with automated hyperparameter tuning,
model selection, and continuous improvement capabilities.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    VotingRegressor, BaggingRegressor, AdaBoostRegressor
)
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, 
    TimeSeriesSplit, StratifiedKFold
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import joblib
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for individual models in ensemble"""
    name: str
    model_class: Any
    param_space: Dict[str, Any]
    weight: float = 1.0
    is_active: bool = True
    performance_history: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class EnsemblePerformance:
    """Ensemble performance metrics"""
    r2_score: float
    rmse: float
    mae: float
    cv_scores: List[float]
    individual_scores: Dict[str, float]
    ensemble_weights: Dict[str, float]
    training_time: float
    prediction_time: float
    model_count: int
    improvement_over_best_individual: float

class AdvancedEnsembleOptimizer:
    """Advanced ensemble optimization with automated tuning"""
    
    def __init__(self, optimization_budget: int = 100, n_jobs: int = -1):
        self.optimization_budget = optimization_budget
        self.n_jobs = n_jobs
        
        # Model configurations
        self.model_configs = self._initialize_model_configs()
        
        # Ensemble components
        self.trained_models = {}
        self.ensemble_weights = {}
        self.meta_learner = None
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.performance_history = []
        self.optimization_history = []
        
        # Continuous learning
        self.performance_threshold = 0.85
        self.retrain_frequency = timedelta(days=7)
        self.last_retrain = datetime.now()
        
        logger.info("Advanced Ensemble Optimizer initialized")
    
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations with parameter spaces"""
        
        configs = {
            'xgboost': ModelConfig(
                name='xgboost',
                model_class=xgb.XGBRegressor,
                param_space={
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 4, 5, 6, 7, 8],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0]
                }
            ),
            'lightgbm': ModelConfig(
                name='lightgbm',
                model_class=lgb.LGBMRegressor,
                param_space={
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 4, 5, 6, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0],
                    'num_leaves': [31, 50, 100, 150]
                }
            ),
            'catboost': ModelConfig(
                name='catboost',
                model_class=CatBoostRegressor,
                param_space={
                    'iterations': [100, 200, 300, 500],
                    'depth': [4, 5, 6, 7, 8],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128, 255],
                    'bagging_temperature': [0, 0.5, 1.0]
                }
            ),
            'random_forest': ModelConfig(
                name='random_forest',
                model_class=RandomForestRegressor,
                param_space={
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
            ),
            'gradient_boosting': ModelConfig(
                name='gradient_boosting',
                model_class=GradientBoostingRegressor,
                param_space={
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            ),
            'extra_trees': ModelConfig(
                name='extra_trees',
                model_class=ExtraTreesRegressor,
                param_space={
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            ),
            'elastic_net': ModelConfig(
                name='elastic_net',
                model_class=ElasticNet,
                param_space={
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [1000, 2000, 5000]
                }
            ),
            'svr': ModelConfig(
                name='svr',
                model_class=SVR,
                param_space={
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'epsilon': [0.01, 0.1, 0.2, 0.5],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            )
        }
        
        return configs
    
    async def optimize_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                              cv_folds: int = 5, 
                              optimization_method: str = 'optuna') -> EnsemblePerformance:
        """Optimize ensemble with advanced hyperparameter tuning"""
        
        try:
            logger.info("Starting ensemble optimization...")
            start_time = datetime.now()
            
            # Prepare data
            X_scaled = self.scaler.fit_transform(X)
            
            if optimization_method == 'optuna':
                performance = await self._optimize_with_optuna(X_scaled, y, cv_folds)
            elif optimization_method == 'grid_search':
                performance = await self._optimize_with_grid_search(X_scaled, y, cv_folds)
            else:
                performance = await self._optimize_with_random_search(X_scaled, y, cv_folds)
            
            # Train final ensemble
            await self._train_final_ensemble(X_scaled, y)
            
            # Calculate performance
            training_time = (datetime.now() - start_time).total_seconds()
            performance.training_time = training_time
            
            # Store performance
            self.performance_history.append(performance)
            
            logger.info(f"Ensemble optimization completed. R2: {performance.r2_score:.4f}")
            return performance
            
        except Exception as e:
            logger.error(f"Error optimizing ensemble: {e}")
            raise
    
    async def _optimize_with_optuna(self, X: np.ndarray, y: pd.Series, 
                                  cv_folds: int) -> EnsemblePerformance:
        """Optimize ensemble using Optuna"""
        
        def objective(trial):
            # Select active models
            active_models = {}
            
            for model_name, config in self.model_configs.items():
                if trial.suggest_categorical(f'{model_name}_active', [True, False]):
                    # Sample hyperparameters
                    params = {}
                    for param_name, param_values in config.param_space.items():
                        if isinstance(param_values[0], int):
                            params[param_name] = trial.suggest_int(
                                f'{model_name}_{param_name}', 
                                min(param_values), max(param_values)
                            )
                        elif isinstance(param_values[0], float):
                            params[param_name] = trial.suggest_float(
                                f'{model_name}_{param_name}', 
                                min(param_values), max(param_values)
                            )
                        else:
                            params[param_name] = trial.suggest_categorical(
                                f'{model_name}_{param_name}', param_values
                            )
                    
                    # Create model
                    if model_name == 'catboost':
                        params['verbose'] = False
                    elif model_name == 'lightgbm':
                        params['verbose'] = -1
                    
                    model = config.model_class(**params, random_state=42)
                    active_models[model_name] = model
            
            if not active_models:
                return 0.0
            
            # Create ensemble
            ensemble = VotingRegressor(list(active_models.items()))
            
            # Cross-validation
            cv_scores = cross_val_score(ensemble, X, y, cv=cv_folds, scoring='r2', n_jobs=self.n_jobs)
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.optimization_budget)
        
        # Get best configuration
        best_params = study.best_params
        
        # Build best ensemble
        best_models = {}
        for model_name, config in self.model_configs.items():
            if best_params.get(f'{model_name}_active', False):
                params = {}
                for param_name in config.param_space.keys():
                    param_key = f'{model_name}_{param_name}'
                    if param_key in best_params:
                        params[param_name] = best_params[param_key]
                
                if model_name == 'catboost':
                    params['verbose'] = False
                elif model_name == 'lightgbm':
                    params['verbose'] = -1
                
                model = config.model_class(**params, random_state=42)
                best_models[model_name] = model
        
        # Train and evaluate best ensemble
        best_ensemble = VotingRegressor(list(best_models.items()))
        cv_scores = cross_val_score(best_ensemble, X, y, cv=cv_folds, scoring='r2', n_jobs=self.n_jobs)
        
        # Calculate individual model scores
        individual_scores = {}
        for name, model in best_models.items():
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2', n_jobs=self.n_jobs)
            individual_scores[name] = scores.mean()
        
        # Calculate ensemble weights (equal for VotingRegressor)
        ensemble_weights = {name: 1.0/len(best_models) for name in best_models.keys()}
        
        # Store trained models
        self.trained_models = best_models
        self.ensemble_weights = ensemble_weights
        
        return EnsemblePerformance(
            r2_score=cv_scores.mean(),
            rmse=np.sqrt(-cv_scores.mean()),  # Approximation
            mae=0.0,  # Will be calculated separately
            cv_scores=cv_scores.tolist(),
            individual_scores=individual_scores,
            ensemble_weights=ensemble_weights,
            training_time=0.0,
            prediction_time=0.0,
            model_count=len(best_models),
            improvement_over_best_individual=cv_scores.mean() - max(individual_scores.values())
        )
    
    async def _train_final_ensemble(self, X: np.ndarray, y: pd.Series):
        """Train final ensemble with optimized parameters"""
        
        # Train individual models
        for name, model in self.trained_models.items():
            model.fit(X, y)
        
        # Train meta-learner (stacking)
        meta_features = np.column_stack([
            model.predict(X) for model in self.trained_models.values()
        ])
        
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(meta_features, y)
        
        logger.info(f"Final ensemble trained with {len(self.trained_models)} models")
    
    async def predict_with_ensemble(self, X: pd.DataFrame, 
                                  method: str = 'stacking') -> np.ndarray:
        """Make predictions using optimized ensemble"""
        
        try:
            if not self.trained_models:
                raise ValueError("Ensemble not trained")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            if method == 'voting':
                # Weighted voting
                predictions = np.zeros(len(X))
                total_weight = sum(self.ensemble_weights.values())
                
                for name, model in self.trained_models.items():
                    weight = self.ensemble_weights[name] / total_weight
                    predictions += weight * model.predict(X_scaled)
                
                return predictions
            
            elif method == 'stacking':
                # Stacking with meta-learner
                if self.meta_learner is None:
                    raise ValueError("Meta-learner not trained")
                
                meta_features = np.column_stack([
                    model.predict(X_scaled) for model in self.trained_models.values()
                ])
                
                return self.meta_learner.predict(meta_features)
            
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
                
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            raise
    
    async def update_ensemble_weights(self, X: pd.DataFrame, y: pd.Series, 
                                    method: str = 'performance_based'):
        """Update ensemble weights based on recent performance"""
        
        try:
            if method == 'performance_based':
                # Calculate individual model performance
                X_scaled = self.scaler.transform(X)
                new_weights = {}
                
                for name, model in self.trained_models.items():
                    predictions = model.predict(X_scaled)
                    r2 = r2_score(y, predictions)
                    new_weights[name] = max(r2, 0.1)  # Minimum weight of 0.1
                
                # Normalize weights
                total_weight = sum(new_weights.values())
                self.ensemble_weights = {
                    name: weight / total_weight 
                    for name, weight in new_weights.items()
                }
                
            elif method == 'diversity_based':
                # Weight based on prediction diversity
                X_scaled = self.scaler.transform(X)
                predictions_matrix = np.column_stack([
                    model.predict(X_scaled) for model in self.trained_models.values()
                ])
                
                # Calculate diversity scores
                diversity_scores = {}
                for i, name in enumerate(self.trained_models.keys()):
                    other_predictions = np.delete(predictions_matrix, i, axis=1)
                    diversity = np.mean(np.std(other_predictions, axis=1))
                    diversity_scores[name] = diversity
                
                # Normalize diversity scores as weights
                total_diversity = sum(diversity_scores.values())
                self.ensemble_weights = {
                    name: score / total_diversity 
                    for name, score in diversity_scores.items()
                }
            
            logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            logger.error(f"Error updating ensemble weights: {e}")
    
    async def continuous_improvement(self, new_data: pd.DataFrame, 
                                   target_column: str = 'conversion_score'):
        """Continuously improve ensemble with new data"""
        
        try:
            # Check if retraining is needed
            if datetime.now() - self.last_retrain < self.retrain_frequency:
                return
            
            logger.info("Starting continuous improvement...")
            
            # Prepare new data
            X_new = new_data.drop(columns=[target_column])
            y_new = new_data[target_column]
            
            # Evaluate current performance
            current_predictions = await self.predict_with_ensemble(X_new)
            current_r2 = r2_score(y_new, current_predictions)
            
            # If performance is below threshold, retrain
            if current_r2 < self.performance_threshold:
                logger.info(f"Performance below threshold ({current_r2:.3f} < {self.performance_threshold})")
                
                # Retrain ensemble
                await self.optimize_ensemble(X_new, y_new)
                self.last_retrain = datetime.now()
                
                logger.info("Ensemble retrained with new data")
            else:
                # Just update weights
                await self.update_ensemble_weights(X_new, y_new)
                
        except Exception as e:
            logger.error(f"Error in continuous improvement: {e}")
    
    async def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from ensemble models"""
        
        importance_dict = {}
        
        for name, model in self.trained_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = {
                    f'feature_{i}': importance 
                    for i, importance in enumerate(model.feature_importances_)
                }
            elif hasattr(model, 'coef_'):
                importance_dict[name] = {
                    f'feature_{i}': abs(coef) 
                    for i, coef in enumerate(model.coef_)
                }
        
        return importance_dict
    
    async def save_ensemble(self, filepath: str):
        """Save trained ensemble to disk"""
        
        ensemble_data = {
            'trained_models': self.trained_models,
            'ensemble_weights': self.ensemble_weights,
            'meta_learner': self.meta_learner,
            'scaler': self.scaler,
            'performance_history': self.performance_history,
            'model_configs': self.model_configs
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    async def load_ensemble(self, filepath: str):
        """Load trained ensemble from disk"""
        
        ensemble_data = joblib.load(filepath)
        
        self.trained_models = ensemble_data['trained_models']
        self.ensemble_weights = ensemble_data['ensemble_weights']
        self.meta_learner = ensemble_data['meta_learner']
        self.scaler = ensemble_data['scaler']
        self.performance_history = ensemble_data['performance_history']
        self.model_configs = ensemble_data['model_configs']
        
        logger.info(f"Ensemble loaded from {filepath}")

# Global ensemble optimizer
ensemble_optimizer = AdvancedEnsembleOptimizer()
