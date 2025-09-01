#!/usr/bin/env python3
"""
Advanced ML Retrainer with Automated Model Updates
Supports multiple architectures, ensemble methods, and intelligent retraining triggers
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtester.core import DataLoader
from backtester.features import build_features
from backtester.ml_baseline import (
    EnhancedMLP, AttentionMLP, LSTMModel, TransformerModel, CNNModel,
    EnsembleModel, AdvancedMLTrainer, trainer
)
from backtester.ml_monitor import monitor, get_ml_health_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ml_retrainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutomatedMLRetrainer:
    """Automated ML retrainer with intelligent scheduling and model selection"""

    def __init__(self, model_dir: str = "eval_runs/ml/", bars_dir: str = "bars/"):
        self.model_dir = Path(model_dir)
        self.bars_dir = Path(bars_dir)
        self.latest_model_path = self.model_dir / "latest"
        self.backup_dir = self.model_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Retraining parameters
        self.min_retrain_interval = timedelta(hours=6)  # Minimum time between retrains
        self.max_models_to_keep = 10  # Keep last N model versions
        self.performance_threshold = 0.55  # Minimum accuracy to keep model

        # Model architectures to try
        self.architectures = {
            'enhanced_mlp': self._train_enhanced_mlp,
            'attention_mlp': self._train_attention_mlp,
            'lstm': self._train_lstm,
            'transformer': self._train_transformer,
            'cnn': self._train_cnn,
            'ensemble': self._train_ensemble
        }

    def should_retrain(self) -> Tuple[bool, Dict]:
        """Determine if retraining is needed based on multiple criteria"""

        # Check time since last retrain
        last_retrain_time = self._get_last_retrain_time()
        if datetime.utcnow() - last_retrain_time < self.min_retrain_interval:
            return False, {'reason': 'too_soon', 'hours_remaining': (self.min_retrain_interval - (datetime.utcnow() - last_retrain_time)).total_seconds() / 3600}

        # Check ML health signals
        health_report = get_ml_health_report()
        retraining_signal = health_report.get('retraining_signal', {})

        if retraining_signal.get('should_retrain', False):
            return True, {
                'reason': 'health_signal',
                'confidence': retraining_signal.get('confidence', 0),
                'reasons': retraining_signal.get('reasons', [])
            }

        # Check performance degradation
        health_score = health_report.get('health_score', {})
        if health_score.get('health_score', 1.0) < 0.6:
            return True, {'reason': 'performance_degradation', 'health_score': health_score.get('health_score', 0)}

        # Check if we have enough new data
        new_data_count = self._count_new_data_points()
        if new_data_count > 100:  # Retrain if we have 100+ new data points
            return True, {'reason': 'new_data_available', 'new_points': new_data_count}

        return False, {'reason': 'no_retrain_needed'}

    def retrain_models(self, architectures: Optional[List[str]] = None) -> Dict:
        """Retrain models with specified architectures"""

        if architectures is None:
            architectures = list(self.architectures.keys())

        logger.info(f"Starting retraining for architectures: {architectures}")

        # Load and prepare data
        try:
            data_loader = DataLoader(str(self.bars_dir))
            bars = data_loader.load()
            X, y = build_features(bars)

            if len(X) < 200:  # Need more data for robust training
                return {'error': f'Insufficient data: {len(X)} samples, need at least 200'}

            logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {'error': f'Data loading failed: {str(e)}'}

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
        )

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

        results = {}
        best_model = None
        best_accuracy = 0

        # Train each architecture
        for arch_name in architectures:
            if arch_name not in self.architectures:
                logger.warning(f"Unknown architecture: {arch_name}")
                continue

            try:
                logger.info(f"Training {arch_name}...")
                result = self.architectures[arch_name](
                    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X.shape[1]
                )

                results[arch_name] = result

                if result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_model = {
                        'architecture': arch_name,
                        'model': result['model'],
                        'metrics': result
                    }

                logger.info(f"{arch_name} accuracy: {result['accuracy']:.4f}")

            except Exception as e:
                logger.error(f"Failed to train {arch_name}: {e}")
                results[arch_name] = {'error': str(e)}

        # Save best model
        if best_model and best_accuracy > self.performance_threshold:
            self._save_best_model(best_model, X.columns.tolist())
            results['best_model'] = {
                'architecture': best_model['architecture'],
                'accuracy': best_accuracy,
                'saved': True
            }
        else:
            results['best_model'] = {
                'architecture': 'none',
                'accuracy': best_accuracy,
                'saved': False,
                'reason': 'below_threshold' if best_accuracy <= self.performance_threshold else 'no_models'
            }

        # Clean up old models
        self._cleanup_old_models()

        logger.info(f"Retraining completed. Best model: {results.get('best_model', {})}")
        return results

    def _train_enhanced_mlp(self, X_train, y_train, X_val, y_val, input_dim):
        """Train Enhanced MLP model"""
        model = EnhancedMLP(input_dim)
        return self._train_model(model, X_train, y_train, X_val, y_val, "EnhancedMLP")

    def _train_attention_mlp(self, X_train, y_train, X_val, y_val, input_dim):
        """Train Attention MLP model"""
        model = AttentionMLP(input_dim)
        return self._train_model(model, X_train, y_train, X_val, y_val, "AttentionMLP")

    def _train_lstm(self, X_train, y_train, X_val, y_val, input_dim):
        """Train LSTM model"""
        model = LSTMModel(input_dim)
        return self._train_model(model, X_train, y_train, X_val, y_val, "LSTM")

    def _train_transformer(self, X_train, y_train, X_val, y_val, input_dim):
        """Train Transformer model"""
        model = TransformerModel(input_dim)
        return self._train_model(model, X_train, y_train, X_val, y_val, "Transformer")

    def _train_cnn(self, X_train, y_train, X_val, y_val, input_dim):
        """Train CNN model"""
        model = CNNModel(input_dim)
        return self._train_model(model, X_train, y_train, X_val, y_val, "CNN")

    def _train_ensemble(self, X_train, y_train, X_val, y_val, input_dim):
        """Train Ensemble model"""
        # Create individual models for ensemble
        models = [
            EnhancedMLP(input_dim),
            AttentionMLP(input_dim),
            LSTMModel(input_dim)
        ]

        ensemble = EnsembleModel(models)

        # Train ensemble
        optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        # Simple training
        ensemble.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = ensemble(X_train)
            loss = criterion(outputs.squeeze(), y_train.squeeze())
            loss.backward()
            optimizer.step()

        # Evaluate
        ensemble.eval()
        with torch.no_grad():
            val_outputs = ensemble(X_val)
            probs = torch.sigmoid(val_outputs).squeeze()
            preds = (probs >= 0.5).float()
            accuracy = (preds == y_val.squeeze()).float().mean().item()

        return {
            'model': ensemble,
            'accuracy': accuracy,
            'loss': loss.item(),
            'architecture': 'Ensemble'
        }

    def _train_model(self, model, X_train, y_train, X_val, y_val, arch_name):
        """Generic model training function"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_accuracy = 0
        best_model_state = None
        patience = 20
        patience_counter = 0

        model.train()
        for epoch in range(100):
            # Training
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train.squeeze())
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val.squeeze())
                val_probs = torch.sigmoid(val_outputs).squeeze()
                val_preds = (val_probs >= 0.5).float()
                accuracy = (val_preds == y_val.squeeze()).float().mean().item()
            model.train()

            # Early stopping
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            scheduler.step(val_loss)

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        return {
            'model': model,
            'accuracy': best_accuracy,
            'loss': loss.item(),
            'architecture': arch_name
        }

    def _save_best_model(self, best_model_info, feature_names):
        """Save the best performing model"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_dir = self.model_dir / f"retrained_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pt"
        torch.save(best_model_info['model'].state_dict(), model_path)

        # Save metadata
        metadata = {
            'architecture': best_model_info['architecture'],
            'accuracy': best_model_info['metrics']['accuracy'],
            'training_timestamp': timestamp,
            'features': feature_names,
            'input_dim': len(feature_names),
            'retraining_reason': getattr(self, '_last_retrain_reason', 'scheduled')
        }

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update latest symlink
        latest_link = self.latest_model_path
        if latest_link.exists():
            latest_link.unlink()

        # Create relative symlink
        try:
            latest_link.symlink_to(model_dir.name)
        except OSError:
            # Fallback for systems that don't support symlinks
            with open(latest_link / "model.pt", 'wb') as f:
                torch.save(best_model_info['model'].state_dict(), f)

        logger.info(f"Saved best model: {best_model_info['architecture']} with accuracy {best_model_info['metrics']['accuracy']:.4f}")

    def _get_last_retrain_time(self) -> datetime:
        """Get timestamp of last retraining"""
        if not self.latest_model_path.exists():
            return datetime.utcnow() - timedelta(days=1)  # Default to yesterday

        try:
            metadata_path = self.latest_model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return datetime.fromisoformat(metadata['training_timestamp'])
            else:
                # Fallback to file modification time
                return datetime.fromtimestamp(self.latest_model_path.stat().st_mtime)
        except Exception:
            return datetime.utcnow() - timedelta(days=1)

    def _count_new_data_points(self) -> int:
        """Count new data points since last retraining"""
        try:
            data_loader = DataLoader(str(self.bars_dir))
            bars = data_loader.load()

            last_retrain_time = self._get_last_retrain_time()
            new_data = bars[bars.index > last_retrain_time]

            return len(new_data)
        except Exception:
            return 0

    def _cleanup_old_models(self):
        """Clean up old model versions, keeping only the most recent ones"""
        try:
            model_dirs = [d for d in self.model_dir.iterdir() if d.is_dir() and d.name.startswith('retrained_')]
            model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep only the most recent models
            for old_dir in model_dirs[self.max_models_to_keep:]:
                import shutil
                shutil.rmtree(old_dir)
                logger.info(f"Cleaned up old model: {old_dir.name}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old models: {e}")

    def get_retraining_status(self) -> Dict:
        """Get current retraining status and recommendations"""
        should_retrain, reason = self.should_retrain()

        status = {
            'should_retrain': should_retrain,
            'reason': reason,
            'last_retrain': self._get_last_retrain_time().isoformat(),
            'new_data_points': self._count_new_data_points(),
            'available_architectures': list(self.architectures.keys())
        }

        if should_retrain:
            status['recommended_architectures'] = ['enhanced_mlp', 'attention_mlp', 'ensemble']

        return status


def main():
    parser = argparse.ArgumentParser(description='Automated ML Retrainer')
    parser.add_argument('--check-only', action='store_true', help='Only check if retraining is needed')
    parser.add_argument('--force', action='store_true', help='Force retraining regardless of conditions')
    parser.add_argument('--architectures', nargs='+', help='Specific architectures to train')
    parser.add_argument('--model-dir', default='eval_runs/ml/', help='Model directory')
    parser.add_argument('--bars-dir', default='bars/', help='Bars data directory')

    args = parser.parse_args()

    retrainer = AutomatedMLRetrainer(args.model_dir, args.bars_dir)

    if args.check_only:
        status = retrainer.get_retraining_status()
        print(json.dumps(status, indent=2))
        return

    # Check if retraining is needed
    should_retrain, reason = retrainer.should_retrain()

    if not should_retrain and not args.force:
        print(f"No retraining needed: {reason}")
        return

    if args.force:
        print("Forcing retraining...")
    else:
        print(f"Retraining triggered: {reason}")

    # Perform retraining
    architectures = args.architectures or ['enhanced_mlp', 'attention_mlp', 'ensemble']
    results = retrainer.retrain_models(architectures)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
