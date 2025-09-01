#!/usr/bin/env python3
"""
Enhanced Automated ML Retraining Script
Periodically retrains the ML model with new data, validation, and monitoring
"""

import argparse
import os
import shutil
import json
import sys
from datetime import datetime, timedelta
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.ml_baseline import run_ml_baseline
from backtester.ml_monitor import monitor


def should_retrain(last_training_time: datetime, min_hours_between_training: int = 6) -> bool:
    """Check if enough time has passed since last training"""
    return datetime.utcnow() - last_training_time > timedelta(hours=min_hours_between_training)


def get_last_training_time(ml_dir: str) -> datetime:
    """Get the timestamp of the last training"""
    if not os.path.exists(ml_dir):
        return datetime.min

    subdirs = [d for d in os.listdir(ml_dir) if d.startswith('ml_')]
    if not subdirs:
        return datetime.min

    # Get the most recent training directory
    latest_dir = max(subdirs, key=lambda x: x.split('_')[1])
    timestamp_str = latest_dir.split('_')[1]

    try:
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except ValueError:
        return datetime.min


def backup_current_model(link_dir: str):
    """Backup the current model before retraining"""
    if not os.path.exists(link_dir):
        return

    backup_dir = os.path.join(os.path.dirname(link_dir), f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    try:
        shutil.copytree(link_dir, backup_dir)
        print(f"✅ Backed up current model to {backup_dir}")
    except Exception as e:
        print(f"⚠️  Failed to backup current model: {e}")


def validate_new_model(run_dir: str) -> bool:
    """Validate that the new model is working correctly"""
    model_path = os.path.join(run_dir, 'model.pt')
    features_path = os.path.join(run_dir, 'features.csv')
    metrics_path = os.path.join(run_dir, 'metrics.json')

    if not all(os.path.exists(p) for p in [model_path, features_path, metrics_path]):
        print("❌ New model missing required files")
        return False

    # Load and check metrics
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        accuracy = metrics.get('accuracy', 0)
        if accuracy < 0.5:  # Minimum acceptable accuracy
            print(f"❌ New model accuracy too low: {accuracy:.3f}")
            return False

        print(f"✅ New model validated - Accuracy: {accuracy:.3f}")
        if 'precision' in metrics:
            print(f"   Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
        return True

    except Exception as e:
        print(f"❌ Failed to validate new model: {e}")
        return False


def update_latest_link(run_dir: str, link_dir: str):
    """Update the latest link directory with new model files"""
    os.makedirs(link_dir, exist_ok=True)

    # Copy model files
    model_src = os.path.join(run_dir, "model.pt")
    feats_src = os.path.join(run_dir, "features.csv")
    metrics_src = os.path.join(run_dir, "metrics.json")

    try:
        shutil.copy2(model_src, os.path.join(link_dir, "model.pt"))
        if os.path.isfile(feats_src):
            shutil.copy2(feats_src, os.path.join(link_dir, "features.csv"))
        if os.path.isfile(metrics_src):
            shutil.copy2(metrics_src, os.path.join(link_dir, "metrics.json"))

        print(f"✅ Updated latest model files in {link_dir}")
    except Exception as e:
        print(f"❌ Failed to update latest link: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Enhanced Automated ML Retraining')
    parser.add_argument("--bars_dir", default="bars", help="Directory of input bars CSVs")
    parser.add_argument("--out_root", default="eval_runs/ml", help="Where to write timestamped ML runs")
    parser.add_argument(
        "--link_dir",
        default="eval_runs/ml/latest",
        help="Directory to place latest copies (model.pt, features.csv, metrics.json)",
    )
    parser.add_argument('--force', action='store_true', help='Force retraining even if recently trained')
    parser.add_argument('--backup', action='store_true', help='Backup current model before retraining')
    parser.add_argument('--min_hours', type=int, default=6, help='Minimum hours between retraining')

    args = parser.parse_args()

    print(f"🚀 Starting Enhanced ML Retraining at {datetime.utcnow()}")

    # Check if retraining is needed
    last_training = get_last_training_time(args.out_root)
    if not args.force and not should_retrain(last_training, args.min_hours):
        hours_since_training = (datetime.utcnow() - last_training).total_seconds() / 3600
        print(f"⏭️  Skipping retraining - last training was {hours_since_training:.1f} hours ago")
        return

    # Backup current model if requested
    if args.backup:
        backup_current_model(args.link_dir)

    try:
        # Run retraining
        print("🔄 Training enhanced ML model...")
        run_dir = run_ml_baseline(args.bars_dir, args.out_root)
        print(f"✅ Training completed: {run_dir}")

        # Validate new model
        if not validate_new_model(run_dir):
            print("❌ New model validation failed - keeping old model")
            return

        # Update latest link
        update_latest_link(run_dir, args.link_dir)

        # Log retraining event to monitor
        try:
            with open(os.path.join(run_dir, 'metrics.json'), 'r') as f:
                metrics = json.load(f)

            monitor.log_metrics({
                'retraining_completed': True,
                'new_model_dir': run_dir,
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': metrics
            })
        except Exception as e:
            print(f"⚠️  Failed to log metrics: {e}")

        print("🎉 Enhanced ML Retraining completed successfully!")
        print(f"📊 New model available at: {run_dir}")

    except Exception as e:
        print(f"❌ ML Retraining failed: {e}")
        monitor.log_metrics({
            'retraining_failed': True,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })
        raise


if __name__ == "__main__":
    main()
