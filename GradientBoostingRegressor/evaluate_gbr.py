from __future__ import annotations

import argparse
from pathlib import Path

from joblib import load

from utils import safe_load_npy, regression_metrics, plot_predictions, plot_residuals, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".", help="X_test / y_test npy dosyalarının bulunduğu klasör")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="GBR modelinin ve çıktılarının bulunduğu / kaydedileceği klasör",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = ensure_dir(args.artifacts_dir)

    model_path = artifacts_dir / "gbr_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}. Önce train_gbr.py çalıştır.")

    # Nihai (baseline veya tuned) modeli yükle
    model = load(model_path)

    # Test verisini yükle
    X_test = safe_load_npy(data_dir / "X_test_processed.npy")
    y_test = safe_load_npy(data_dir / "y_test.npy").reshape(-1)

    # Test seti tahminleri ve metrikler (MAE, RMSE, R^2)
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    print("=== Final GBR model - Test metrics (evaluate_gbr) ===")
    print(metrics)

    # Test seti için ek görseller (eval run'ı)
    plot_predictions(
        y_test,
        y_pred,
        artifacts_dir / "pred_vs_actual_test_eval.png",
        title="Test: Predicted vs Actual (eval run)",
    )
    plot_residuals(
        y_test,
        y_pred,
        artifacts_dir / "residuals_test_eval",
        title_prefix="Test residuals (eval run)",
    )

    print(f"Saved eval plots to: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
