# meta_classifier.py — Gradient Boosted Meta-Classifier for Model Selection
# Replaces the heuristic AutoAnalyzer with a learned classifier that
# selects the optimal forecasting model based on time series features.
#
# Training data comes from ablation_full_results.csv — each row maps
# (ticker features) → (best model config by RMSE).

import os
import numpy as np
import pandas as pd
import pickle
import yfinance as yf
from datetime import datetime, timedelta

# ============================================================================
# FEATURE EXTRACTOR — Same features used by AutoAnalyzer, formalized
# ============================================================================

class TimeSeriesFeatureExtractor:
    """
    Extracts statistical features from a price array for meta-learning.

    Features (10-dimensional vector):
        0. data_length         — number of price observations
        1. ann_volatility      — annualized std of returns
        2. trend_strength      — |OLS slope| / mean(price) over last 60d
        3. excess_kurtosis     — kurtosis of returns - 3
        4. skewness            — skewness of returns
        5. autocorrelation_1   — lag-1 autocorrelation
        6. vol_regime_ratio    — recent_vol / historical_vol
        7. max_drawdown        — maximum peak-to-trough decline (fraction)
        8. hurst_exponent      — estimated Hurst exponent (mean-reverting vs trending)
        9. return_mean         — annualized mean return
    """

    @staticmethod
    def extract(prices: np.ndarray) -> np.ndarray:
        """Extract feature vector from price array. Returns shape (10,)."""
        n = len(prices)
        returns = np.diff(prices) / prices[:-1]

        # 0. Data length (normalized)
        data_length = min(n / 1000.0, 1.0)

        # 1. Annualized volatility
        ann_vol = float(np.std(returns) * np.sqrt(252))

        # 2. Trend strength
        recent = prices[-min(60, n):]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        trend_strength = abs(slope) / np.mean(recent) if np.mean(recent) > 0 else 0.0

        # 3. Excess kurtosis
        kurt = float(pd.Series(returns).kurtosis())

        # 4. Skewness
        skew = float(pd.Series(returns).skew())

        # 5. Lag-1 autocorrelation
        if len(returns) > 2:
            ac1 = float(pd.Series(returns).autocorr(lag=1))
            if np.isnan(ac1):
                ac1 = 0.0
        else:
            ac1 = 0.0

        # 6. Vol regime ratio (recent 60d vs historical)
        if n > 120:
            recent_vol = np.std(returns[-60:])
            hist_vol = np.std(returns[:-60])
            vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        # 7. Max drawdown
        cummax = np.maximum.accumulate(prices)
        drawdowns = (prices - cummax) / cummax
        max_dd = float(np.min(drawdowns))

        # 8. Hurst exponent (R/S method, simplified)
        hurst = _estimate_hurst(returns)

        # 9. Annualized mean return
        ret_mean = float(np.mean(returns) * 252)

        return np.array([
            data_length, ann_vol, trend_strength, kurt, skew,
            ac1, vol_ratio, max_dd, hurst, ret_mean
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list:
        return [
            "data_length", "ann_volatility", "trend_strength",
            "excess_kurtosis", "skewness", "autocorrelation_1",
            "vol_regime_ratio", "max_drawdown", "hurst_exponent",
            "return_mean",
        ]


def _estimate_hurst(returns: np.ndarray, max_lag: int = 20) -> float:
    """
    Estimate the Hurst exponent using the R/S (Rescaled Range) method.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending
    """
    n = len(returns)
    if n < max_lag * 2:
        return 0.5  # Default to random walk for short series

    lags = range(2, max_lag + 1)
    rs_values = []

    for lag in lags:
        chunks = [returns[i:i + lag] for i in range(0, n - lag, lag)]
        if len(chunks) < 2:
            continue
        rs_chunk = []
        for chunk in chunks:
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1) if np.std(chunk, ddof=1) > 0 else 1e-10
            rs_chunk.append(r / s)
        rs_values.append((np.log(lag), np.log(np.mean(rs_chunk))))

    if len(rs_values) < 3:
        return 0.5

    x_log = [v[0] for v in rs_values]
    y_log = [v[1] for v in rs_values]
    hurst = np.polyfit(x_log, y_log, 1)[0]
    return float(np.clip(hurst, 0.0, 1.0))


# ============================================================================
# META-CLASSIFIER — Gradient Boosted Model Selector
# ============================================================================

class MetaClassifier:
    """
    Gradient Boosted classifier that maps time series features to the
    optimal forecasting model.

    Training: Learns from ablation results (historical performance data).
    Inference: Given a new ticker's features, predicts which model will
               perform best.

    This replaces the heuristic AutoAnalyzer scoring system with a
    data-driven approach that improves with each ablation run.
    """

    MODEL_SAVE_PATH = os.path.join(
        os.path.dirname(__file__), "meta_classifier_model.pkl"
    )

    # Canonical ordering of target classes
    TARGET_CLASSES = [
        "naive_baseline", "random_walk_baseline", "arima_baseline",
        "mc_only", "ets_only", "full_ensemble",
    ]

    def __init__(self):
        self.model = None
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.is_trained = False
        self._load_model()

    def _load_model(self):
        """Load a previously trained model from disk if available."""
        if os.path.exists(self.MODEL_SAVE_PATH):
            try:
                with open(self.MODEL_SAVE_PATH, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.is_trained = True
            except Exception:
                self.is_trained = False

    def train_from_ablation_csv(self, csv_path: str = None) -> dict:
        """
        Train the meta-classifier from ablation results CSV.

        The CSV should have columns: Config, Ticker, RMSE, MAPE (%), DA (%).
        For each ticker, the config with the lowest RMSE is the target label.

        Returns training metrics dict.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder

        if csv_path is None:
            csv_path = os.path.join(
                os.path.dirname(__file__), "results", "ablation_full_results.csv"
            )

        if not os.path.exists(csv_path):
            return {"success": False, "error": f"CSV not found: {csv_path}"}

        # Load ablation results
        df = pd.read_csv(csv_path)
        if df.empty:
            return {"success": False, "error": "Empty CSV"}

        # Find best model per ticker (lowest RMSE)
        df = df.dropna(subset=["RMSE"])
        best_per_ticker = df.loc[df.groupby("Ticker")["RMSE"].idxmin()]
        tickers = best_per_ticker["Ticker"].tolist()
        best_configs = best_per_ticker["Config"].tolist()

        if len(tickers) < 3:
            return {"success": False, "error": f"Need >=3 tickers, got {len(tickers)}"}

        # Extract features for each ticker
        X = []
        y = []
        failed = []

        for ticker, config in zip(tickers, best_configs):
            try:
                end = datetime.today()
                start = end - timedelta(days=756)  # ~3 years
                data = yf.download(ticker, start=start, end=end, progress=False)
                if data.empty:
                    failed.append(ticker)
                    continue
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                prices = data["Close"].dropna().values.astype(float)
                if len(prices) < 60:
                    failed.append(ticker)
                    continue
                features = self.feature_extractor.extract(prices)
                X.append(features)
                y.append(config)
            except Exception:
                failed.append(ticker)

        if len(X) < 3:
            return {"success": False, "error": f"Only {len(X)} valid tickers"}

        X = np.array(X)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Train Gradient Boosted Classifier
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_split=2,
            random_state=42,
        )

        # Cross-validation (use standard KFold for small datasets to avoid stratified small-class error)
        try:
            from sklearn.model_selection import KFold
            n_samples = len(X)
            n_splits = min(5, n_samples)
            if n_samples >= 2:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(clf, X, y_encoded, cv=cv)
                cv_accuracy = float(np.mean(cv_scores))
            else:
                cv_accuracy = 0.0
        except Exception:
            cv_accuracy = 0.0

        # Final training on all data
        clf.fit(X, y_encoded)

        # Feature importances
        importances = dict(zip(
            self.feature_extractor.feature_names(),
            [round(float(v), 4) for v in clf.feature_importances_]
        ))

        # Save model
        model_data = {
            "model": clf,
            "label_encoder": le,
            "feature_names": self.feature_extractor.feature_names(),
            "classes": le.classes_.tolist(),
            "train_date": datetime.now().isoformat(),
            "n_tickers": len(X),
            "cv_accuracy": cv_accuracy,
        }

        with open(self.MODEL_SAVE_PATH, "wb") as f:
            pickle.dump(model_data, f)

        self.model = clf
        self.is_trained = True

        return {
            "success": True,
            "n_tickers_trained": len(X),
            "n_tickers_failed": len(failed),
            "failed_tickers": failed,
            "cv_accuracy": round(cv_accuracy * 100, 1),
            "feature_importances": importances,
            "classes": le.classes_.tolist(),
            "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
        }

    def predict(self, ticker: str) -> dict:
        """
        Given a ticker, extract features and predict the best model.

        Returns:
            {
                'ticker': str,
                'predicted_model': str,
                'confidence': float (0-1),
                'all_probabilities': {model: prob, ...},
                'features': {feature_name: value, ...},
                'success': bool,
            }
        """
        if not self.is_trained or self.model is None:
            return {"success": False, "error": "Model not trained. Run train_from_ablation_csv() first."}

        # Load model data for label encoder
        try:
            with open(self.MODEL_SAVE_PATH, "rb") as f:
                model_data = pickle.load(f)
            le = model_data["label_encoder"]
        except Exception as e:
            return {"success": False, "error": f"Cannot load model: {e}"}

        # Download price data
        try:
            end = datetime.today()
            start = end - timedelta(days=756)
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                return {"success": False, "error": f"No data for {ticker}"}
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            prices = data["Close"].dropna().values.astype(float)
            if len(prices) < 60:
                return {"success": False, "error": f"Insufficient data ({len(prices)} days)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

        # Extract features
        features = self.feature_extractor.extract(prices)
        feature_dict = dict(zip(self.feature_extractor.feature_names(),
                                [round(float(v), 4) for v in features]))

        # Predict
        X = features.reshape(1, -1)
        pred_idx = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        predicted_label = le.inverse_transform([pred_idx])[0]
        confidence = float(np.max(probabilities))

        all_probs = {}
        for idx, cls in enumerate(le.classes_):
            if idx < len(probabilities):
                all_probs[cls] = round(float(probabilities[idx]), 4)

        return {
            "success": True,
            "ticker": ticker,
            "predicted_model": predicted_label,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "features": feature_dict,
        }

    def predict_batch(self, tickers: list) -> list:
        """Predict best model for multiple tickers."""
        return [self.predict(t) for t in tickers]


# ============================================================================
# CLI — Train and evaluate from command line
# ============================================================================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  META-CLASSIFIER: Training from Ablation Results")
    print("=" * 60)

    mc = MetaClassifier()

    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    result = mc.train_from_ablation_csv(csv_path)

    if result["success"]:
        print(f"\n  Trained on {result['n_tickers_trained']} tickers")
        print(f"  Failed:    {result['n_tickers_failed']} ({result['failed_tickers']})")
        print(f"  CV Accuracy: {result['cv_accuracy']}%")
        print(f"\n  Feature Importances:")
        for feat, imp in sorted(result["feature_importances"].items(),
                                key=lambda x: -x[1]):
            bar = "#" * int(imp * 50)
            print(f"    {feat:25s} {imp:.4f}  {bar}")
        print(f"\n  Classes: {result['classes']}")
        print(f"  Distribution: {result['class_distribution']}")

        # Test prediction
        print("\n  --- Test Predictions ---")
        for ticker in ["AAPL", "TSLA", "GLD"]:
            pred = mc.predict(ticker)
            if pred["success"]:
                print(f"    {ticker}: {pred['predicted_model']} "
                      f"(conf={pred['confidence']:.2f})")
    else:
        print(f"\n  ERROR: {result['error']}")
