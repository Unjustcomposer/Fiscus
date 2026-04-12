# evaluation.py — Research-Grade Evaluation Metrics Suite
# Implements directional accuracy, hit rate, profit factor, information ratio,
# Calmar ratio, and statistical significance tests (paired t-test, Wilcoxon,
# Diebold-Mariano) for rigorous model comparison.

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# CORE FORECASTING METRICS
# ============================================================================

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Directional Accuracy (DA) — percentage of times the predicted direction
    of price change matches the actual direction.

    DA = (1/N) * Σ 𝟙(sign(Δy_t) == sign(Δŷ_t)) × 100

    Parameters:
        actual:    Array of actual prices (length T)
        predicted: Array of predicted prices (length T)

    Returns:
        DA as a percentage (0-100)
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if len(actual) < 2:
        return 0.0
    actual_changes = np.diff(actual)
    predicted_changes = np.diff(predicted)
    n = min(len(actual_changes), len(predicted_changes))
    if n == 0:
        return 0.0
    matches = np.sign(actual_changes[:n]) == np.sign(predicted_changes[:n])
    return float(np.mean(matches) * 100)


def hit_rate(signals: np.ndarray, returns: np.ndarray) -> float:
    """
    Hit Rate — percentage of signals (buy=+1, sell=-1, hold=0) where
    the signal direction matched the subsequent return direction.

    Parameters:
        signals: Array of trading signals (+1, -1, 0)
        returns: Array of subsequent actual returns

    Returns:
        Hit rate as a percentage (0-100)
    """
    signals, returns = np.asarray(signals), np.asarray(returns)
    active = signals != 0
    if active.sum() == 0:
        return 0.0
    correct = np.sign(signals[active]) == np.sign(returns[active])
    return float(np.mean(correct) * 100)


def profit_factor(signals: np.ndarray, returns: np.ndarray) -> float:
    """
    Profit Factor — ratio of gross profits to gross losses from signals.

    PF = Σ(profits from correct signals) / |Σ(losses from incorrect signals)|

    A PF > 1 means the strategy is profitable overall.

    Parameters:
        signals: Array of trading signals (+1, -1, 0)
        returns: Array of subsequent actual returns

    Returns:
        Profit factor (0 to inf). Returns inf if no losses.
    """
    signals, returns = np.asarray(signals, dtype=float), np.asarray(returns, dtype=float)
    active = signals != 0
    if active.sum() == 0:
        return 0.0
    strategy_returns = signals[active] * returns[active]
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)


# ============================================================================
# RISK-ADJUSTED PERFORMANCE METRICS
# ============================================================================

def information_ratio(portfolio_returns: np.ndarray,
                      benchmark_returns: np.ndarray) -> float:
    """
    Information Ratio — risk-adjusted excess return vs benchmark.

    IR = (R_p - R_b) / σ(R_p - R_b)

    Parameters:
        portfolio_returns: Array of portfolio period returns
        benchmark_returns: Array of benchmark period returns (same length)

    Returns:
        Information ratio (annualized if daily returns provided)
    """
    portfolio_returns = np.asarray(portfolio_returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    excess = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess)
    if tracking_error == 0:
        return 0.0
    return float(np.mean(excess) / tracking_error * np.sqrt(252))


def calmar_ratio(returns: np.ndarray) -> float:
    """
    Calmar Ratio — annualized return / maximum drawdown.

    Parameters:
        returns: Array of daily returns

    Returns:
        Calmar ratio
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = abs(float(drawdowns.min()))
    ann_return = float(np.mean(returns) * 252)
    if max_dd == 0:
        return float('inf') if ann_return > 0 else 0.0
    return float(ann_return / max_dd)


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
    """
    Annualized Sharpe Ratio from daily returns.

    SR = (μ_annual - r_f) / σ_annual
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns) * 252)
    sigma = float(np.std(returns) * np.sqrt(252))
    if sigma == 0:
        return 0.0
    return float((mu - risk_free_rate) / sigma)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from a return series (as a negative percentage)."""
    returns = np.asarray(returns, dtype=float)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(drawdowns.min() * 100)


# ============================================================================
# STATISTICAL SIGNIFICANCE TESTS
# ============================================================================

def paired_ttest(errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
    """
    Paired t-test to determine if two models have significantly different
    prediction errors.

    H0: mean(errors_a) == mean(errors_b)
    H1: mean(errors_a) != mean(errors_b)

    Parameters:
        errors_a: Absolute prediction errors from model A
        errors_b: Absolute prediction errors from model B

    Returns:
        {'t_stat': float, 'p_value': float, 'significant_5pct': bool}
    """
    errors_a, errors_b = np.asarray(errors_a), np.asarray(errors_b)
    n = min(len(errors_a), len(errors_b))
    if n < 3:
        return {'t_stat': 0.0, 'p_value': 1.0, 'significant_5pct': False}
    t_stat, p_value = stats.ttest_rel(errors_a[:n], errors_b[:n])
    return {
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant_5pct': p_value < 0.05,
    }


def wilcoxon_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank test — non-parametric alternative to paired t-test.
    More robust when error distributions are non-normal.

    H0: median(errors_a - errors_b) == 0

    Returns:
        {'w_stat': float, 'p_value': float, 'significant_5pct': bool}
    """
    errors_a, errors_b = np.asarray(errors_a), np.asarray(errors_b)
    n = min(len(errors_a), len(errors_b))
    diff = errors_a[:n] - errors_b[:n]
    diff = diff[diff != 0]  # Wilcoxon requires non-zero differences
    if len(diff) < 10:
        return {'w_stat': 0.0, 'p_value': 1.0, 'significant_5pct': False}
    try:
        w_stat, p_value = stats.wilcoxon(diff)
        return {
            'w_stat': float(w_stat),
            'p_value': float(p_value),
            'significant_5pct': p_value < 0.05,
        }
    except Exception:
        return {'w_stat': 0.0, 'p_value': 1.0, 'significant_5pct': False}


def diebold_mariano_test(errors_a: np.ndarray, errors_b: np.ndarray,
                          horizon: int = 1) -> dict:
    """
    Diebold-Mariano test for comparing predictive accuracy of two forecasts.

    Uses squared error loss differential: d_t = e_a_t^2 - e_b_t^2
    DM = mean(d) / se(d) ~ N(0,1) under H0

    H0: Both models have equal predictive accuracy
    H1: Model accuracies differ

    Parameters:
        errors_a: Prediction errors from model A (y - ŷ_a)
        errors_b: Prediction errors from model B (y - ŷ_b)
        horizon: Forecast horizon (for Newey-West bandwidth)

    Returns:
        {'dm_stat': float, 'p_value': float, 'significant_5pct': bool,
         'better_model': 'A' or 'B' or 'tie'}
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    n = min(len(errors_a), len(errors_b))
    if n < 10:
        return {'dm_stat': 0.0, 'p_value': 1.0, 'significant_5pct': False,
                'better_model': 'tie'}

    # Loss differential (squared error)
    d = errors_a[:n] ** 2 - errors_b[:n] ** 2
    d_mean = np.mean(d)

    # Newey-West standard error (autocorrelation-robust)
    gamma_0 = np.var(d)
    gamma_sum = 0.0
    for k in range(1, horizon):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d) > k else 0.0
        gamma_sum += gamma_k
    var_d = (gamma_0 + 2 * gamma_sum) / n

    if var_d <= 0:
        return {'dm_stat': 0.0, 'p_value': 1.0, 'significant_5pct': False,
                'better_model': 'tie'}

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    better = 'tie'
    if p_value < 0.05:
        better = 'A' if d_mean < 0 else 'B'

    return {
        'dm_stat': float(dm_stat),
        'p_value': float(p_value),
        'significant_5pct': p_value < 0.05,
        'better_model': better,
    }


# ============================================================================
# EVALUATION REPORT AGGREGATOR
# ============================================================================

class EvaluationReport:
    """
    Aggregates all evaluation metrics for multiple models into a structured
    report suitable for paper tables.
    """

    def __init__(self):
        self.results = []

    def add_model(self, model_name: str, actual: np.ndarray,
                  predicted: np.ndarray, signals: np.ndarray = None,
                  returns: np.ndarray = None,
                  benchmark_returns: np.ndarray = None):
        """
        Add a model's predictions and compute all metrics.

        Parameters:
            model_name: Name of the model
            actual: Array of actual prices
            predicted: Array of predicted prices
            signals: Optional array of trading signals (+1, -1, 0)
            returns: Optional array of actual returns (for signal evaluation)
            benchmark_returns: Optional benchmark returns (for IR)
        """
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        entry = {
            'Model': model_name,
            'RMSE': rmse(actual, predicted),
            'MAPE (%)': mape(actual, predicted),
            'MAE': mae(actual, predicted),
            'DA (%)': directional_accuracy(actual, predicted),
        }

        if signals is not None and returns is not None:
            signals = np.asarray(signals)
            returns = np.asarray(returns)
            entry['Hit Rate (%)'] = hit_rate(signals, returns)
            entry['Profit Factor'] = profit_factor(signals, returns)
        else:
            entry['Hit Rate (%)'] = None
            entry['Profit Factor'] = None

        if returns is not None:
            returns = np.asarray(returns, dtype=float)
            entry['Sharpe Ratio'] = sharpe_ratio(returns)
            entry['Calmar Ratio'] = calmar_ratio(returns)
            entry['Max Drawdown (%)'] = max_drawdown(returns)
        else:
            entry['Sharpe Ratio'] = None
            entry['Calmar Ratio'] = None
            entry['Max Drawdown (%)'] = None

        if returns is not None and benchmark_returns is not None:
            entry['Information Ratio'] = information_ratio(returns, benchmark_returns)
        else:
            entry['Information Ratio'] = None

        self.results.append(entry)

    def get_dataframe(self) -> pd.DataFrame:
        """Return results as a pandas DataFrame."""
        df = pd.DataFrame(self.results)
        # Round numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        return df

    def get_summary(self) -> dict:
        """Return summary statistics."""
        df = self.get_dataframe()
        if df.empty:
            return {}
        best_rmse = df.loc[df['RMSE'].idxmin(), 'Model']
        best_da = df.loc[df['DA (%)'].idxmax(), 'Model']
        best_mape = df.loc[df['MAPE (%)'].idxmin(), 'Model']
        return {
            'models_evaluated': len(df),
            'best_rmse': best_rmse,
            'best_da': best_da,
            'best_mape': best_mape,
        }

    def compare_models(self, model_a: str, model_b: str,
                       actual: np.ndarray,
                       predicted_a: np.ndarray,
                       predicted_b: np.ndarray) -> dict:
        """
        Run all statistical significance tests between two models.

        Returns dict with t-test, Wilcoxon, and Diebold-Mariano results.
        """
        actual = np.asarray(actual, dtype=float)
        errors_a = np.abs(actual - np.asarray(predicted_a, dtype=float))
        errors_b = np.abs(actual - np.asarray(predicted_b, dtype=float))
        raw_errors_a = actual - np.asarray(predicted_a, dtype=float)
        raw_errors_b = actual - np.asarray(predicted_b, dtype=float)

        return {
            'models': f'{model_a} vs {model_b}',
            'paired_ttest': paired_ttest(errors_a, errors_b),
            'wilcoxon': wilcoxon_test(errors_a, errors_b),
            'diebold_mariano': diebold_mariano_test(raw_errors_a, raw_errors_b),
        }


# ============================================================================
# EXPORT FORMATTERS
# ============================================================================

def format_latex_table(report: EvaluationReport, caption: str = "Model Comparison Results") -> str:
    """
    Export evaluation report as a LaTeX table for paper insertion.
    """
    df = report.get_dataframe()
    cols = [c for c in df.columns if df[c].notna().any()]
    df_clean = df[cols].copy()

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\label{tab:model_comparison}",
        f"\\begin{{tabular}}{{{' '.join(['l'] + ['r'] * (len(cols) - 1))}}}",
        r"\toprule",
    ]

    # Header
    header = " & ".join([f"\\textbf{{{c}}}" for c in cols]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for _, row in df_clean.iterrows():
        values = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                values.append("--")
            elif isinstance(v, float):
                values.append(f"{v:.4f}")
            else:
                values.append(str(v))
        lines.append(" & ".join(values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def format_markdown_table(report: EvaluationReport) -> str:
    """Export evaluation report as a GitHub-flavored Markdown table."""
    df = report.get_dataframe()
    cols = [c for c in df.columns if df[c].notna().any()]
    df_clean = df[cols].copy()

    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df_clean.iterrows():
        values = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                values.append("--")
            elif isinstance(v, float):
                values.append(f"{v:.4f}")
            else:
                values.append(str(v))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)
