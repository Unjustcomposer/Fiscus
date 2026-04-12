# math_formulations.py — Formal Mathematical Documentation
# Contains all equations used in the system as LaTeX-exportable docstrings.
# Serves dual purpose: code documentation + paper methodology source.

"""
=============================================================================
MATHEMATICAL FORMULATIONS FOR THE ENSEMBLE PORTFOLIO INTELLIGENCE SYSTEM
=============================================================================

This module documents every mathematical formula used across the system.
Each section corresponds to a component in the paper's methodology.

Notation:
    S_t     = asset price at time t
    r_t     = log return at time t = ln(S_t / S_{t-1})
    μ       = drift (annualized expected return)
    σ       = diffusion (annualized volatility)
    T       = forecast horizon
    Δt      = time step size
    Z       = standard normal random variable Z ~ N(0,1)
"""


# ============================================================================
# 1. GEOMETRIC BROWNIAN MOTION (Monte Carlo)
# ============================================================================

GBM_FORMULATION = r"""
\subsection{Geometric Brownian Motion (GBM)}

The continuous-time dynamics of the asset price $S_t$ under the
risk-neutral measure follow:

\begin{equation}
    dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
\end{equation}

where $W_t$ is a standard Wiener process. The discrete-time solution
used for Monte Carlo simulation is:

\begin{equation}
    S_{t+\Delta t} = S_t \exp\left[
        \left(\mu - \frac{\sigma^2}{2}\right) \Delta t
        + \sigma \sqrt{\Delta t} \cdot Z
    \right], \quad Z \sim \mathcal{N}(0, 1)
\end{equation}

Parameter estimation uses Maximum Likelihood on historical log returns:

\begin{align}
    \hat{\mu} &= \frac{252}{N} \sum_{i=1}^{N} r_i \\
    \hat{\sigma} &= \sqrt{252} \cdot \text{std}(r_1, \ldots, r_N)
\end{align}

where $r_i = \ln(S_i / S_{i-1})$ and 252 is the annualization factor
for trading days.
"""


# ============================================================================
# 2. LSTM NEURAL NETWORK
# ============================================================================

LSTM_FORMULATION = r"""
\subsection{Long Short-Term Memory (LSTM) Network}

The LSTM cell at time step $t$ computes:

\begin{align}
    f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
        & \text{(Forget gate)} \\
    i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
        & \text{(Input gate)} \\
    \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
        & \text{(Candidate cell)} \\
    C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
        & \text{(Cell state)} \\
    o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
        & \text{(Output gate)} \\
    h_t &= o_t \odot \tanh(C_t)
        & \text{(Hidden state)}
\end{align}

where $\sigma$ is the sigmoid function, $\odot$ denotes element-wise
multiplication, and $[h_{t-1}, x_t]$ is the concatenation of the previous
hidden state and current input.

\textbf{Architecture:}
\begin{itemize}
    \item LSTM Layer 1: $\text{input}=1 \to \text{hidden}=128$
    \item Dropout: $p=0.2$
    \item LSTM Layer 2: $\text{input}=128 \to \text{hidden}=64$
    \item Dropout: $p=0.2$
    \item Dense: $64 \to 32$ (ReLU)
    \item Output: $32 \to 1$
\end{itemize}

\textbf{Training:} Adam optimizer ($\eta=0.001$), MSE loss,
batch size 32, lookback window $L=60$ days.

\textbf{MC Dropout Uncertainty:}
\begin{equation}
    \hat{y}_t^{(k)} = f_\theta(x_t; \text{dropout on}), \quad k=1,\ldots,K
\end{equation}
\begin{align}
    \hat{y}_t &= \frac{1}{K}\sum_{k=1}^K \hat{y}_t^{(k)} \\
    \text{Var}(\hat{y}_t) &= \frac{1}{K}\sum_{k=1}^K
        (\hat{y}_t^{(k)} - \hat{y}_t)^2
\end{align}

95\% confidence interval: $\hat{y}_t \pm 1.96\sqrt{\text{Var}(\hat{y}_t)}$
"""


# ============================================================================
# 3. DOUBLE EXPONENTIAL SMOOTHING (Holt's Method)
# ============================================================================

HOLT_FORMULATION = r"""
\subsection{Double Exponential Smoothing (Holt's Method)}

Holt's method decomposes the time series into level ($\ell_t$) and
trend ($b_t$) components:

\begin{align}
    \ell_t &= \alpha y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1})
        & \text{(Level update)} \\
    b_t &= \beta(\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}
        & \text{(Trend update)}
\end{align}

The $h$-step-ahead forecast is:

\begin{equation}
    \hat{y}_{t+h|t} = \ell_t + h \cdot b_t
\end{equation}

Prediction intervals widen with horizon:

\begin{equation}
    \text{CI}_{95\%}(h) = \hat{y}_{t+h|t}
        \pm 1.96 \cdot \hat{\sigma}_\epsilon \cdot \sqrt{h}
\end{equation}

where $\hat{\sigma}_\epsilon$ is the residual standard deviation from
the fitted values.

Smoothing parameters: $\alpha=0.3$ (level), $\beta=0.1$ (trend).
"""


# ============================================================================
# 4. ENSEMBLE CONSENSUS WITH SENTIMENT GATING
# ============================================================================

ENSEMBLE_FORMULATION = r"""
\subsection{Ensemble Consensus with Sentiment Gating}

Given $M$ forecasting models, the ensemble consensus forecast is:

\begin{equation}
    \hat{y}_{\text{ensemble}} = \sum_{i=1}^{M} w_i \cdot \hat{y}_i
\end{equation}

\textbf{Base weights} are computed as inverse MAPE:

\begin{equation}
    w_i^{(\text{base})} =
        \frac{1/\text{MAPE}_i}{\sum_{j=1}^{M} 1/\text{MAPE}_j}
\end{equation}

\textbf{Sentiment gating} modulates weights using the FinBERT
aggregate sentiment score $s \in [-1, 1]$:

\begin{equation}
    w_i' = w_i^{(\text{base})} \cdot g(s, m_i)
\end{equation}

where the gating function $g$ is defined as:

\begin{equation}
    g(s, m_i) = \begin{cases}
        1 + 0.3s & \text{if } m_i = \text{LSTM} \\
        1 - 0.4s & \text{if } m_i = \text{Monte Carlo} \\
        1 + 0.5s & \text{if } m_i = \text{Exp. Smoothing}
    \end{cases}
\end{equation}

Rationale: Bearish sentiment ($s < 0$) increases Monte Carlo weight
(captures tail risk via stochastic simulation), while bullish sentiment
($s > 0$) increases trend-following model weights.

After gating, weights are re-normalized:
\begin{equation}
    \tilde{w}_i = \frac{w_i'}{\sum_{j=1}^{M} w_j'}
\end{equation}

\textbf{Agreement Score:}
\begin{equation}
    A = \max\left(0, \min\left(100,
        \left(1 - \min\left(
            \frac{\text{std}(\hat{y}_1, \ldots, \hat{y}_M)}
                 {|\text{mean}(\hat{y}_1, \ldots, \hat{y}_M)|}, 1
        \right)\right) \times 100
    \right)\right)
\end{equation}

\textbf{Signal Generation:}
\begin{equation}
    \text{Signal} = \begin{cases}
        \text{STRONG BUY}  & \text{if } R > 10\% \wedge A > 70 \wedge B \geq 2/3 \\
        \text{BUY}         & \text{if } R > 3\%  \wedge A > 50 \wedge B \geq 1/2 \\
        \text{STRONG SELL} & \text{if } R < -10\% \wedge A > 70 \wedge B \leq 1/3 \\
        \text{SELL}        & \text{if } R < -3\%  \wedge A > 50 \wedge B \leq 1/2 \\
        \text{HOLD}        & \text{otherwise}
    \end{cases}
\end{equation}

where $R = (\hat{y}_{\text{ensemble}} - S_T) / S_T$ is the expected
return and $B$ is the fraction of models predicting upward movement.
"""


# ============================================================================
# 5. AUTO-ANALYZER MODEL SELECTION
# ============================================================================

AUTO_ANALYZER_FORMULATION = r"""
\subsection{Data-Driven Model Selection (AutoAnalyzer)}

For each ticker, the AutoAnalyzer evaluates data characteristics and
computes a suitability score $\phi_i$ for each model $m_i$:

\begin{equation}
    \phi_i = \sum_{j} \delta_j(D) \cdot c_{ij}
\end{equation}

where $D$ is the data profile, $\delta_j(D)$ are binary feature
indicators, and $c_{ij}$ are predefined contribution scores.

\textbf{Data features:}
\begin{align}
    \text{Annualized Volatility:} & \quad
        \hat{\sigma}_a = \text{std}(r_1, \ldots, r_N) \cdot \sqrt{252} \\
    \text{Trend Strength:} & \quad
        \tau = \frac{|\hat{\beta}_1|}{\bar{S}_{60}},
        \text{ where } \hat{\beta}_1 = \text{OLS slope on last 60 days} \\
    \text{Excess Kurtosis:} & \quad
        \kappa = \frac{1}{N}\sum_{i=1}^N\left(\frac{r_i - \bar{r}}{\hat{\sigma}}\right)^4 - 3 \\
    \text{Regime Change:} & \quad
        \rho = \left|\frac{\sigma_{\text{recent}}}{\sigma_{\text{historical}}} - 1\right| > 0.5
\end{align}

The model with highest $\phi_i$ is selected. If the gap between the
top two scores is $< 1.5$, ensemble mode is triggered.
"""


# ============================================================================
# 6. COMPOSITE RISK SCORE
# ============================================================================

RISK_SCORE_FORMULATION = r"""
\subsection{Composite Risk Score}

The composite risk score $R \in [0, 100]$ aggregates five risk dimensions:

\begin{equation}
    R = \min\left(100, \max\left(0,
        R_\sigma + R_\beta + R_{\text{DD}} + R_{\text{CVaR}} + R_\eta
    \right)\right)
\end{equation}

where each component is normalized to a bounded contribution:

\begin{align}
    R_\sigma &= \min\left(\frac{\sigma_a}{0.60}, 1\right) \times 25
        & \text{(Volatility: max 25 pts)} \\
    R_\beta &= \min\left(\frac{|\beta|}{2.5}, 1\right) \times 20
        & \text{(Beta: max 20 pts)} \\
    R_{\text{DD}} &= \min\left(\frac{|\text{MaxDD}|}{0.50}, 1\right) \times 25
        & \text{(Drawdown: max 25 pts)} \\
    R_{\text{CVaR}} &= \min\left(\frac{|\text{CVaR}_{95}|}{0.05}, 1\right) \times 20
        & \text{(Tail risk: max 20 pts)} \\
    R_\eta &= \max\left(0, 1 - \frac{\max(\text{SR}, 0)}{2}\right) \times 10
        & \text{(Efficiency penalty: max 10 pts)}
\end{align}
"""


# ============================================================================
# 7. AUTOENCODER ANOMALY DETECTION
# ============================================================================

ANOMALY_FORMULATION = r"""
\subsection{Autoencoder Anomaly Detection}

The symmetric autoencoder maps a feature vector $\mathbf{x} \in \mathbb{R}^d$
through an encoder $f_\theta$ and decoder $g_\phi$:

\begin{align}
    \mathbf{z} &= f_\theta(\mathbf{x})
        & \text{(Latent representation, } \mathbf{z} \in \mathbb{R}^{16}\text{)} \\
    \hat{\mathbf{x}} &= g_\phi(\mathbf{z})
        & \text{(Reconstruction)}
\end{align}

Architecture: $d \to 64 \to 32 \to 16 \to 32 \to 64 \to d$

Training loss (MSE):
\begin{equation}
    \mathcal{L} = \frac{1}{N}\sum_{i=1}^N
        \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2
\end{equation}

\textbf{Feature vector} per rolling 30-day window:
\begin{equation}
    \mathbf{x} = [\bar{r}, \sigma_r, \text{skew}(r), \kappa(r),
        \text{MaxDD}, \text{VaR}_{95}, P(r>0), \rho_1, r_{\max}, r_{\min}]
\end{equation}

\textbf{Anomaly detection:} A holding is flagged as anomalous if its
reconstruction error exceeds a calibrated threshold $\tau^*$:

\begin{equation}
    e_i = \frac{1}{d}\|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2
\end{equation}

\textbf{Threshold calibration} (F1-optimal):
\begin{equation}
    \tau^* = \arg\max_{\tau \in [\text{P}_{80}, \text{P}_{99}]}
        F_1(\tau)
\end{equation}

where $F_1$ is computed using synthetic anomaly labels derived from
assets with extreme kurtosis ($\kappa > 5$) or drawdown
($\text{MaxDD} < -20\%$).
"""


# ============================================================================
# 8. EVALUATION METRICS
# ============================================================================

EVALUATION_FORMULATION = r"""
\subsection{Evaluation Metrics}

\textbf{Root Mean Squared Error:}
\begin{equation}
    \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}
\end{equation}

\textbf{Mean Absolute Percentage Error:}
\begin{equation}
    \text{MAPE} = \frac{100}{N}\sum_{i=1}^N
        \left|\frac{y_i - \hat{y}_i}{y_i}\right|
\end{equation}

\textbf{Directional Accuracy:}
\begin{equation}
    \text{DA} = \frac{100}{N-1}\sum_{i=2}^N
        \mathbb{1}\left[\text{sgn}(\Delta y_i) = \text{sgn}(\Delta \hat{y}_i)\right]
\end{equation}

\textbf{Hit Rate:}
\begin{equation}
    \text{HR} = \frac{\sum_{t: s_t \neq 0}
        \mathbb{1}[\text{sgn}(s_t) = \text{sgn}(r_t)]}
        {\sum_{t} \mathbb{1}[s_t \neq 0]} \times 100
\end{equation}

\textbf{Profit Factor:}
\begin{equation}
    \text{PF} = \frac{\sum_{t: s_t r_t > 0} s_t r_t}
        {|\sum_{t: s_t r_t < 0} s_t r_t|}
\end{equation}

\textbf{Diebold-Mariano Test:}
\begin{equation}
    d_t = e_{A,t}^2 - e_{B,t}^2, \quad
    \text{DM} = \frac{\bar{d}}{\hat{\text{se}}(\bar{d})}
        \xrightarrow{d} \mathcal{N}(0, 1)
\end{equation}
"""


# ============================================================================
# UTILITY — Export all formulations
# ============================================================================

def get_all_formulations() -> str:
    """Return all formulations concatenated for paper insertion."""
    sections = [
        GBM_FORMULATION,
        LSTM_FORMULATION,
        HOLT_FORMULATION,
        ENSEMBLE_FORMULATION,
        AUTO_ANALYZER_FORMULATION,
        RISK_SCORE_FORMULATION,
        ANOMALY_FORMULATION,
        EVALUATION_FORMULATION,
    ]
    return "\n\n".join(sections)


def export_to_file(filepath: str = "formulations.tex"):
    """Export all formulations to a .tex file."""
    content = r"""
\section{Mathematical Formulations}
\label{sec:math}

""" + get_all_formulations()

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath
