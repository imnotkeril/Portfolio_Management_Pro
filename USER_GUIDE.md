# Wild Market Capital - User Guide

**Complete guide to using the Portfolio Management Terminal.**

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard](#dashboard)
3. [Portfolio Management](#portfolio-management)
4. [Portfolio Analysis](#portfolio-analysis)
5. [Portfolio Optimization](#portfolio-optimization)
6. [Risk Analysis](#risk-analysis)
7. [Forecasting](#forecasting)
8. [Transactions](#transactions)
9. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First Launch

1. **Start the Application**:
   ```bash
   python run.py
   ```
   The application opens at `http://localhost:8501`

2. **Check System Status**:
   - Look at the sidebar "System Status" section
   - Ensure API status shows "Online" (green)
   - Response time should be <100ms
   - Click "Refresh Status" if needed

3. **Understand Navigation**:
   - **Sidebar**: Main navigation menu (always visible)
   - **Pages**: Dashboard, Create Portfolio, Portfolio List, Portfolio Analysis, Optimization, Risk Analysis, Forecasting

### Creating Your First Portfolio

**Recommended Steps**:

1. **Start with Dashboard**:
   - View market conditions
   - Check major indices (S&P 500, NASDAQ, Dow Jones, Russell 2000)
   - Use quick navigation buttons

2. **Create a Test Portfolio**:
   - Click "Create Portfolio" in sidebar or dashboard
   - Use "Text Input" method (fastest)
   - Enter: `AAPL 40%, MSFT 30%, GOOGL 30%`
   - Follow the 5-step wizard to completion

3. **View Your Portfolio**:
   - Go to "Portfolio List"
   - Click "View" on your portfolio
   - Explore Overview, Positions, Transactions, Strategies tabs

4. **Run Analysis**:
   - Navigate to "Portfolio Analysis"
   - Select your portfolio
   - Choose date range (default: 1 year)
   - Explore all 5 tabs and metrics

---

## Dashboard

**Location**: Sidebar → "Dashboard"

### Features

**Quick Navigation**:
- 6 navigation buttons: Portfolio List, Create Portfolio, Analysis, Optimization, Risk Analysis, Forecasting

**Market Indices**:
- **S&P 500 (^GSPC)**: Current price, daily change
- **NASDAQ (^NDX)**: Technology-focused index
- **Dow Jones (^DJI)**: 30 large-cap stocks
- **Russell 2000 (^RUT)**: Small-cap index
- Real-time updates with color coding (green/red)

**Indices Comparison Chart**:
- Always shows 1 year of data
- Cumulative returns for all 4 indices
- Interactive Plotly chart (zoom, pan, hover)

**Market Statistics**:
- Market volatility indicators
- Trading volume statistics
- Market breadth metrics

---

## Portfolio Management

### Creating Portfolios

**5 Methods Available**:

1. **Wizard** (Step-by-Step):
   - **Step 1**: Portfolio Information (name, description, currency, initial investment)
   - **Step 2**: Choose Input Method (Text, File, Manual, Template)
   - **Step 3**: Add Assets (varies by method)
   - **Step 4**: Settings & Review (cash management, portfolio mode)
   - **Step 5**: Create & Results
   - Best for beginners
   - Validation at each step

2. **Text Input** (Fastest):
   - Natural language entry
   - Supports multiple formats:
     - `AAPL:40%, MSFT:30%, GOOGL:30%` (colon with percentages)
     - `AAPL 0.4, MSFT 0.3, GOOGL 0.3` (space with decimals)
     - `AAPL 40, MSFT 30, GOOGL 30` (numbers > 1 auto-percentage)
     - `AAPL, MSFT, GOOGL` (equal weights)
   - Real-time parsing and validation
   - Automatic weight normalization

3. **CSV Import**:
   - Upload CSV or Excel files
   - Required columns: `ticker`, `weight` (or `shares`)
   - Automatic column mapping
   - Data preview and validation
   - Error reporting with downloadable error report

4. **Manual Entry**:
   - Add positions one by one
   - Real-time ticker validation
   - Current price fetching
   - Full control over allocation
   - Dynamic form with add/remove

5. **Templates**:
   - Pre-built investment strategies:
     - Value Factor, Quality Factor, Growth Factor
     - Low Volatility, Small Cap Factor, Dividend Factor
     - Profitability Factor, 60/40 Balanced, All Weather Portfolio
     - Tech Focus
   - Customizable allocations
   - Quick start option

### Managing Portfolios

**Portfolio List View**:
- **Search & Filter**: Search by name/description, sort by name/date/value
- **Portfolio Table**: Editable table with selection checkboxes
- **Bulk Operations**: Update prices, delete selected portfolios
- **Individual Actions**: Edit, View, Analyze, Delete buttons per portfolio

**Portfolio View** (4 Tabs):
- **Overview Tab**: Portfolio summary, key metrics, positions table, allocation charts
- **Positions Tab**: Full position management (edit, add, remove)
- **Transactions Tab**: Transaction history, add/edit/delete, import/export CSV
- **Strategies Tab**: Strategy management (for backtesting)

**Portfolio Editor**:
- Edit name, description, currency, starting capital
- Inline position editing (shares, purchase price, weight target)
- Add/remove positions
- Transaction management
- Strategy configuration

**Available Operations**:
- **View**: See portfolio details, positions, transactions, strategies
- **Edit**: Update name, description, positions, transactions
- **Clone**: Duplicate with modifications
- **Delete**: Remove portfolio (with confirmation)
- **Export**: CSV/Excel export (coming soon)

---

## Portfolio Analysis

**Location**: Sidebar → "Portfolio Analysis"

### Analysis Configuration

**Parameters**:
- **Portfolio Selection**: Dropdown with all portfolios
- **Date Range**: Start date and end date pickers
- **Benchmark**: Optional benchmark ticker (SPY, QQQ, VTI, etc.) or another portfolio
- **Calculate Button**: Triggers metric calculation

### Tab 1: Overview

**Key Metrics Dashboard**:
- **Performance Metrics**: Total Return, CAGR, Annualized Return, YTD, MTD, QTD, Best/Worst Periods, Win Rate
- **Risk Metrics**: Volatility (daily/weekly/monthly/annual), Max Drawdown, VaR (90%/95%/99%), CVaR
- **Risk-Adjusted Ratios**: Sharpe, Sortino, Calmar, Information Ratio
- **Market Metrics**: Beta, Alpha (CAPM), R², Correlation, Tracking Error

**Charts**:
- **Cumulative Returns**: Portfolio vs benchmark over time
- **Drawdown Chart**: Underwater plot showing all drawdown periods
- **Key Metrics Summary Table**: Side-by-side comparison with benchmark

### Tab 2: Performance

**Sub-tab 2.1: Returns Analysis**:
- Cumulative Returns chart (full period)
- Daily Active Returns area chart (portfolio - benchmark)
- Return by Periods table (1M, 3M, 6M, 1Y, 3Y, 5Y, Since Inception)
- Expected Returns table (daily, weekly, monthly, quarterly, yearly)
- Common Performance Periods (CPP) analysis
- Best/Worst 3-month periods

**Sub-tab 2.2: Periodic Analysis**:
- Monthly Returns Heatmap (calendar view, color-coded)
- Yearly Returns bar chart
- Quarterly Returns bar chart
- Period Comparison table (sortable, filterable)

**Sub-tab 2.3: Distribution**:
- Returns Distribution histogram with normal distribution overlay
- Q-Q Plot (Quantile-Quantile) for normality testing
- Statistical Tests (Shapiro-Wilk, skewness, kurtosis)
- Outlier Analysis scatter plot

### Tab 3: Risk

**Sub-tab 3.1: Key Risk Metrics**:
- Volatility metrics (daily, weekly, monthly, annual)
- Drawdown metrics (Max, Current, Average, Duration, Recovery)
- VaR summary (90%, 95%, 99% at different confidence levels)
- CVaR summary (Expected Shortfall)
- Complete Risk Metrics table (all 22 metrics)

**Sub-tab 3.2: Drawdown Analysis**:
- Underwater Plot (full drawdown visualization)
- Drawdown Periods table (all drawdowns with dates, depth, duration)
- Drawdown Recovery Analysis (recovery time statistics)
- Drawdown Statistics summary

**Sub-tab 3.3: VaR & CVaR**:
- VaR Distribution chart (histogram with VaR levels marked)
- VaR Methods Comparison table (Historical, Parametric, Cornish-Fisher, Monte Carlo)
- CVaR Analysis (expected shortfall visualization)
- Rolling VaR (VaR over time, time series chart)

**Sub-tab 3.4: Rolling Risk Metrics**:
- Rolling Volatility (30-day, 60-day, 90-day windows)
- Rolling Sharpe Ratio
- Rolling Sortino Ratio
- Rolling Beta
- All displayed as time series charts

### Tab 4: Assets & Correlations

**Sub-tab 4.1: Asset Overview & Impact**:
- Asset Metrics table (individual asset performance, contribution to return/risk)
- Asset Impact on Returns (bar chart)
- Asset Impact on Risk (bar chart)
- Risk vs Weight Comparison (scatter plot)

**Sub-tab 4.2: Correlation Analysis**:
- Correlation Matrix Heatmap (all assets vs all assets)
- Correlation with Benchmark (bar chart)
- Correlation Statistics (average, min/max, distribution)
- Clustered Correlation Matrix (hierarchical clustering with dendrogram)

**Sub-tab 4.3: Asset Details & Dynamics**:
- Asset Price Charts (individual price charts for all assets)
- Volume Analysis (trading volume charts)
- Price vs Volume scatter plots
- Detailed Asset Analysis (comprehensive asset view)

### Tab 5: Export & Reports

**CSV Export**:
- Portfolio data
- Metrics (all or selected)
- Returns data
- Positions data

**Excel Export**:
- Multi-sheet workbook
- Summary, Holdings, Metrics, Returns, History sheets
- Formatted tables

**PDF Report Generation**:
- Screenshot-based PDF (captures actual Streamlit pages)
- Selectable sections: Overview, Performance, Risk, Assets & Correlations
- Full page screenshots with charts
- High quality output

---

## Portfolio Optimization

**Location**: Sidebar → "Portfolio Optimization"

### Optimization Methods

**18 Methods Available**:

1. **Mean-Variance (Markowitz)**: Classic portfolio theory (Max Sharpe, Min Variance, Max Return objectives)
2. **Black-Litterman**: Bayesian approach with market views
3. **Risk Parity**: Equal risk contribution
4. **Hierarchical Risk Parity (HRP)**: Clustering-based, robust
5. **CVaR Optimization**: Tail risk focus (minimize CVaR)
6. **Mean-CVaR**: Return vs tail risk trade-off
7. **Robust Optimization**: Uncertainty sets, worst-case optimization
8. **Maximum Diversification**: Maximize diversification ratio
9. **Minimum Correlation**: Minimize average correlation
10. **Inverse Correlation Weighting**: Anti-correlation focus
11. **Equal Weight**: Simple equal allocation
12. **Min Variance**: Minimum volatility portfolio
13. **Max Sharpe**: Maximum Sharpe ratio
14. **Max Return**: Maximum expected return
15. **Kelly Criterion**: Optimal bet sizing
16. **Min Tracking Error**: Minimize deviation from benchmark
17. **Max Alpha**: Maximize alpha (excess return)
18. **Market Cap Weighting**: Market capitalization weights

### Configuration

**Parameters**:
- **Date Range**: Historical period for optimization (start date, end date)
- **Out-of-Sample Testing**: Validate on unseen data
  - **Training Window**: 30% (recommended), 50%, or 60% of analysis period
  - Shows training period dates and validation period dates
- **Benchmark**: Compare to SPY, QQQ, VTI, DIA, IWM, or custom ticker
- **Objective Function**: Varies by method (Max Sharpe, Min Variance, Max Return, Min CVaR)

**Constraints**:
- **Weight Limits**: Min/Max weight per asset (0% to 100%)
- **Portfolio Constraints**: Long-only (no shorting), sum to 100%
- **Group Constraints**: Sector/geographic limits (if sector data available)
- **Cardinality**: Min/Max number of positions
- **Turnover**: Maximum turnover limit (coming soon)

### Results Display

**Optimized Weights Table**:
- Ticker, Current Weight, Optimized Weight, Change, Change %, New Shares

**Trade List**:
- Required transactions (Buy/Sell for each asset)
- Shares to buy/sell
- Estimated cost
- Transaction type

**Before/After Comparison**:
- Metrics comparison: Expected Return, Volatility, Sharpe Ratio, Max Drawdown, Beta
- Visual comparison with improvement indicators
- Color-coded differences

**Efficient Frontier**:
- Risk-return curve visualization
- Current portfolio marked
- Optimized portfolio marked
- Interactive: click points to see weights

**Sensitivity Analysis**:
- Parameter variation testing
- Weight stability assessment
- Robustness recommendations

---

## Risk Analysis

**Location**: Sidebar → "Risk Analysis"

### Tab 1: VaR Analysis

**Configuration**:
- **Confidence Level**: Slider from 90% to 99% (default: 95%)
- **Time Horizon**: 1-30 days forward (default: 1 day)

**Methods**:
1. **Historical VaR**: Based on historical returns (no distribution assumptions)
2. **Parametric VaR**: Assumes normal distribution (uses mean and volatility)
3. **Cornish-Fisher VaR**: Adjusts for skewness and kurtosis (more accurate for non-normal)
4. **Monte Carlo VaR**: Simulation-based (10,000+ simulations)

**Results**:
- VaR and CVaR at different confidence levels (90%, 95%, 99%)
- Methods Comparison table (all methods side-by-side)
- VaR Distribution chart (histogram with VaR levels marked)
- Rolling VaR (VaR over time, time series chart)
- VaR Sensitivity Analysis (parameter variation)
- Portfolio VaR vs Component VaR (risk decomposition by asset)

### Tab 2: Monte Carlo Simulation

**Configuration**:
- **Number of Paths**: 1,000 - 100,000 (default: 10,000)
- **Time Horizon**: 1-365 days (default: 30 days)
- **Simulation Method**: Geometric Brownian Motion (GBM)

**Results**:
- **Distribution of Outcomes**: Histogram of final values
- **Probability Metrics**: Probability of loss, gain, >X% loss, >X% gain
- **Confidence Intervals**: 90%, 95%, 99% confidence intervals
- **Path Visualization**: Sample paths shown (representative scenarios)
- **Statistics**: Mean, median, std dev, min, max, percentiles (5th, 25th, 50th, 75th, 95th)
- **Extreme Scenarios**: Worst-case and best-case scenarios

### Tab 3: Historical Scenarios

**Available Scenarios**:
- **2008 Financial Crisis**: Sep 2008 - Mar 2009
- **2020 COVID-19 Crash**: Feb 2020 - Apr 2020
- **2000 Dot-com Bubble**: Mar 2000 - Oct 2002
- **1987 Black Monday**: Oct 1987
- **Custom Historical Period**: Select any date range

**Results for Each Scenario**:
- Portfolio Impact (total loss/gain, percentage change, absolute value change)
- Position-Level Impacts (each asset's contribution, worst/best positions)
- Comparison Across Scenarios (all scenarios in one table)
- Recovery Analysis (time to recover, recovery path)

### Tab 4: Custom Scenario

**Scenario Creation**:
- **Scenario Name**: Descriptive name (e.g., "Tech Crash", "Inflation Spike")
- **Asset Shocks**: Define % change for each asset (positive or negative)
- **Correlation Changes**: Modify correlations (advanced, optional)

**Results**:
- Portfolio Impact (total impact percentage and value)
- Position Breakdown (each position's impact)
- Interpretation (automatic analysis, risk assessment, recommendations)

### Tab 5: Scenario Chain

**Chain Configuration**:
- Define sequence of scenarios (e.g., Crisis → Recovery → Growth)
- Cumulative impact analysis
- Recovery paths

**Results**:
- Cumulative Impact (total impact across all scenarios)
- Step-by-step breakdown
- Path Visualization (portfolio value over time with scenario transitions)
- Recovery Analysis

---

## Forecasting

**Location**: Sidebar → "Forecasting"

### Asset/Portfolio Selection

**Single Asset Forecasting**:
- Dropdown of all tickers in portfolios
- Or enter custom ticker
- Real-time validation

**Portfolio Forecasting**:
- Choose from portfolio list
- Forecasts aggregate portfolio (weighted)

### Forecasting Parameters

**Training Period**:
- **Start Date**: Historical data start (default: 1 year ago)
- **End Date**: Training data end (default: Today)
- Recommendations: 1-2 years for short-term, 3-5 years for long-term

**Out-of-Sample Testing**:
- **Training Window**: 30% (recommended), 50%, or 60% of analysis period
- Validates forecast accuracy on unseen data
- Shows training period dates and validation period dates

**Forecast Horizon**:
- Options: 1 Day, 1 Week (5 days), 2 Weeks (10 days), 1 Month (21 days), 3 Months (63 days), 6 Months (126 days), 1 Year (252 days)
- Custom: Any number of days

### Forecasting Methods

**Classical** (3 methods):
- **ARIMA**: AutoRegressive Integrated Moving Average (trend forecasting)
- **GARCH**: Generalized Autoregressive Conditional Heteroskedasticity (volatility forecasting)
- **ARIMA-GARCH**: Combined mean and volatility model

**Machine Learning** (3 methods):
- **Random Forest**: Ensemble of decision trees
- **SVM/SVR**: Support Vector Regression
- **XGBoost**: Extreme Gradient Boosting

**Deep Learning** (3 methods):
- **LSTM**: Long Short-Term Memory (recurrent neural network)
- **TCN**: Temporal Convolutional Network
- **SSA-MAEMD-TCN**: Hybrid decomposition + deep learning (most sophisticated)

**Simple** (1 method):
- **Prophet**: Facebook's forecasting tool (handles seasonality well)

**Ensemble**:
- **Weighted Average**: Combines multiple forecasts
- **Model Selection**: Automatically selects best model

### Forecast Results

**4 Results Tabs**:

1. **Forecasts Comparison**:
   - Side-by-side comparison of all selected methods
   - Historical data + validation period + forecast period
   - Confidence intervals
   - Best model selection

2. **Individual Forecasts**:
   - Detailed view for each method
   - Model-specific charts and information
   - Parameter details

3. **Forecast Quality**:
   - Validation metrics (MAE, RMSE, MAPE, Directional Accuracy)
   - Residuals analysis
   - Model performance comparison

4. **Detailed Analysis**:
   - Model information (parameters, training time, complexity)
   - Forecast statistics (point forecast, confidence intervals, percentiles)
   - Technical analysis

---

## Transactions

**Location**: Portfolio List → View Portfolio → Transactions Tab

### Adding Transactions

**Transaction Types**:
- **BUY**: Purchase shares
- **SELL**: Sell shares
- **DEPOSIT**: Add cash to portfolio
- **WITHDRAWAL**: Remove cash from portfolio

**Transaction Form**:
- **Date**: Transaction date (date picker, max: today)
- **Type**: BUY, SELL, DEPOSIT, WITHDRAWAL (dropdown)
- **Ticker**: Ticker symbol (auto-set to CASH for DEPOSIT/WITHDRAWAL)
- **Shares**: Number of shares (or amount for CASH)
- **Price**: Price per share (auto-set to $1.00 for CASH)
- **Fees**: Transaction fees (optional, default: $0.00)
- **Notes**: Optional notes (max 500 characters)

**Features**:
- Real-time validation
- Total amount calculation (shares × price)
- Ticker validation for BUY/SELL
- Automatic CASH handling for DEPOSIT/WITHDRAWAL

### Transaction Management

**Transaction History Table**:
- Date, Type, Ticker, Shares, Price, Amount, Fees, Notes
- Sortable columns
- Color-coded transaction types
- Edit/Delete actions

**Available Operations**:
- **View**: See all transactions in chronological order
- **Add**: Use "Add Transaction" expander form
- **Edit**: Edit existing transactions (coming soon)
- **Delete**: Delete transactions with confirmation
- **Import**: Import from CSV/Excel file
- **Export**: Export to CSV file

**Transaction Summary**:
- Total transactions count
- First transaction date
- Last transaction date
- Total invested (sum of BUY + DEPOSIT amounts)

### Portfolio Modes

**Buy-and-Hold** (Default):
- Fixed positions (shares don't change)
- No transaction tracking
- Simple mode for static portfolios

**With Transactions**:
- Track all operations (BUY, SELL, DEPOSIT, WITHDRAWAL)
- Calculate positions from transaction history
- Real portfolio history
- Positions update automatically based on transactions

**Mode Indicator**:
- Shown in portfolio view
- Green badge: "With Transactions"
- Blue badge: "Buy-and-Hold"

---

## Troubleshooting

### Common Issues

**"Portfolio not found"**:
- Check "Portfolio List" to verify portfolio exists
- Refresh the page
- Restart application if persistent
- Check database connection

**"Ticker not found" or "Invalid ticker symbol"**:
- Verify ticker symbol spelling (e.g., AAPL, not APPL)
- Check if ticker is still active (not delisted)
- Try alternative format (e.g., BRK.B vs BRK-B)
- Check system status in sidebar (API should be online)
- Wait a few minutes and retry (API might be temporarily unavailable)

**"Insufficient data" or "Not enough historical data"**:
- Reduce date range (try 6 months instead of 1 year)
- Check ticker listing date (newly listed stocks have limited history)
- Verify data availability in price charts
- Use different ticker if data insufficient

**"Optimization failed" or "No solution found"**:
- Relax constraints (increase max weight, reduce min weight)
- Check if constraints are feasible (e.g., min weights sum to >100%)
- Increase date range for more data
- Try different optimization method
- Remove cardinality constraints if too restrictive

**"Forecast error" or "Model training failed"**:
- Increase training period (more historical data)
- Check data quality (no missing values, no extreme outliers)
- Try simpler model (ARIMA instead of LSTM)
- Reduce forecast horizon
- Enable out-of-sample testing to validate

**"Slow performance" or "Calculation taking too long"**:
- Reduce date range
- Use simpler optimization method
- Run forecasts one at a time (not simultaneously)
- Check API status (might be slow)
- Clear cache and retry
- Restart application if needed

**"Chart not displaying" or "Empty chart"**:
- Check date range (ensure data exists for selected period)
- Verify portfolio has positions
- Check if benchmark data available
- Try different browser
- Refresh page
- Clear browser cache

**"Export failed" or "PDF generation error"**:
- Ensure Streamlit app is running
- Check browser automation tools installed (Playwright)
- Verify write permissions
- Try CSV export instead of PDF
- Check available disk space

### Performance Tips

**For Large Portfolios** (20+ assets):
- Use shorter date ranges for faster calculations
- Prefer simpler optimization methods (Equal Weight, Min Variance)
- Disable out-of-sample testing for faster results
- Use cached data when possible

**For Long Date Ranges** (5+ years):
- Calculate metrics in batches
- Use sampling for Monte Carlo (fewer paths, e.g., 1,000 instead of 10,000)
- Simplify forecast models
- Export data for external analysis if needed

**For Multiple Forecasts**:
- Run forecasts sequentially, not simultaneously
- Use faster methods (ARIMA, Prophet) for quick results
- Save results before running next forecast
- Use ensemble only for final decision

---

## Best Practices

### Portfolio Creation
- **Start Simple**: Begin with 5-10 positions, expand gradually
- **Diversify**: Max 20-30% per asset to reduce concentration risk
- **Use Templates**: Leverage pre-built strategies for quick start
- **Validate Tickers**: Always verify ticker symbols before creating
- **Set Realistic Capital**: Use actual investment amounts for accurate analysis

### Analysis
- **Date Range Selection**:
  - **Short-term**: 3-6 months (for active trading)
  - **Medium-term**: 1 year (most common, balanced view)
  - **Long-term**: 3-5 years (for strategic planning)
- **Benchmark Selection**:
  - Tech portfolios: Use QQQ (NASDAQ-100)
  - Broad market: Use SPY (S&P 500) or VTI (Total Market)
  - Sector-specific: Use relevant sector ETF
- **Metric Interpretation**:
  - Sharpe Ratio: >1 is good, >2 is excellent
  - Max Drawdown: Consider your risk tolerance
  - Beta: <1 = less volatile, >1 = more volatile
  - Alpha: Positive = outperformance, negative = underperformance

### Optimization
- **Method Selection**:
  - **General use**: Mean-Variance (Max Sharpe)
  - **Stable results**: Black-Litterman
  - **Risk-focused**: Risk Parity or CVaR Optimization
  - **Many assets**: Hierarchical Risk Parity (HRP)
  - **Conservative**: Mean-CVaR or Robust Optimization
- **Constraint Guidelines**:
  - Min weight: 2-5% (prevents tiny positions)
  - Max weight: 20-30% (prevents concentration)
  - Cardinality: 10-20 positions (balance diversification and complexity)
- **Out-of-Sample Testing**: Always enable for validation (30% window recommended)

### Risk Management
- **VaR Usage**: Use multiple methods (Historical + Parametric + Cornish-Fisher) and compare
- **Stress Testing**: Test against historical crises and create custom scenarios for your specific risks
- **Monte Carlo**: Run for probabilistic analysis (10,000 paths recommended)
- **Drawdown Management**: Monitor current drawdown regularly, set drawdown limits

### Forecasting
- **Model Selection**:
  - **Short-term** (1-5 days): ARIMA or LSTM
  - **Medium-term** (1-3 months): ARIMA-GARCH or XGBoost
  - **Long-term** (6-12 months): Prophet or Ensemble
  - **Always**: Use ensemble for production decisions
- **Training Data**: Use sufficient history (at least 1 year, preferably 2-3 years)
- **Validation**: Always enable out-of-sample testing, review MAE, RMSE, MAPE metrics
- **Interpretation**: Trust confidence intervals, not just point forecasts

### Transaction Management
- **Record Promptly**: Enter transactions as soon as possible for accurate history
- **Include Fees**: Add transaction fees for accurate cost basis
- **Use Notes**: Add notes for context (e.g., "Rebalancing", "Dividend reinvestment")
- **Regular Review**: Review transaction history regularly for accuracy
- **Export Backup**: Export transactions to CSV regularly for backup

---

## Keyboard Shortcuts

**Chart Interactions** (Plotly):
- **Mouse Wheel**: Zoom in/out
- **Double-Click**: Reset zoom
- **Click and Drag**: Pan around chart
- **Hover**: See exact values
- **Click Legend**: Show/hide series
- **Right-Click**: Context menu (reset, export, etc.)

**Form Navigation**:
- **Tab**: Move to next field
- **Enter**: Submit form
- **Escape**: Close dialogs/modals

**Table Interactions**:
- **Click Column Header**: Sort by that column
- **Click Again**: Reverse sort order
- **Search Box**: Type to filter rows (real-time)

---

## Additional Resources

For technical documentation and architecture details, please refer to the project repository or contact the development team.

---

**For questions or issues, please open an issue on GitHub.**
