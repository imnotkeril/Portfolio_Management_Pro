# WILD MARKET CAPITAL - Complete Project Vision

## Vision Statement

Professional-grade portfolio management system combining institutional-quality analytics with modern, intuitive interface. The system should provide comprehensive tools for portfolio construction, risk management, performance analysis, and optimization - everything a professional portfolio manager needs in one integrated platform.

---

## Design Philosophy

### Visual Identity

**Inspired by**: TradingView professional interface with custom color scheme

**Color Palette**:

Primary Colors:
- Background: #0D1015
- Primary UI elements: #BF9FFB  
- Text: #FFFFFF
- Panel borders: #2A2E39

Chart Colors:
- Bullish candles: #D1D4DC
- Bearish candles: #BF9FFB
- Positive values: #74F174
- Negative values: #FAA1A4
- Volume bars: #D1D4DC
- Additional indicator 1: #90BFF9
- Additional indicator 2: #FFF59D

Interface Elements:
- Panel separators: #2A2E39 solid
- Price scale separators: transparent
- Crosshair: #9598A1 dashed

### Design Principles

1. **Clean & Professional** - No emojis, minimal decorations, focus on data
2. **Information Density** - Show maximum relevant information without clutter
3. **Responsive Layout** - Adapt to different screen sizes
4. **Consistent Spacing** - 8px grid system (8, 16, 24, 32px)
5. **Typography** - Inter for UI, JetBrains Mono for numbers
6. **Interactive Charts** - Zoom, pan, hover tooltips on all visualizations
7. **Dark Theme First** - Optimized for extended viewing sessions

---

## System Architecture Overview

```
BACKEND (Core Modules)
├── Data Manager
├── Analytics Engine
├── Risk Engine
├── Optimization Engine
├── Scenario Engine
└── Reporting Engine

FRONTEND (Streamlit MVP)
├── Dashboard
├── Portfolio Creation
├── Portfolio Management
├── Portfolio Analysis
├── Risk Management
├── Optimization
├── Scenario Analysis
├── Reports
└── Settings

FUTURE (React Web App)
└── Full web application with API backend
```

---

## Page Structure & Functionality

### 1. MAIN DASHBOARD

**Purpose**: Overview of all portfolios and key metrics at a glance

**Layout Sections**:

**Top Metrics Bar** (Fixed, 100px height):
- Total Portfolio Value
- Today's P&L (with percentage)
- Month-to-Date Return
- Year-to-Date Return  
- Current Risk Level indicator

**Portfolio Grid** (Main area, 60% width):
- Card-based portfolio display (2x3 grid)
- Each card shows:
  - Portfolio name and tags
  - Current value
  - Daily change (absolute and percentage)
  - 30-day mini sparkline chart
  - Number of positions
  - Quick action buttons (Analyze, Edit, Report)

**Market Overview Panel** (Right sidebar, 40% width):
- Major indices (S&P 500, NASDAQ, DOW)
- Sector performance heatmap
- Top movers (gainers/losers)
- Market breadth indicators
- VIX level

**Performance Charts** (Bottom section, horizontal scroll):
- Portfolio comparison chart (normalized returns)
- Asset allocation across all portfolios
- Risk-return scatter plot
- Monthly performance heatmap

**Features**:
- Real-time price updates (during market hours)
- Drag-and-drop portfolio reordering
- Favorite/pin portfolios
- Search and filter portfolios
- Quick portfolio creation button

---

### 2. PORTFOLIO CREATION

**Purpose**: Multiple methods to create new portfolios with validation

**Creation Methods**:

**Method 1: Text Input**
- Parse ticker lists from text
- Support formats:
  - "AAPL 30%, MSFT 25%, GOOGL 45%"
  - "AAPL 0.3, MSFT 0.25, GOOGL 0.45"
  - "AAPL:30 MSFT:25 GOOGL:45"
  - "AAPL,MSFT,GOOGL" (equal weight auto-assign)
- Real-time ticker validation
- Weight normalization

**Method 2: CSV/Excel Upload**
- Drag-and-drop file area
- Column mapping interface
- Required: ticker, weight
- Optional: name, sector, purchase_price, purchase_date
- Preview parsed data before creation
- Error reporting for invalid data

**Method 3: Manual Entry**
- Interactive table for adding assets
- Ticker autocomplete with company name
- Live price fetching
- Weight or dollar amount input
- Real-time weight sum validation
- Drag to reorder assets

**Method 4: Template-Based**
- Pre-defined portfolio templates:
  - 60/40 (stocks/bonds)
  - All Weather
  - Risk Parity
  - Sector Rotation
  - Dividend Growth
- Customize template weights
- Select specific tickers for each category

**Method 5: Clone Existing**
- Select portfolio to clone
- Modify name and settings
- Adjust weights if desired

**Portfolio Settings** (All methods):

Basic Information:
- Portfolio name (required)
- Description (optional)
- Tags (for categorization)
- Starting capital amount
- Base currency

Advanced Settings (Expandable):

Rebalancing:
- Frequency (none, monthly, quarterly, semi-annual, annual)
- Threshold trigger (%)
- Auto-rebalance on/off

Position Constraints:
- Minimum position size ($)
- Maximum position size ($)
- Maximum number of positions
- Position sizing method (equal, custom, market-cap)

Risk Management:
- Stop-loss level (%)
- Take-profit level (%)
- Maximum drawdown limit (%)
- VaR limit

Tax Settings:
- Enable tax-loss harvesting
- Avoid wash sales
- Tax rate (%)

ESG Filters:
- Exclude tobacco
- Exclude weapons
- Exclude fossil fuels  
- Minimum ESG score

Geographic Constraints:
- US only
- Developed markets only
- Include emerging markets
- Include frontier markets

Sector Constraints:
- Min/max allocation per sector

**Validation Dashboard**:
- All tickers valid (checkmark/error)
- Weights sum to 100% (checkmark/error)
- Constraint violations (warnings)
- Diversification score
- Expected analytics preview:
  - Historical return (based on backtest)
  - Historical volatility
  - Estimated Sharpe ratio
  - Projected max drawdown

**Finalization**:
- Review summary
- Allocation pie chart preview
- Sector breakdown preview
- Create portfolio button
- Save as template option

---

### 3. PORTFOLIO MANAGEMENT

**Purpose**: View, edit, and manage existing portfolios

**Portfolio List View**:
- Sortable/filterable table
- Columns:
  - Name
  - Value
  - # Positions  
  - Daily Change
  - MTD Return
  - YTD Return
  - Risk Level
  - Last Updated
  - Actions (Edit, Delete, Export, Clone)
- Bulk actions (delete multiple, export multiple)
- Import portfolio

**Portfolio Detail View**:

Header:
- Portfolio name (editable inline)
- Description (editable inline)
- Tags (editable)
- Created date
- Last modified date

Holdings Table:
- Ticker
- Company Name
- Sector
- Shares
- Current Price
- Purchase Price (if available)
- Current Value
- Weight (%)
- P&L ($)
- P&L (%)
- Actions (Edit, Remove)

Table Features:
- Sort by any column
- Color-coded P&L (green/red)
- In-line editing (shares, purchase price)
- Add new position button
- Rebalance to target weights button

Weight Adjustment:
- Pie chart with draggable slices
- Table with editable weight fields
- Auto-normalize button
- Calculate required trades
- Preview impact on metrics

Actions:
- Update prices (manual refresh)
- Rebalance portfolio
- Export holdings (CSV, Excel, JSON)
- Generate report
- Delete portfolio (with confirmation)
- Duplicate portfolio

Transaction History (if implemented):
- Date
- Action (Buy/Sell/Rebalance)
- Ticker
- Shares
- Price
- Value
- Notes

---

### 4. PORTFOLIO ANALYSIS (Main Analytics Page)

**Purpose**: Deep dive into portfolio performance, risk, and characteristics

**Control Panel** (Top, 80px height):
- Portfolio selector dropdown (with search)
- Date range picker:
  - Quick buttons: 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y, MAX
  - Custom date range
- Benchmark selector:
  - Popular: S&P 500, NASDAQ, Russell 2000
  - Custom portfolio
  - None
- Export button
- Settings gear

**Metrics Dashboard** (Expandable cards grid):

Card Layout:
- 4 columns on desktop
- Scrollable/collapsible
- Hover for definition
- Click for detailed chart/analysis

**Performance Metrics** (18 cards):
1. Total Return - Overall gain/loss over period
2. CAGR - Compound Annual Growth Rate
3. Annualized Return - Average yearly return
4. YTD Return - Year to date performance
5. MTD Return - Month to date
6. QTD Return - Quarter to date  
7. 1M Return - Last month
8. 3M Return - Last 3 months
9. 6M Return - Last 6 months
10. 1Y Return - Last year
11. 3Y Return - Last 3 years (if data available)
12. 5Y Return - Last 5 years (if data available)
13. Best Month - Highest monthly return
14. Worst Month - Lowest monthly return
15. Win Rate - Percentage of positive periods
16. Payoff Ratio - Average win / average loss
17. Profit Factor - Gross profit / gross loss
18. Expectancy - Expected value per period

**Risk Metrics** (22 cards):
1. Volatility (Daily) - Daily standard deviation  
2. Volatility (Weekly) - Weekly standard deviation
3. Volatility (Monthly) - Monthly standard deviation
4. Volatility (Annual) - Annualized volatility
5. Maximum Drawdown - Largest peak-to-trough decline
6. Current Drawdown - Current decline from peak
7. Average Drawdown - Mean of all drawdowns
8. Max Drawdown Duration - Longest drawdown period (days)
9. Average Drawdown Duration - Average time in drawdown
10. Recovery Time - Days to recover from max drawdown
11. Ulcer Index - Depth and duration of drawdown stress
12. Pain Index - Cumulative drawdown measure
13. VaR 90% - Value at Risk at 90% confidence
14. VaR 95% - Value at Risk at 95% confidence  
15. VaR 99% - Value at Risk at 99% confidence
16. CVaR 90% - Conditional VaR (Expected Shortfall) 90%
17. CVaR 95% - Conditional VaR 95%
18. CVaR 99% - Conditional VaR 99%
19. Downside Deviation - Volatility of negative returns only
20. Semi-Deviation - Below-mean volatility
21. Skewness - Return distribution asymmetry
22. Kurtosis - Tail weight of distribution

**Risk-Adjusted Ratios** (15 cards):
1. Sharpe Ratio - Excess return per unit of risk
2. Sortino Ratio - Return per unit of downside risk
3. Calmar Ratio - Return / max drawdown
4. Sterling Ratio - Return / average drawdown
5. Burke Ratio - Return / square root of sum squared drawdowns
6. Treynor Ratio - Excess return per unit of systematic risk
7. Information Ratio - Active return / tracking error
8. Modigliani Ratio (M²) - Risk-adjusted return vs benchmark
9. Omega Ratio - Prob. weighted gains / losses
10. Kappa 3 Ratio - Return / downside risk cubed
11. Gain-Pain Ratio - Sum gains / sum losses
12. Martin Ratio - Return / Ulcer Index
13. Tail Ratio - 95th percentile / 5th percentile
14. Common Sense Ratio - Profit factor * tail ratio
15. Rachev Ratio - CVaR ratio at different levels

**Market-Related Metrics** (15 cards):
1. Beta - Systematic risk vs benchmark
2. Alpha - Excess return above expected (CAPM)
3. R-Squared - Correlation to benchmark
4. Correlation - Linear relationship to benchmark
5. Tracking Error - Standard deviation of active returns
6. Active Return - Return above benchmark
7. Up Capture - Return in up markets / benchmark up return
8. Down Capture - Return in down markets / benchmark down return  
9. Up/Down Capture Ratio - Up capture / down capture
10. Jensen's Alpha - Risk-adjusted excess return
11. Treynor Ratio - Return per unit of systematic risk
12. Active Share - % of holdings different from benchmark
13. Batting Average - % of periods beating benchmark
14. Benchmark Relative Return - Cumulative outperformance
15. Rolling Beta (avg) - Average of rolling 30-day betas

**Tab Navigation** (Main content area):

**TAB 1: Performance**

Main Chart Options (dropdown selector):
- Cumulative Returns (line)
- Rolling Returns (line with configurable window)
- Drawdown (underwater curve)
- Daily Returns (bar chart)
- Monthly Returns Grid (heatmap)

Chart Controls:
- Log/Linear scale toggle
- Show/hide benchmark
- Show/hide buy&hold comparison
- Date range zoom
- Crosshair with values
- Export chart (PNG, SVG, CSV data)

Sub-Charts (2-column grid below main):
- Return Distribution (histogram with normal overlay)
- Q-Q Plot (normality test)
- Rolling Sharpe (line chart, 30-day window)
- Rolling Volatility (line chart, 30-day window)

Calendar View:
- Monthly returns heatmap (year rows, month columns)
- Color intensity = return magnitude
- Hover for exact values
- Click month to drill down to daily

Tables:
- Period Returns table (daily, weekly, monthly, quarterly, yearly)
- Best/Worst Periods table (top 10 / bottom 10 days/weeks/months)
- Consecutive Wins/Losses table

**TAB 2: Risk Analysis**

Risk Dashboard Summary:
- Overall risk score (0-100)
- Risk level indicator (Low/Moderate/High/Extreme)
- Key risk metrics in gauge charts

Subtabs:

**2a. VaR Analysis**
- Method selector (Historical, Parametric, Monte Carlo, Cornish-Fisher, All)
- Confidence level slider (90-99%)
- Time horizon selector (1-day, 1-week, 1-month)
- VaR calculation results:
  - VaR amount ($)
  - VaR percentage (%)
  - CVaR (Expected Shortfall)
  - Probability of exceeding VaR
- Distribution chart with VaR cutoff lines
- Backtesting results table:
  - Breaches count
  - Expected breaches
  - Kupiec test result
  - Basel traffic light
- Historical VaR vs actual returns scatter

**2b. Stress Testing**
- Historical Scenarios selector (checkboxes):
  - 2008 Financial Crisis
  - 2020 COVID Crash
  - 2000 Dot-com Bust  
  - 1987 Black Monday
  - 2011 European Debt Crisis
  - 2015 China Devaluation
  - 2018 Volatility Spike
  - 2022 Rate Hikes
  - 2023 Banking Crisis
  - Custom scenario builder
- Scenario results table:
  - Scenario name
  - Portfolio impact (%)
  - Portfolio impact ($)
  - Worst position impact
  - Best position impact
  - Recovery time estimate
- Waterfall chart of cumulative impacts
- Asset-level impact heatmap

**2c. Monte Carlo Simulation**
- Simulation parameters:
  - Number of paths (1,000 / 10,000 / 50,000)
  - Time horizon (days)
  - Model (GBM, Jump Diffusion, GARCH)
  - Return assumptions (Historical, Custom, Implied)
  - Volatility assumptions
- Run simulation button (with progress bar)
- Results:
  - Probability cone chart (5th, 25th, 50th, 75th, 95th percentiles)
  - Final value distribution histogram
  - Statistics table:
    - Median outcome
    - Mean outcome  
    - 5th / 95th percentile
    - Probability of loss
    - Probability of target achievement
    - Max simulated gain/loss

**2d. Risk Monitoring**
- Risk limit status table:
  - Metric
  - Current value
  - Limit
  - Status (OK/Warning/Breach)
- Time series of key risk metrics
- Concentration risk analysis:
  - Single position concentration
  - Sector concentration
  - Geographic concentration
  - Asset class concentration
- Correlation risk:
  - Correlation matrix heatmap (all positions)
  - Average pairwise correlation
  - Highly correlated pairs table
  - Correlation breakdown analysis

**TAB 3: Allocation**

Current Allocation:
- Interactive pie chart (hover, click to drill down)
- Treemap visualization (hierarchical)
- Sunburst chart (multi-level: asset class > sector > position)

Allocation Breakdown Tables:
- By Asset (ticker, name, weight, value)
- By Sector (sector, weight, # positions, avg position size)
- By Asset Class (stocks, bonds, cash, alternatives, etc)
- By Geography (US, International Developed, Emerging, etc)
- By Market Cap (Large, Mid, Small)

Allocation Over Time:
- Stacked area chart showing weight evolution
- Identify drift from targets
- Rebalancing events marked

Target vs Actual:
- Side-by-side bar chart
- Deviation table  
- Rebalancing suggestions

What-If Analysis:
- Adjust slider for each position
- See real-time impact on:
  - Expected return
  - Expected risk
  - Sharpe ratio
  - Sector/geo balance
- Compare to current allocation

**TAB 4: Factor Analysis**

Factor Models:
- Fama-French 3-Factor
- Fama-French 5-Factor  
- Carhart 4-Factor (adds Momentum)
- Custom factor selection

Regression Results:
- Factor loadings (bar chart)
- Factor loadings table:
  - Factor name
  - Loading (beta)
  - T-statistic
  - P-value
  - Significance indicator
- R-squared
- Adjusted R-squared
- Alpha (unexplained return)
- Residual standard error

Factor Exposures Over Time:
- Rolling factor loadings (line charts)
- Identify style drift

Factor Attribution:
- Decompose returns into:
  - Market factor contribution
  - Size factor contribution
  - Value factor contribution  
  - Momentum factor contribution
  - Alpha contribution
- Pie chart of attribution
- Time series of factor contributions

Style Analysis:
- Style box (Value/Blend/Growth × Large/Mid/Small)
- Style drift chart
- Style consistency metrics

**TAB 5: Contribution Analysis**

Return Attribution:
- Asset contribution to return (bar chart)
- Cumulative contribution over time (area chart)
- Best/worst contributors table

Risk Attribution:
- Asset contribution to portfolio risk (bar chart)
- Marginal contribution to risk
- Component VaR
- Sector risk contribution

Sector Attribution:
- Sector allocation effect
- Sector selection effect
- Interaction effect
- Total attribution

Brinson Attribution (vs Benchmark):
- Allocation effect
- Selection effect
- Interaction effect
- Total active return
- Attribution waterfall chart

**TAB 6: Holdings Detail**

Advanced Holdings Table:
- Ticker
- Name
- Sector / Industry
- Market Cap
- Current Price
- Purchase Price (if available)
- Shares  
- Weight
- Value
- % of Portfolio
- Cost Basis
- Unrealized P&L ($)
- Unrealized P&L (%)
- Daily Change ($)
- Daily Change (%)
- Dividend Yield
- P/E Ratio
- Beta
- 52-Week Range
- Sparkline (30-day)

Position Detail View (click to expand):
- Full company information
- Historical price chart (1Y)
- Fundamental metrics
- News feed
- Analyst recommendations
- Transaction history for this position
- Edit/Remove buttons

Filters & Sorting:
- Filter by sector
- Filter by P&L status (profitable/loss)
- Filter by weight threshold
- Sort by any column
- Search by ticker/name

Export Options:
- CSV
- Excel (formatted)
- PDF report

**TAB 7: Comparison**

Compare Against:
- Another portfolio (selector)
- Benchmark index
- Hypothetical portfolio

Comparison Types:

**Side-by-Side**:
- Metrics comparison table (all 70+ metrics)
- Color-code better performing
- Difference column  
- Percentage difference

**Performance Charts**:
- Cumulative returns overlay
- Rolling returns comparison
- Drawdown comparison
- Monthly returns side-by-side bars

**Risk-Return Scatter**:
- Plot both portfolios
- Efficient frontier reference
- Sharpe ratio lines

**Correlation Analysis**:
- Correlation between portfolios
- Rolling correlation
- Overlap analysis (common holdings)

---

### 5. RISK MANAGEMENT (Dedicated Risk Page)

**Purpose**: Centralized risk monitoring and management tools

**Risk Dashboard Header**:
- Overall Risk Score (0-100 with gauge)
- Risk Level (Low/Moderate/High/Extreme)
- Risk Budget Usage (%)
- VaR 95% ($ and %)
- Stress Test Summary (worst scenario impact)

**Multi-Portfolio Risk View**:
- Select multiple portfolios  
- Aggregate risk metrics
- Diversification benefits
- Correlation matrix across portfolios
- Total VaR (with diversification)

**Risk Limit Management**:
- Define risk limits:
  - Maximum VaR
  - Maximum drawdown
  - Maximum volatility
  - Maximum single position size
  - Maximum sector concentration
  - Beta range
  - Minimum Sharpe ratio
- Monitor current vs limits
- Alert system for breaches
- Historical limit breach log

**Tail Risk Analysis**:
- Extreme Value Theory (EVT) analysis
- Tail index estimation
- Probability of extreme losses
- Stress scenarios focused on tail events

**Liquidity Risk**:
- Position liquidity scores
- Average daily volume vs position size
- Days to liquidate estimate
- Liquidity risk by market conditions

**Concentration Risk**:
- Single name concentration
- Sector concentration (Herfindahl index)
- Geographic concentration
- Asset class concentration
- Currency exposure
- Concentration limits and current status

**Correlation Risk**:
- Current correlation matrix
- Historical correlation evolution
- Correlation breakdown risk
- PCA analysis (principal components)
- Factor correlation

**Scenario Library**:
- Pre-built scenarios (25+ historical)
- Custom scenario builder:
  - Define market shocks (equity, rates, FX, vol, credit)
  - Set magnitudes and probabilities
  - Chain events together
  - Feedback loops
- Scenario impact calculator
- Scenario comparison tool

**Risk Reports**:
- Daily risk report
- Weekly risk summary
- Monthly risk review
- Custom risk report builder

---

### 6. PORTFOLIO OPTIMIZATION

**Purpose**: Find optimal portfolio weights using various methods

**Current Portfolio Analysis**:
- Efficiency Score (0-100)
- Current position on efficient frontier
- Issues detected:
  - Below efficient frontier
  - High concentration
  - Poor diversification
  - Suboptimal factor exposures
  - Constraint violations

**Optimization Method Selection**:

**Classic Methods**:
1. Mean-Variance (Markowitz)
   - Target: Minimize risk for given return OR maximize return for given risk
   - Best for: Traditional optimization with normal distributions

2. Minimum Variance
   - Target: Lowest possible portfolio risk
   - Best for: Risk-averse investors

3. Maximum Sharpe Ratio
   - Target: Best risk-adjusted returns
   - Best for: Maximizing efficiency

4. Maximum Return
   - Target: Highest expected return
   - Best for: Aggressive growth (with risk constraints)

**Risk-Based Methods**:
5. Risk Parity
   - Target: Equal risk contribution from each asset
   - Best for: Balanced risk exposure

6. Hierarchical Risk Parity (HRP)
   - Target: Diversification using machine learning  
   - Best for: Large portfolios, reduces estimation error

7. Maximum Diversification
   - Target: Highest diversification ratio
   - Best for: Maximum risk reduction through diversification

8. Minimum Correlation
   - Target: Lowest average pairwise correlation
   - Best for: Crisis-resistant portfolios

**Advanced Methods**:
9. Black-Litterman
   - Target: Combine market equilibrium with investor views
   - Best for: Incorporating market outlooks
   - Requires: Specify views (asset, expected return, confidence)

10. Robust Optimization
    - Target: Account for parameter uncertainty
    - Best for: Stable allocations over time
    - Reduces sensitivity to input estimation errors

11. CVaR Optimization  
    - Target: Minimize tail risk (Expected Shortfall)
    - Best for: Downside risk management

12. Mean-CVaR
    - Target: Optimize return per unit of tail risk
    - Best for: Balance return and extreme risk

13. Kelly Criterion
    - Target: Maximize long-term growth rate
    - Best for: Aggressive traders with leverage
    - Includes: Full Kelly, Half Kelly options

14. Equal Weight
    - Target: 1/N allocation
    - Best for: Simplicity, low turnover

15. Market Cap Weight
    - Target: Weight by market capitalization
    - Best for: Passive indexing

16. Minimum Tracking Error  
    - Target: Minimize deviation from benchmark
    - Best for: Enhanced indexing strategies

17. Maximum Alpha
    - Target: Maximize active return vs benchmark
    - Best for: Active management

**Constraint Configuration**:

Weight Constraints:
- Long-only vs long-short
- Minimum weight per asset (%)
- Maximum weight per asset (%)
- Specific asset weight bounds

Group Constraints:
- Sector limits (min/max per sector)
- Geographic limits
- Asset class limits
- Market cap limits

Risk Constraints:
- Maximum portfolio volatility  
- Maximum VaR
- Maximum CVaR
- Maximum beta
- Minimum Sharpe ratio
- Maximum tracking error (if benchmark)

Turnover Constraints:
- Maximum total turnover (%)
- Turnover penalty (cost per %)
- Minimum trade size ($)
- Round lot constraints

Transaction Costs:
- Proportional costs (% of trade)
- Fixed costs ($ per trade)
- Market impact model
- Bid-ask spread consideration

Cardinality:
- Minimum number of assets
- Maximum number of assets
- Maximum new positions
- Maximum closed positions

**Optimization Results**:

Summary Statistics:
- Expected Return (before → after)
- Expected Risk (before → after)
- Sharpe Ratio (before → after)
- Turnover required
- Number of trades
- Estimated transaction costs
- Break-even period

Efficient Frontier:
- Interactive scatter plot
- Current portfolio (marked)
- Optimal portfolio (marked)
- Efficient frontier curve
- Capital Market Line
- Click point on frontier to see weights

Weight Changes Table:
- Ticker
- Current Weight
- Optimal Weight  
- Change (pp)
- Change (%)
- Action (Buy/Sell/Hold)
- Shares to trade
- Dollar amount
- % of portfolio

Trade List:
- Executable trade instructions
- Order type recommendations
- Execution strategy suggestions
- Estimated slippage
- Total cost estimate

Allocation Comparison:
- Before/After pie charts
- Before/After sector breakdown
- Before/After factor exposures
- Before/After risk metrics

Sensitivity Analysis:
- How results change with:
  - Different target returns
  - Different risk levels
  - Different constraint levels
  - Different cost assumptions

Backtest (Optional):
- Historical performance of optimal weights
- Rebalancing simulation
- Cost drag analysis
- Compare to current portfolio

Actions:
- Save optimal weights as new portfolio
- Create rebalancing orders
- Schedule future rebalancing
- Export results (PDF, Excel)

---

### 7. SCENARIO ANALYSIS

**Purpose**: Test portfolio under hypothetical market conditions

**Scenario Builder**:

**Pre-Built Scenarios** (Historical events):
- 2008 Financial Crisis (Sep-Dec 2008)
- 2020 COVID Crash (Feb-Mar 2020)
- 2000 Dot-com Bust (Mar 2000-Oct 2002)  
- 1987 Black Monday (Oct 1987)
- 1997 Asian Crisis
- 1998 Russian Default / LTCM
- 2010 Flash Crash
- 2011 European Debt Crisis
- 2013 Taper Tantrum
- 2015 China Devaluation
- 2016 Brexit
- 2018 Q4 Selloff
- 2022 Rate Hike Cycle
- 2023 Banking Crisis
- Plus 10+ more

Each includes:
- Equity shock (%)
- Bond shock (%)
- FX movements
- Credit spread change
- Volatility spike
- Commodity impacts
- Duration (days)

**Custom Scenario Creator**:

Market Factors:
- Equity Markets:
  - US Large Cap (%)
  - US Small Cap (%)
  - International Developed (%)
  - Emerging Markets (%)
- Fixed Income:
  - Treasury yields (bp change)
  - IG Corporate spreads (bp)
  - HY Corporate spreads (bp)
- Currencies:
  - USD Index (%)
  - EUR/USD (%)
  - Other pairs
- Commodities:
  - Oil (%)  
  - Gold (%)
  - Other commodities
- Volatility:
  - VIX level
  - Realized vol change

Scenario Parameters:
- Severity (mild, moderate, severe, extreme)
- Duration (days/weeks/months)
- Speed (sudden, gradual)
- Recovery shape (V, U, L, W)

**Scenario Chain Builder**:

Visual Chain Interface:
- Drag events to chain
- Define trigger conditions
- Set transition probabilities
- Feedback loops

Available Events:
- Rate Hike/Cut
- Market Rally/Selloff
- Recession
- Economic Expansion
- Credit Crunch
- Liquidity Crisis
- Bank Failure
- Geopolitical Crisis
- Natural Disaster
- Pandemic
- Tech Disruption
- Policy Change
- Custom Event

Example Chains:
- "Recession Chain": Rate Hikes → Credit Crunch → Recession → Recovery
- "Inflation Chain": Supply Shock → Inflation Surge → Rate Response → Slowdown
- "Crisis Chain": Geopolitical Event → Market Panic → Flight to Safety → Normalization

**Scenario Analysis Results**:

Impact Summary:
- Portfolio value impact ($)
- Portfolio value impact (%)
- Recovery time estimate (days)
- Peak drawdown
- Prob-weighted expected loss

Asset-Level Impact:
- Heatmap: each position × each scenario
- Best/worst performing positions
- Positions causing most loss
- Defensive positions

Sector Impact:
- Sector performance in scenario
- Sector contribution to total loss  
- Sector correlations in stress

Geographic Impact:
- Regional performance
- Currency effects
- Country-specific factors

Factor Impact:
- Factor exposures under stress
- Factor returns in scenario
- Style biases revealed

Path Analysis (for chains):
- Probability tree diagram
- All possible paths
- Path probabilities
- Expected value per path
- Worst-case path
- Best-case path

Hedging Recommendations:
- Suggested hedge instruments:
  - Put options on indices
  - Long volatility positions
  - Increase cash allocation
  - Add defensive sectors
  - Geographic diversification
  - Currency hedges
- Hedge effectiveness estimates
- Cost-benefit analysis
- Optimal hedge ratios

Stress Test Matrix:
- Multiple scenarios × multiple portfolios
- Aggregate risk view
- Identify systemic vulnerabilities

Historical Analogy Finder:
- Current market conditions input:
  - Volatility level
  - Interest rate environment  
  - Economic growth
  - Inflation level
  - Credit conditions
  - Valuation metrics
- Find similar historical periods
- Show what happened next
- Portfolio performance in those periods
- Lessons and implications
- Probability-weighted forecasts

---

### 8. BACKTESTING

**Purpose**: Test strategies and rebalancing rules on historical data

**Strategy Definition**:

Rebalancing Rules:
- Calendar-based (monthly, quarterly, etc)
- Threshold-based (drift > X%)
- Volatility-target
- Risk-parity maintenance
- Custom trigger logic

Optimization Schedule:
- Fixed weights (buy & hold)
- Periodic re-optimization (monthly, quarterly)
- Rolling-window optimization
- Adaptive optimization

Transaction Costs:
- Proportional (%)
- Fixed per trade ($)
- Market impact model
- Bid-ask spread

**Backtest Configuration**:
- Start date
- End date  
- Initial capital
- Benchmark
- Rebalancing frequency
- Transaction cost assumptions
- Slippage assumptions

**Backtest Results**:

Performance Summary:
- Total Return
- CAGR
- Volatility
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Calmar Ratio
- All 70+ metrics

vs Buy & Hold:
- Outperformance (%)
- Risk reduction (%)
- Sharpe improvement
- Drawdown comparison

vs Benchmark:
- Active return
- Tracking error
- Information ratio
- Alpha
- Up/down capture

Equity Curve:
- Cumulative returns chart
- Drawdown chart
- Rolling returns
- Monthly returns grid

Transaction Analysis:
- Total trades
- Win rate
- Average profit/loss per trade
- Total transaction costs  
- Cost as % of returns
- Turnover by period

Rebalancing Events:
- Timeline of rebalances
- Trades at each rebalance
- Cost per rebalance
- Impact on returns

Period Analysis:
- Performance by year
- Performance by quarter
- Performance by month
- Best/worst periods

Robustness Tests:
- Sensitivity to costs
- Sensitivity to slippage
- Out-of-sample performance
- Walk-forward analysis results

---

### 9. REPORTS

**Purpose**: Generate professional PDF/Excel reports

**Report Types**:

**1. Executive Summary** (1 page)
- Portfolio overview
- Key metrics (8-10 most important)
- Performance chart (YTD)
- Current allocation pie chart
- Brief commentary

**2. Performance Tearsheet** (2-3 pages)
- All return metrics
- Performance charts  
- Period returns table
- Best/worst periods
- vs Benchmark comparison

**3. Risk Report** (3-5 pages)
- All risk metrics
- VaR analysis
- Stress test results
- Risk monitoring dashboards
- Concentration analysis
- Recommendations

**4. Comprehensive Report** (15-20 pages)
- Executive summary
- Performance analysis
- Risk analysis
- Attribution analysis
- Holdings detail
- Factor analysis
- Recommendations
- Appendix (methodology)

**5. Monthly Review** (5-7 pages)
- Month summary
- Performance vs benchmark
- Attribution
- Significant events
- Changes made
- Outlook

**6. Quarterly Review** (8-12 pages)  
- Quarter summary
- Detailed performance
- Risk review
- Rebalancing activity
- Market commentary
- Strategy review
- Forward outlook

**7. Custom Report**
- Select sections to include
- Choose metrics
- Select charts
- Add custom commentary
- Branding options

**Report Configuration**:

Content Selection:
- Choose sections
- Choose metrics
- Choose charts
- Choose time period

Branding:
- Company logo
- Color scheme
- Header/footer text
- Contact information

Format Options:
- PDF (print-ready)
- Excel (editable)
- PowerPoint (presentation)
- HTML (email/web)

Distribution:
- Download
- Email to recipients
- Save to cloud  
- Schedule recurring

**Report Sections Available**:
- Cover page
- Table of contents
- Executive summary
- Portfolio summary
- Performance metrics
- Performance charts
- Risk metrics
- Risk analysis
- Allocation breakdown
- Holdings detail
- Transaction history
- Attribution analysis
- Factor analysis
- Benchmark comparison
- Market commentary
- Recommendations
- Methodology appendix
- Disclosures

---

### 10. SETTINGS

**Purpose**: Configure application preferences and data sources

**User Preferences**:
- Theme (dark/light)
- Default currency
- Number formatting (decimal places, separators)
- Date format
- Default time period for charts
- Default benchmark
- Chart preferences (colors, style)

**Portfolio Defaults**:
- Default rebalancing frequency
- Default constraints
- Default risk limits  
- Default transaction costs

**Data Providers**:
- Primary data source selection:
  - Yahoo Finance (free)
  - Alpha Vantage (API key required)
  - IEX Cloud (API key required)
  - Polygon.io (API key required)
- Backup data sources
- Update frequency
- Cache settings

**Risk-Free Rate**:
- Manual input
- Auto-fetch (US 3-month T-bill)
- Historical values

**Calculation Settings**:
- Trading days per year (252 default)
- Return calculation method (simple, log)
- Volatility window (default periods)
- VaR confidence levels
- Monte Carlo simulation defaults

**Export Settings**:
- Default export format
- Default file location
- Include raw data
- Date in filename

**Notifications** (Future):
- Email alerts on risk breaches
- Daily portfolio summary
- Weekly performance report

**API Keys Management**:
- Enter/edit API keys
- Test connectivity
- Usage quotas monitoring

**Data Management**:
- Clear cache
- Re-download historical data
- Data storage location
- Backup portfolios
- Import portfolios
- Export all portfolios

---

## Data & Analytics Requirements

### Data Sources

**Market Data** (Required):
- Historical daily prices (OHLCV)
- Real-time prices (during market hours)
- Adjusted prices (splits, dividends)
- Benchmark index data (S&P 500, etc)

**Fundamental Data** (Nice-to-have):
- Company info (name, sector, industry)
- Market capitalization
- P/E ratio, dividend yield
- Basic financials

**Risk-Free Rate**:
- US 3-month T-bill rate
- Historical values

**Factor Data** (If factor analysis implemented):
- Fama-French factors
- Momentum factor
- Custom factors

### Calculation Engine Requirements

**Performance Calculations**:
- All return metrics (18 listed)
- Cumulative returns
- Period returns
- Rolling returns
- Time-weighted returns
- Money-weighted returns (if cash flows)

**Risk Calculations**:
- All risk metrics (22 listed)
- VaR (multiple methods)
- CVaR / Expected Shortfall
- Drawdown analysis
- Volatility measures

**Ratio Calculations**:
- All risk-adjusted ratios (15 listed)
- Sharpe, Sortino, Calmar, etc
- Information ratio
- Capture ratios

**Portfolio Math**:
- Portfolio returns from constituent returns
- Portfolio volatility from covariance matrix
- Efficient frontier calculation
- Optimization algorithms
- Rebalancing calculations

**Statistical Functions**:
- Mean, median, std dev
- Percentiles, quantiles
- Skewness, kurtosis  
- Correlation, covariance
- Regression analysis
- Monte Carlo simulation

### Performance Requirements

**Speed**:
- Page load: < 2 seconds
- Calculations: < 1 second for typical portfolio
- Optimization: < 5 seconds
- Monte Carlo (10k): < 10 seconds
- Report generation: < 30 seconds

**Scalability**:
- Support portfolios up to 100 positions
- Handle 10+ years of daily data
- Support 50+ portfolios per user

**Data Caching**:
- Cache historical prices (5 min TTL for current prices)
- Cache calculation results
- Invalidate on data changes

---

## User Workflows

### Workflow 1: Create New Portfolio and Analyze

1. User clicks "Create Portfolio"
2. Chooses text input method
3. Pastes: "AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, NVDA 10%"
4. System validates tickers, fetches prices  
5. User names portfolio "Tech Leaders"
6. User sets starting capital: $100,000
7. User clicks "Create"
8. System calculates initial metrics, shows success
9. User navigates to Portfolio Analysis
10. Selects "Tech Leaders" from dropdown
11. Reviews all 70+ metrics
12. Examines performance charts
13. Checks risk metrics, notes high volatility
14. Reviews allocation breakdown
15. Exports report to PDF

### Workflow 2: Optimize Existing Portfolio

1. User opens Portfolio Management
2. Selects "Tech Leaders" portfolio
3. Clicks "Optimize" button
4. System opens Optimization page with portfolio loaded
5. User reviews current efficiency score: 68/100
6. User selects optimization method: "Maximum Sharpe Ratio"
7. User sets constraints:
   - Max weight per asset: 30%
   - Max volatility: 18%
8. User clicks "Optimize"  
9. System calculates, shows results:
   - Sharpe improves 0.85 → 1.15
   - Risk decreases 20% → 18%
   - Requires 35% turnover
10. User reviews weight changes table
11. User examines efficient frontier chart
12. User satisfied, clicks "Apply Optimization"
13. Option: Create new portfolio OR Update existing
14. User creates new portfolio "Tech Leaders - Optimized"

### Workflow 3: Risk Assessment and Stress Testing

1. User opens Risk Management page
2. Selects "Tech Leaders" portfolio
3. Reviews overall risk score: 72/100 (High)
4. Examines VaR:
   - 95% 1-day VaR: -$2,340 (-2.34%)
5. Runs stress test:
   - Selects "2008 Financial Crisis" scenario
   - Selects "2020 COVID Crash" scenario
6. Reviews results:
   - 2008: -38% portfolio impact
   - 2020: -26% portfolio impact  
7. Identifies NVDA as highest risk contributor
8. Examines hedging recommendations:
   - Reduce tech concentration
   - Add defensive sectors
   - Increase cash to 10%
9. User decides to create hedged version
10. Returns to Optimization page
11. Adds sector constraints:
    - Tech max: 60% (down from 100%)
    - Add Healthcare min: 10%
    - Add Consumer Staples min: 10%
12. Re-optimizes with new constraints
13. Creates "Tech Leaders - Hedged" portfolio

### Workflow 4: Monthly Portfolio Review

1. User opens Dashboard
2. Reviews all portfolios at a glance
3. Notes "Tech Leaders" is up 8.2% MTD
4. Clicks "Analyze" on Tech Leaders
5. Navigates to Performance tab
6. Sets date range: Last 30 days  
7. Reviews metrics:
   - MTD return: 8.2%
   - vs S&P 500: +3.1%
   - Max drawdown: -4.2%
8. Examines daily returns chart
9. Notes 3 days of significant moves
10. Navigates to Attribution tab
11. Sees NVDA contributed +4.1% to total return
12. GOOGL lagged, contributed only +0.8%
13. Navigates to Holdings tab
14. Reviews position-level performance
15. Decides to rebalance back to target weights
16. Clicks "Rebalance to Targets"
17. Reviews proposed trades
18. Confirms rebalancing
19. Generates monthly report
20. Exports to PDF
21. Saves report to records

---

## Chart & Visualization Library

### Chart Types Needed

**Line Charts**:
- Cumulative returns  
- Rolling metrics
- Time series of any metric
- Efficient frontier
- Factor exposures over time

**Bar Charts**:
- Daily/monthly returns
- Performance comparison
- Attribution contributions
- Before/after comparisons

**Area Charts**:
- Allocation over time (stacked)
- Drawdown (filled)
- Probability cone (Monte Carlo)

**Scatter Plots**:
- Risk-return
- Portfolio comparison
- Efficient frontier with points

**Heatmaps**:
- Correlation matrix
- Monthly returns calendar
- Scenario impact matrix
- Sector performance

**Pie/Donut Charts**:
- Current allocation  
- Sector breakdown
- Attribution contributions

**Treemap**:
- Hierarchical allocation
- Position sizing visualization

**Sunburst**:
- Multi-level allocation (asset class > sector > position)

**Waterfall Charts**:
- Attribution analysis
- Scenario impact breakdown

**Box Plots**:
- Return distributions
- Risk distributions across scenarios

**Gauge Charts**:
- Risk score (0-100)
- Efficiency score

**Sparklines**:
- Mini charts in tables
- 30-day price trends

### Chart Features (All Charts)

- **Interactive**: Hover tooltips, zoom, pan
- **Responsive**: Adapt to container size
- **Customizable**: Colors, labels, legends
- **Exportable**: PNG, SVG, CSV data
- **Crosshair**: Show values at cursor  
- **Legends**: Toggle series on/off
- **Annotations**: Mark events, thresholds
- **Multiple Y-axes**: When comparing different scales
- **Date range selector**: Brush to zoom
- **Comparison**: Overlay multiple series

### Chart Color Usage

**Performance Charts**:
- Portfolio line: #BF9FFB
- Benchmark line: #90BFF9
- Positive bars: #74F174
- Negative bars: #FAA1A4

**Risk Charts**:
- VaR line: #FAA1A4
- Distribution fill: #BF9FFB with transparency
- Breach points: #FAA1A4

**Allocation Charts**:
- Use color palette:
  - #BF9FFB
  - #90BFF9
  - #74F174
  - #FFF59D
  - #FAA1A4
  - #D1D4DC
- Assign consistently to sectors

**Heatmaps**:
- Diverging scale:
  - Negative: #FAA1A4 (light) → #8B0000 (dark red)
  - Zero: #D1D4DC
  - Positive: #74F174 (light) → #006400 (dark green)

---

## Tables & Data Display

### Table Features (All Tables)

- **Sortable**: Click column header to sort
- **Filterable**: Search/filter rows
- **Selectable**: Select rows for bulk actions
- **Resizable**: Drag columns to resize
- **Reorderable**: Drag columns to reorder
- **Sticky Header**: Header stays visible when scrolling
- **Pagination**: For large datasets (50-100 rows per page)
- **Export**: CSV, Excel, Copy to clipboard
- **Formatting**:
  - Numbers: Right-aligned, thousands separator
  - Currency: $ symbol, 2 decimals  
  - Percentages: % symbol, 2 decimals
  - Dates: YYYY-MM-DD or user preference
  - Color-coding: Green/red for positive/negative

### Key Tables

**Holdings Table**:
- Columns: Ticker, Name, Sector, Shares, Price, Value, Weight, P&L ($), P&L (%), Change ($), Change (%)
- Color-code P&L
- Sparkline column for 30-day trend
- Expandable rows for detail
- Footer row with totals

**Metrics Table**:
- Two columns: Metric Name, Value
- Group by category (Performance, Risk, Ratios)
- Collapsible groups
- Hover for definition
- Click to see detailed chart

**Comparison Table**:
- Columns: Metric, Portfolio A, Portfolio B, Difference
- Color-code better value
- Highlight significant differences

**Transaction Table**:
- Columns: Date, Action, Ticker, Shares, Price, Value, Fees, Notes  
- Filter by date range
- Filter by action type
- Running balance column

**Backtest Results Table**:
- Columns: Period, Return, Benchmark Return, Active Return, Volatility, Sharpe
- Row per year/quarter/month
- Summary row at bottom

---

## Export Formats

### CSV Export
- Standard comma-separated
- UTF-8 encoding
- Column headers
- Date format: YYYY-MM-DD
- Number format: No thousands separator, use decimal point

### Excel Export
- .xlsx format
- Formatted cells (colors, borders)
- Multiple sheets if applicable
- Charts embedded (if report)
- Conditional formatting on metrics
- Frozen header row
- Auto-filter enabled

### PDF Export
- Letter or A4 size
- Professional layout  
- Embedded charts (high resolution)
- Page numbers
- Table of contents (for long reports)
- Branding (logo, colors)

### JSON Export
- Portfolio structure
- Positions array
- Metadata
- Formatted for re-import

---

## Import Formats

### CSV Import
- Required columns: ticker, weight
- Optional columns: name, sector, shares, purchase_price, purchase_date
- First row = headers
- Support common variations (symbol, allocation, etc)

### Excel Import
- .xlsx and .xls
- Select sheet
- Column mapping UI
- Skip rows option
- Preview before import

### JSON Import
- Standard portfolio JSON structure
- Validation before import
- Merge or replace options

### Text Import
- Free-form text parsing
- Multiple formats:
  - "AAPL 30%, MSFT 25%"  
  - "AAPL 0.3, MSFT 0.25"
  - "AAPL:30 MSFT:25"
  - "AAPL,MSFT,GOOGL" (equal weight)
- Robust regex parsing
- Error messages for unparseable input

---

## Error Handling & Validation

### Input Validation

**Ticker Validation**:
- Check ticker exists
- Fetch basic info (name, price)
- Show validation status (checkmark, loading, error)
- Suggest corrections for typos

**Weight Validation**:
- Weights sum to 100% (or 1.0)
- No negative weights (unless short selling enabled)
- Within min/max bounds
- Auto-normalize option

**Constraint Validation**:
- Min < Max checks
- Logical consistency
- Feasibility checks

**Date Validation**:
- Start < End
- Within available data range  
- Reasonable ranges (not 100 years)

### Error Messages

**User-Friendly**:
- Clear description of problem
- Suggestion for fix
- Examples of valid input

**Error Types**:
- Validation errors (red, blocking)
- Warnings (yellow, non-blocking)
- Info messages (blue)
- Success messages (green)

**Error Display**:
- Inline (next to field)
- Toast notifications (temporary)
- Modal dialogs (important errors)
- Error log (for debugging)

### Loading States

**Indicators**:
- Spinner for calculations
- Progress bar for long operations
- Skeleton screens for data loading
- Disable buttons during processing

**Timeouts**:
- Show message if operation takes > 10 seconds  
- Allow cancellation of long operations
- Retry logic for failed API calls

---

## Performance Considerations

### Optimization Strategies

**Data Loading**:
- Lazy load: Only fetch data when needed
- Pagination: Don't load all data at once
- Caching: Cache expensive calculations and API results
- Incremental updates: Only update changed data

**Calculations**:
- Batch calculations when possible
- Use vectorized operations (NumPy/Pandas)
- Cache results with TTL
- Background jobs for heavy computations

**UI Rendering**:
- Virtual scrolling for large tables
- Debounce user inputs
- Throttle chart updates
- Memoize expensive components

**Memory Management**:
- Clear unused data  
- Limit cache size
- Stream large datasets
- Garbage collection for completed tasks

---

## Future Enhancements (Not MVP, but good to consider)

### Phase 2 Features:
- Real-time WebSocket price updates
- Multi-currency support
- Options and derivatives
- Bonds and fixed income
- Alternative assets (crypto, commodities, real estate)
- Tax-loss harvesting automation
- Dividend tracking and reinvestment
- Corporate actions handling (splits, mergers)
- Transaction cost analysis
- Slippage modeling
- Order execution simulation

### Phase 3 Features:
- Multi-user support
- User authentication and authorization
- Portfolio sharing
- Collaborative portfolios  
- Comments and annotations
- Audit trail
- Alerts and notifications system
- Email/SMS alerts on risk breaches
- Scheduled reports
- API access for external tools
- Mobile app
- Integration with brokers (import positions)
- Paper trading

### Advanced Analytics:
- Machine learning factor models
- Regime detection
- Market timing models
- Sentiment analysis
- Event studies
- Custom indicators and signals
- Strategy backtesting with complex rules
- Walk-forward optimization
- Genetic algorithms for optimization
- Deep learning for return prediction

---

## Success Metrics

### Functional Completeness:
- All 9 pages implemented
- All 70+ metrics calculating correctly
- All 17 optimization methods working  
- All 25+ stress scenarios available
- All chart types rendering correctly
- All import/export formats functional

### Performance Targets:
- Page load < 2 seconds
- Calculations < 1 second (typical portfolio)
- Optimization < 5 seconds
- Monte Carlo (10k) < 10 seconds
- Report generation < 30 seconds

### User Experience:
- Intuitive navigation
- Clear error messages
- Responsive on desktop (laptop and larger)
- No workflow dead-ends
- Undo where appropriate
- Autosave of work in progress

### Data Quality:
- Accurate calculations (validated against known results)
- No data loss
- Proper error handling
- Graceful degradation when data unavailable

### Code Quality:
- Modular architecture (core vs UI separated)
- Well-documented  
- Tested (unit tests for core logic)
- No hardcoded values
- Configurable parameters
- Extensible for future additions

---

## Non-Functional Requirements

### Reliability:
- Handle missing data gracefully
- Recover from API failures
- Validate all inputs
- Prevent data corruption

### Maintainability:
- Clean code structure
- Separation of concerns (core backend vs UI)
- Configuration files for settings
- Logging for debugging
- Comments for complex logic

### Security:
- API keys stored securely (environment variables)
- Input sanitization
- No sensitive data in logs
- Secure data storage

### Usability:
- Consistent UI patterns
- Helpful tooltips and documentation
- Keyboard shortcuts for power users  
- Responsive feedback to actions
- Undo/redo where applicable

### Accessibility (Future):
- Screen reader support
- Keyboard navigation
- High contrast mode
- Customizable font sizes

---

## Documentation Requirements

### User Documentation:
- Getting started guide
- Feature tutorials
- Video walkthroughs
- FAQ
- Glossary of terms
- Keyboard shortcuts reference

### Technical Documentation:
- API documentation (for future web app)
- Data models
- Calculation methodologies
- Architecture overview
- Deployment guide
- Configuration guide

### In-App Help:
- Tooltips on metrics (definition, interpretation)
- Help icons with pop-up explanations  
- Contextual help on each page
- Example portfolios and walkthroughs

---

## Appendix: Complete Metrics List

### Performance Metrics (18):
1. Total Return
2. CAGR (Compound Annual Growth Rate)
3. Annualized Return
4. YTD Return
5. MTD Return
6. QTD Return
7. 1M Return
8. 3M Return
9. 6M Return
10. 1Y Return
11. 3Y Return
12. 5Y Return
13. Best Month
14. Worst Month
15. Win Rate
16. Payoff Ratio
17. Profit Factor
18. Expectancy

### Risk Metrics (22):
1. Daily Volatility
2. Weekly Volatility  
3. Monthly Volatility
4. Annual Volatility
5. Maximum Drawdown
6. Current Drawdown
7. Average Drawdown
8. Max Drawdown Duration
9. Avg Drawdown Duration
10. Recovery Time
11. Ulcer Index
12. Pain Index
13. VaR 90%
14. VaR 95%
15. VaR 99%
16. CVaR 90%
17. CVaR 95%
18. CVaR 99%
19. Downside Deviation
20. Semi-Deviation
21. Skewness
22. Kurtosis

### Risk-Adjusted Ratios (15):
1. Sharpe Ratio
2. Sortino Ratio
3. Calmar Ratio
4. Sterling Ratio
5. Burke Ratio  
6. Treynor Ratio
7. Information Ratio
8. Modigliani Ratio (M²)
9. Omega Ratio
10. Kappa 3 Ratio
11. Gain-Pain Ratio
12. Martin Ratio
13. Tail Ratio
14. Common Sense Ratio
15. Rachev Ratio

### Market-Related Metrics (15):
1. Beta
2. Alpha
3. R-Squared
4. Correlation
5. Tracking Error
6. Active Return
7. Up Capture
8. Down Capture
9. Up/Down Capture Ratio
10. Jensen's Alpha
11. Active Share
12. Batting Average
13. Benchmark Relative Return  
14. Rolling Beta (average)
15. Market Timing Ratio

**Total: 70 Metrics**

---

## Appendix: Optimization Methods

1. Mean-Variance (Markowitz)
2. Minimum Variance
3. Maximum Sharpe Ratio
4. Maximum Return
5. Risk Parity
6. Hierarchical Risk Parity
7. Maximum Diversification
8. Minimum Correlation
9. Black-Litterman
10. Robust Optimization
11. CVaR Optimization
12. Mean-CVaR
13. Kelly Criterion
14. Equal Weight
15. Market Cap Weight
16. Minimum Tracking Error
17. Maximum Alpha

---

## Appendix: Stress Scenarios

**Historical Scenarios (25+)**:
1. 2008 Financial Crisis
2. 2020 COVID-19 Crash  
3. 2000 Dot-com Bust
4. 1987 Black Monday
5. 1997 Asian Crisis
6. 1998 Russian Default / LTCM
7. 2010 Flash Crash
8. 2011 European Debt Crisis
9. 2013 Taper Tantrum
10. 2015 China Devaluation
11. 2015 Oil Price Collapse
12. 2016 Brexit
13. 2018 Q4 Volatility
14. 2018 Trade War
15. 2020 March Volatility Spike
16. 2021 GameStop / Meme Stock
17. 2021 Archegos Collapse
18. 2022 Rate Hike Cycle
19. 2022 Russia-Ukraine War
20. 2022 Energy Crisis
21. 2023 Banking Crisis (SVB)
22. 2023 Japanese Intervention
23. FTX Collapse (crypto contagion)
24. Tech Selloff 2022
25. Bond Rout 2022  

Plus custom scenario builder for hypothetical events.

---

## Final Notes

This document represents the complete vision for the Wild Market Capital portfolio management system. It combines:

1. **Professional-grade analytics**: All the metrics, ratios, and analyses that portfolio managers need
2. **Modern UI/UX**: Clean, data-dense interface inspired by TradingView
3. **Comprehensive functionality**: From portfolio creation to optimization to risk management
4. **Flexibility**: Multiple methods for every task (creation, import, optimization)
5. **Depth**: Not just surface-level metrics, but deep analytical capabilities

**Key Differentiators**:
- 70+ analytics metrics (more comprehensive than most platforms)
- 17 optimization methods (institutional-grade variety)
- 25+ stress scenarios (thorough risk assessment)
- Full scenario chain builder (unique capability)
- Complete attribution analysis (professional requirement)
- Extensive factor analysis (Fama-French and beyond)

**Design Philosophy**:
- Backend-first: Clean separation of calculation engine from UI
- Extensible: Easy to add new metrics, methods, or features  
- Professional: No compromises on analytical rigor
- Practical: Workflows match real portfolio management tasks

This vision document should serve as the complete reference for what the system should do and how it should work. The next step is to create technical specifications that define HOW to implement each component.

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-29  
**Status**: Complete Vision - Ready for Technical Specification Phase

