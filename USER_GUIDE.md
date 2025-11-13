# Wild Market Capital - Complete User Guide

**Comprehensive guide to all features, pages, and capabilities of the Portfolio Management Terminal.**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Application Pages](#application-pages)
   - [1. Dashboard](#1-dashboard)
   - [2. Create Portfolio](#2-create-portfolio)
   - [3. Portfolio List](#3-portfolio-list)
   - [4. Portfolio Analysis](#4-portfolio-analysis)
   - [5. Portfolio Optimization](#5-portfolio-optimization)
   - [6. Risk Analysis](#6-risk-analysis)
   - [7. Forecasting](#7-forecasting)
4. [Complete Features Reference](#complete-features-reference)
5. [Workflow Examples](#workflow-examples)
6. [Tips & Best Practices](#tips--best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Keyboard Shortcuts & UI Tips](#keyboard-shortcuts--ui-tips)

---

## Introduction

**Wild Market Capital** is a professional portfolio management terminal that provides comprehensive analytics, optimization, and risk management for investment portfolios. This guide covers every feature, page, and capability of the application.

### What This Application Does

- **Portfolio Management**: Create, manage, and analyze multiple portfolios
- **Advanced Analytics**: Calculate 70+ financial metrics across 4 categories
- **Optimization**: Optimize portfolio weights using 17 different methods
- **Risk Analysis**: VaR, Monte Carlo simulations, stress testing
- **Forecasting**: Predict future returns using multiple forecasting models
- **Visualizations**: Interactive charts and graphs for all analyses

### Who Should Use This Guide

- **New Users**: Start with "Getting Started" and "Application Pages"
- **Regular Users**: Reference specific pages for detailed feature descriptions
- **Advanced Users**: Check "Complete Features Reference" for all capabilities
- **Developers**: See [Architecture Documentation](ARCHITECTURE.md) for technical details

---

## Getting Started

### First Launch

1. **Start the Application**:
   ```bash
   python run.py
   ```
   The application opens automatically in your default browser at `http://localhost:8501`

2. **Check System Status**:
   - Look at the sidebar "System Status" section
   - Ensure API status shows "Online" (green)
   - Response time should be <100ms
   - If offline, check your internet connection and click "Refresh Status"

3. **Understand Navigation**:
   - **Sidebar**: Main navigation menu (always visible)
   - **Top Bar**: Page title and context
   - **Main Area**: Page content and features

### Creating Your First Portfolio

**Recommended First Steps**:

1. **Start with Dashboard**:
   - View market conditions
   - Check major indices (S&P 500, NASDAQ, etc.)
   - Familiarize yourself with navigation

2. **Create a Test Portfolio**:
   - Click "Create Portfolio" in sidebar
   - Use "Text Input" method (fastest)
   - Enter: `AAPL 40%, MSFT 30%, GOOGL 30%`
   - Follow the wizard to completion

3. **View Your Portfolio**:
   - Go to "Portfolio List"
   - Click on your portfolio
   - Explore the detail view

4. **Run Analysis**:
   - Navigate to "Portfolio Analysis"
   - Select your portfolio
   - Choose date range (default: 1 year)
   - Explore all tabs and metrics

---

## Application Pages

### 1. Dashboard

**Location**: Sidebar → "Dashboard"  
**Purpose**: Market overview and quick navigation hub

#### Features

##### Quick Navigation Section
**6 Navigation Buttons**:
- **Portfolio List**: View all your portfolios
- **Create Portfolio**: Start creating a new portfolio
- **Analysis**: Go to Portfolio Analysis page
- **Optimization**: Access Portfolio Optimization
- **Risk Analysis**: Open Risk Analysis tools
- **Forecasting**: Navigate to Forecasting page

**Use Case**: Quick access to all major features without using sidebar

##### Market Indices Section
**Real-Time Index Data**:

Displays 4 major market indices with live data:

1. **S&P 500 (^GSPC)**
   - Current price
   - Daily change (absolute and percentage)
   - Color-coded (green for positive, red for negative)

2. **NASDAQ (^NDX)**
   - Technology-focused index
   - Real-time updates

3. **Dow Jones (^DJI)**
   - 30 large-cap stocks
   - Industrial average

4. **Russell 2000 (^RUT)**
   - Small-cap index
   - Broader market indicator

**Data Updates**: 
- Refreshed on page load
- 5-minute cache for current prices
- Historical data cached for 24 hours

##### Indices Comparison Chart
**Interactive Plotly Chart**:
- **Time Period**: Always 1 year (365 days)
- **Display**: Cumulative returns for all 4 indices
- **Features**:
  - Hover for exact values
  - Zoom with mouse wheel
  - Pan by dragging
  - Double-click to reset zoom
  - Legend click to show/hide indices

**Interpretation**:
- Compare relative performance
- Identify market trends
- See which indices outperformed

##### Market Statistics Section
**Key Market Metrics**:
- Market volatility indicators
- Trading volume statistics
- Market breadth metrics
- Sector performance summaries

**Use Cases**:
- Assess overall market conditions before portfolio decisions
- Identify market trends
- Quick access to all application features
- Monitor major indices performance

---

### 2. Create Portfolio

**Location**: Sidebar → "Create Portfolio"  
**Purpose**: Create new portfolios using 5 different methods

#### Overview

This page provides **5 distinct methods** for creating portfolios, each optimized for different use cases:

1. **Wizard** - Step-by-step guided process (best for beginners)
2. **Text Input** - Fast natural language entry (best for quick creation)
3. **CSV Import** - Upload from spreadsheets (best for existing portfolios)
4. **Manual Entry** - Add positions one by one (best for precision)
5. **Templates** - Pre-built strategies (best for quick start)

#### Method 1: Wizard (Step-by-Step)

**Best For**: First-time users, guided creation, learning the system

**Process**: 5 sequential steps with validation at each stage

##### Step 1: Portfolio Information

**Required Fields**:
- **Portfolio Name**: 
  - Must be unique
  - 1-100 characters
  - Cannot be empty
  - Examples: "My Tech Portfolio", "Retirement Fund 2025"

- **Description** (Optional):
  - Free-form text
  - Can include notes, strategy, goals
  - Examples: "Technology-focused growth portfolio", "Conservative income strategy"

- **Currency**:
  - Default: USD
  - Options: USD, EUR, GBP, JPY, etc.
  - Used for display formatting

- **Starting Capital**:
  - Initial investment amount
  - Must be > 0
  - Used for value calculations
  - Examples: 100000 (for $100,000)

**Validation**:
- Name uniqueness checked
- All required fields validated
- Error messages shown inline

##### Step 2: Choose Input Method

**Selection Options**:

1. **Text Input**:
   - Fastest method
   - Supports multiple formats
   - Good for 5-20 assets
   - Preview: `AAPL 30%, MSFT 25%, GOOGL 20%`

2. **Upload File**:
   - From spreadsheet
   - CSV or Excel files
   - Perfect for existing data
   - Preview: Column mapping interface

3. **Manual Entry**:
   - Full control
   - Add one by one
   - Real-time validation
   - Preview: Dynamic form interface

4. **Use Template**:
   - Pre-built strategies
   - Factor-based portfolios
   - Classic allocations
   - Preview: Template selection grid

**Help Section**: Expandable guide explaining when to use each method

##### Step 3: Add Assets

**Content varies by selected method** (see detailed method descriptions below)

**Common Features**:
- Real-time validation
- Preview table
- Weight calculation
- Ticker validation
- Error highlighting

##### Step 4: Settings & Review

**Cash Management Options**:
- **None**: No cash position
- **Fixed Amount**: Specify exact cash amount
- **Percentage**: Cash as % of portfolio
- **Remainder**: All remaining after asset allocation

**Review Section**:
- Complete portfolio summary
- Position table with:
  - Ticker symbols
  - Shares/weights
  - Current prices (if available)
  - Values
  - Percentages
- Total weight verification
- Validation status

**Actions**:
- Edit any step (go back)
- Proceed to creation
- Cancel (start over)

##### Step 5: Create

**Final Confirmation**:
- Portfolio summary
- Creation progress bar
- Status messages

**On Success**:
- Success message
- Portfolio created confirmation
- Option to:
  - View portfolio
  - Create another
  - Go to dashboard

**On Error**:
- Error message with details
- Recovery options:
  - Try Again (go back to Step 4)
  - Start Over (reset wizard)

#### Method 2: Text Input

**Best For**: Quick entry, simple portfolios, experienced users

**How It Works**:
1. Enter tickers and weights in text area
2. System parses automatically
3. Validates tickers
4. Shows preview
5. Normalizes weights

##### Supported Formats

**Format 1: Colon with Percentages**
```
AAPL:40%, MSFT:30%, GOOGL:30%
```
- Most explicit
- Clear percentage notation
- Easy to read

**Format 2: Space with Decimals**
```
AAPL 0.4, MSFT 0.3, GOOGL 0.3
```
- Decimal notation (0.0 to 1.0)
- 0.4 = 40%
- Common in programming contexts

**Format 3: Numbers > 1 (Auto-Percentage)**
```
AAPL 40, MSFT 30, GOOGL 30
```
- Numbers treated as percentages
- 40 = 40%
- Most intuitive for many users

**Format 4: Equal Weights**
```
AAPL, MSFT, GOOGL
```
- No weights specified
- Automatically equal allocation
- Each asset gets 1/N weight

**Mixed Formats** (also supported):
```
AAPL:40%, MSFT 0.3, GOOGL 30, AMZN, TSLA 10%
```
- Can mix formats in same input
- System handles automatically

##### Features

**Real-Time Parsing**:
- Parses as you type
- Shows parse results immediately
- Highlights errors

**Ticker Validation**:
- Validates against market data
- Shows invalid tickers
- Suggests corrections (if available)
- Filters out invalid entries

**Weight Normalization**:
- Automatically normalizes to 100%
- Shows original vs normalized
- Preserves relative proportions

**Preview Table**:
- Ticker column
- Weight column (formatted as %)
- Status column (Valid/Invalid)
- Sortable and filterable

**Metrics Display**:
- Number of assets
- Total weight (before normalization)
- Status (Perfect/Will normalize)

**Example Workflow**:
1. Type: `AAPL 40, MSFT 30, GOOGL 30`
2. See: "Parsed 3 assets successfully"
3. See preview table with all assets
4. See: "Total Weight: 100.0%" (Perfect)
5. Click "Next Step →"

**Error Handling**:
- Invalid tickers highlighted
- Parse errors shown clearly
- Suggestions provided when possible

#### Method 3: CSV/Excel Import

**Best For**: Importing existing portfolios from brokers, spreadsheets, other tools

##### Supported File Formats

- **CSV** (.csv): Comma-separated values
- **Excel** (.xlsx, .xls): Microsoft Excel files

##### Required Columns

**Minimum Required**:
- `ticker` or `symbol`: Ticker symbols (e.g., AAPL, MSFT)

**Weight/Shares** (one required):
- `weight`: Portfolio weights (0.0 to 1.0 or percentages)
- `shares`: Number of shares
- `quantity`: Alternative to shares

**Optional Columns**:
- `name`: Company name
- `sector`: Sector classification
- `industry`: Industry classification
- `price`: Current price (if known)

##### Column Mapping

**Automatic Detection**:
- System tries to auto-detect columns
- Matches common column names
- Handles variations (ticker/Ticker/TICKER)

**Manual Mapping**:
- If auto-detection fails
- Dropdown for each required column
- Preview of mapped data

##### Import Process

1. **Upload File**:
   - Click "Browse Files" or drag & drop
   - Select CSV or Excel file
   - File validated on upload

2. **Column Mapping**:
   - Review auto-detected mappings
   - Adjust if needed
   - Verify preview

3. **Data Preview**:
   - Shows first 10 rows
   - Highlights mapped columns
   - Shows data types

4. **Validation**:
   - Ticker validation
   - Weight/shares validation
   - Data type checks
   - Error reporting

5. **Error Handling**:
   - Row-level errors shown
   - Invalid rows highlighted
   - Option to skip error rows
   - Download error report (CSV)

##### Error Types

**File-Level Errors**:
- Invalid file format
- File too large
- Corrupted file
- Missing required columns

**Row-Level Errors**:
- Invalid ticker symbol
- Negative weights
- Missing required data
- Data type mismatches

**Column-Level Errors**:
- Unmapped required column
- Ambiguous column names
- Multiple matches

##### Example CSV Format

```csv
ticker,weight,name
AAPL,0.40,Apple Inc.
MSFT,0.30,Microsoft Corporation
GOOGL,0.30,Alphabet Inc.
```

Or with shares:
```csv
ticker,shares,price
AAPL,100,150.00
MSFT,50,300.00
GOOGL,30,2500.00
```

##### Features

- **Automatic Weight Calculation**: If shares provided, calculates weights
- **Price Fetching**: Fetches current prices if not provided
- **Data Cleaning**: Handles extra whitespace, case variations
- **Error Recovery**: Continue with valid rows, skip invalid ones
- **Export Error Report**: Download CSV with all errors for fixing

#### Method 4: Manual Entry

**Best For**: Full control, complex portfolios, precise allocation

##### Interface

**Dynamic Form**:
- Add positions one by one
- Each position has:
  - Ticker input (with autocomplete)
  - Shares input
  - Weight target (optional)
  - Remove button

**Real-Time Updates**:
- Weight calculation as you type
- Total weight display
- Remaining weight shown
- Validation feedback

##### Features

**Ticker Input**:
- Autocomplete suggestions
- Real-time validation
- Company name display
- Price fetching

**Shares Input**:
- Number input
- Decimal support (for fractional shares)
- Validation (must be > 0)

**Weight Target** (Optional):
- Set target weight
- System calculates required shares
- Based on current price

**Position Management**:
- Add new position (button)
- Remove position (X button)
- Reorder positions (drag & drop, if implemented)
- Edit existing positions

**Portfolio Summary**:
- Total positions count
- Total weight
- Remaining weight
- Average position size

**Validation**:
- Duplicate ticker detection
- Weight sum validation
- Ticker existence check
- Price availability check

##### Workflow Example

1. Click "Add Position"
2. Enter ticker: "AAPL"
   - See: "Apple Inc." (autocomplete)
   - See: Current price: $150.00
3. Enter shares: 100
   - See: Value: $15,000
   - See: Weight: 40% (if starting capital = $37,500)
4. Click "Add Position" again
5. Enter ticker: "MSFT", shares: 50
6. Continue until portfolio complete
7. Review summary
8. Proceed to next step

#### Method 5: Templates

**Best For**: Quick start, proven strategies, beginners

##### Available Templates

**Factor-Based Portfolios**:

1. **Value Factor**:
   - Focus: Undervalued stocks
   - Characteristics: Low P/E, high dividend yield
   - Typical allocation: 10-15 positions
   - Example tickers: BRK.B, JNJ, PG

2. **Quality Factor**:
   - Focus: High-quality companies
   - Characteristics: Strong balance sheets, consistent earnings
   - Typical allocation: 10-15 positions
   - Example tickers: AAPL, MSFT, GOOGL

3. **Growth Factor**:
   - Focus: High-growth companies
   - Characteristics: High revenue growth, innovation
   - Typical allocation: 10-15 positions
   - Example tickers: TSLA, NVDA, AMZN

**Classic Strategies**:

4. **60/40 Balanced**:
   - 60% stocks, 40% bonds
   - Classic balanced allocation
   - Lower risk, steady returns
   - Example: 60% SPY, 40% AGG

5. **All Weather**:
   - Diversified across asset classes
   - Weathers all market conditions
   - Balanced risk exposure

**Sector-Focused**:

6. **Tech Focus**:
   - Technology sector concentration
   - High growth potential
   - Higher volatility
   - Example tickers: AAPL, MSFT, GOOGL, NVDA, META

7. **Dividend Portfolio**:
   - Dividend-paying stocks
   - Income-focused
   - Lower growth, steady income
   - Example tickers: JNJ, PG, KO, T

**ESG/Sustainable**:

8. **ESG Portfolio**:
   - Environmentally responsible
   - Socially conscious
   - Governance-focused
   - Sustainable investing

**Sector Diversified**:

9. **Sector Diversified**:
   - Equal weight across sectors
   - Broad market exposure
   - Reduced sector risk

##### Template Features

**Pre-Configured Allocations**:
- Weights already set
- Based on strategy
- Can be customized

**Customization Options**:
- Adjust individual weights
- Add/remove positions
- Modify allocation percentages
- Change starting capital

**Template Information**:
- Strategy description
- Risk level
- Expected characteristics
- Use case recommendations

##### Using Templates

1. **Select Template**:
   - Browse available templates
   - Read descriptions
   - Choose based on goals

2. **Review Allocation**:
   - See pre-configured positions
   - Review weights
   - Understand strategy

3. **Customize** (Optional):
   - Adjust weights
   - Add positions
   - Remove positions
   - Modify allocation

4. **Proceed**:
   - Continue to Settings & Review
   - Complete portfolio creation

---

### 3. Portfolio List

**Location**: Sidebar → "Portfolio List"  
**Purpose**: Manage all portfolios - view, edit, delete, clone, bulk operations

#### Main View Modes

##### List View (Default)

**Portfolio Cards Display**:
Each portfolio shown as a card with:

- **Portfolio Name**: Large, prominent
- **Description**: Subtitle (if provided)
- **Key Metrics**:
  - Number of positions
  - Total value (if prices available)
  - Creation date
  - Last modified date

- **Quick Actions**:
  - View button
  - Edit button
  - Delete button (with confirmation)

**Layout Options**:
- Grid view (default)
- List view (compact)
- Toggle between views

**Search & Filter**:
- **Search Bar**: 
  - Search by name
  - Real-time filtering
  - Case-insensitive

- **Filters**:
  - Date range (created between)
  - Minimum positions
  - Maximum positions
  - Sort options:
    - Name (A-Z, Z-A)
    - Date (newest first, oldest first)
    - Value (highest first, lowest first)

**Bulk Operations**:
- **Select Multiple**: Checkboxes on each card
- **Bulk Delete**: Delete selected portfolios
- **Bulk Export**: Export multiple to CSV
- **Select All / Deselect All**: Quick selection

##### Detail View

**Access**: Click "View" button on any portfolio card

**Sections**:

1. **Portfolio Information**:
   - Name, description
   - Starting capital
   - Currency
   - Creation date
   - Last modified

2. **Position Table**:
   - All positions listed
   - Columns:
     - Ticker
     - Company Name (if available)
     - Shares
     - Current Price
     - Current Value
     - Weight (%)
     - Target Weight (if set)
   - Sortable columns
   - Searchable

3. **Visualizations**:
   - **Asset Allocation Chart** (Pie/Donut):
     - Visual weight distribution
     - Hover for details
     - Click to highlight

   - **Sector Allocation Chart** (if sector data available):
     - Grouped by sector
     - Sector percentages
     - Color-coded

4. **Portfolio Summary Metrics**:
   - Total positions
   - Total value
   - Cash position (if any)
   - Diversification metrics

#### Edit Portfolio

**Access**: Click "Edit" button

**Editable Fields**:
- **Name**: Change portfolio name (must be unique)
- **Description**: Update description
- **Starting Capital**: Modify initial capital
- **Currency**: Change currency (with conversion warning)

**Position Management**:
- **Add Position**:
  - Ticker input
  - Shares input
  - Weight target (optional)
  - Real-time validation

- **Remove Position**:
  - Click remove button
  - Confirmation dialog
  - Immediate update

- **Update Position**:
  - Edit shares
  - Edit weight target
  - Update ticker (with validation)

**Save Changes**:
- "Save" button
- Validation before save
- Success/error messages
- Auto-refresh after save

#### Clone Portfolio

**Access**: Click "Clone" button (or from detail view)

**Process**:
1. Enter new portfolio name
2. Option to modify:
   - Description
   - Starting capital
   - Positions (add/remove)
3. Preview cloned portfolio
4. Confirm creation

**Use Cases**:
- Create variations of existing portfolio
- Test different allocations
- Backup before major changes
- Create similar portfolios with modifications

#### Delete Portfolio

**Access**: Click "Delete" button

**Safety Features**:
- Confirmation dialog
- Portfolio name must be typed to confirm
- Undo option (within 5 seconds)
- Permanent deletion warning

**Bulk Delete**:
- Select multiple portfolios
- "Delete Selected" button
- Batch confirmation
- Progress indicator

#### Export Options

**Single Portfolio Export**:
- **CSV Format**:
  - Portfolio information
  - All positions
  - Current values
  - Weights

- **Excel Format**:
  - Multiple sheets
  - Formatted tables
  - Charts (if supported)

**Bulk Export**:
- Export all selected portfolios
- Combined CSV/Excel file
- Separate sheets per portfolio

---

### 4. Portfolio Analysis

**Location**: Sidebar → "Portfolio Analysis"  
**Purpose**: Comprehensive analytics with 70+ metrics and interactive charts

**This is the most feature-rich page** with 5 main tabs and multiple sub-tabs.

#### Analysis Parameters Section

**Date Range Selection**:
- **Start Date**: 
  - Date picker
  - Default: 1 year ago
  - Maximum: Today
  - Validation: Must be before end date

- **End Date**:
  - Date picker
  - Default: Today
  - Minimum: Start date
  - Maximum: Today

**Portfolio Selection**:
- Dropdown with all portfolios
- Shows portfolio name
- Auto-loads when selected

**Comparison Options**:
- **None**: No comparison
- **Benchmark Ticker**: 
  - Common benchmarks: SPY, QQQ, VTI, DIA, IWM
  - Custom ticker input
  - Real-time validation

- **Another Portfolio**:
  - Select from portfolio list
  - Compare performance side-by-side

**Calculate Button**:
- Triggers metric calculation
- Shows loading spinner
- Progress indicator
- Error handling

#### Tab 1: Overview

**Purpose**: High-level summary with key metrics and charts

##### Key Metrics Dashboard

**Performance Metrics Section**:

**Returns**:
- **Total Return**: 
  - Absolute return over period
  - Formula: (End Value / Start Value) - 1
  - Display: Percentage with color coding

- **Annualized Return**:
  - Return per year (compounded)
  - Accounts for time period
  - More comparable across periods

- **CAGR** (Compound Annual Growth Rate):
  - Geometric mean return
  - Best for long-term analysis
  - Accounts for compounding

**Period Returns**:
- **YTD** (Year-to-Date): Since January 1
- **MTD** (Month-to-Date): Since month start
- **QTD** (Quarter-to-Date): Since quarter start
- **1M, 3M, 6M**: Last 1/3/6 months
- **1Y, 3Y, 5Y**: Last 1/3/5 years

**Best/Worst Periods**:
- Best Month: Highest monthly return
- Worst Month: Lowest monthly return
- Best Quarter: Highest quarterly return
- Worst Quarter: Lowest quarterly return

**Win Rate**:
- Percentage of positive periods
- Daily, weekly, monthly win rates
- Shows consistency

**Risk Metrics Section**:

**Volatility**:
- **Daily Volatility**: Standard deviation of daily returns
- **Weekly Volatility**: Standard deviation of weekly returns
- **Monthly Volatility**: Standard deviation of monthly returns
- **Annual Volatility**: Annualized volatility (daily * √252)

**Drawdown Metrics**:
- **Max Drawdown**: 
  - Largest peak-to-trough decline
  - Value and percentage
  - Start and end dates
  - Duration in days

- **Current Drawdown**: 
  - Current decline from peak
  - Real-time calculation

- **Average Drawdown**: 
  - Mean of all drawdowns
  - Shows typical decline magnitude

**VaR (Value at Risk)**:
- **90% VaR**: Maximum loss at 90% confidence
- **95% VaR**: Maximum loss at 95% confidence
- **99% VaR**: Maximum loss at 99% confidence
- Methods: Historical, Parametric, Cornish-Fisher

**CVaR (Conditional VaR)**:
- **90% CVaR**: Expected loss beyond 90% VaR
- **95% CVaR**: Expected loss beyond 95% VaR
- **99% CVaR**: Expected loss beyond 99% VaR
- Also called "Expected Shortfall"

**Risk-Adjusted Ratios Section**:

**Sharpe Ratio**:
- Risk-adjusted return measure
- Formula: (Return - Risk-free) / Volatility
- Higher is better
- Standard: >1 is good, >2 is excellent

**Sortino Ratio**:
- Like Sharpe, but uses downside deviation
- Only penalizes negative volatility
- Better for asymmetric returns

**Calmar Ratio**:
- Return / Max Drawdown
- Measures return per unit of drawdown risk
- Higher is better

**Information Ratio**:
- Active return / Tracking error
- Measures skill vs benchmark
- Positive = outperformance

**Market Metrics Section**:

**Beta**:
- Sensitivity to market
- 1.0 = moves with market
- >1.0 = more volatile than market
- <1.0 = less volatile than market

**Alpha (CAPM)**:
- Excess return vs expected (from Beta)
- Positive = outperformance
- Negative = underperformance
- Annualized percentage

**R² (R-squared)**:
- Correlation with benchmark squared
- 0-1 scale
- 1.0 = perfect correlation
- 0.0 = no correlation

**Correlation**:
- Correlation coefficient with benchmark
- -1 to +1 scale
- Positive = moves together
- Negative = moves opposite

**Tracking Error**:
- Standard deviation of active returns
- Measures deviation from benchmark
- Lower = more closely tracks benchmark

**Up/Down Capture**:
- **Up Capture**: Performance in up markets
- **Down Capture**: Performance in down markets
- >100% = amplifies market moves
- <100% = dampens market moves

##### Charts in Overview Tab

**Cumulative Returns Chart**:
- Portfolio vs Benchmark
- Time series line chart
- Interactive (zoom, pan, hover)
- Shows outperformance/underperformance periods

**Drawdown Chart**:
- Underwater plot
- Shows all drawdown periods
- Depth and duration visible
- Color-coded by severity

**Key Metrics Summary Table**:
- All major metrics in one table
- Portfolio vs Benchmark comparison
- Color-coded differences
- Sortable columns

##### Comparison Table

**Portfolio vs Benchmark**:
- Side-by-side comparison
- All key metrics
- Difference calculation
- Visual indicators (↑ ↓ =)

**Interpretation**:
- Automatic insights
- Outperformance indicators
- Risk assessment
- Recommendations

#### Tab 2: Performance

**Purpose**: Detailed performance analysis with 3 sub-tabs

##### Sub-tab 2.1: Returns Analysis

**Cumulative Returns Chart**:
- Full period cumulative returns
- Portfolio and benchmark
- Interactive features
- Period selection

**Daily Active Returns (Area Chart)**:
- Portfolio return minus benchmark return
- Area chart (positive/negative)
- Shows daily alpha
- Statistics:
  - Average daily active return
  - Positive days count
  - Negative days count
  - Max daily alpha
  - Min daily alpha

**Return by Periods Table**:
- Multiple timeframes:
  - 1 Month
  - 3 Months
  - 6 Months
  - 1 Year
  - 3 Years
  - 5 Years
  - Since Inception
- Portfolio vs Benchmark
- Difference column
- Bar chart visualization

**Expected Returns Table**:
- Mean historical returns
- Multiple timeframes:
  - Daily
  - Weekly
  - Monthly
  - Quarterly
  - Yearly
- Portfolio vs Benchmark
- Based on arithmetic mean

**Common Performance Periods (CPP)**:
- Same direction percentage
- CPP Index (correlation of moves)
- Interpretation of correlation level

**Best/Worst Periods**:
- Best 3-month periods (top 3)
- Worst 3-month periods (top 3)
- Dates and returns
- Comparison with benchmark

##### Sub-tab 2.2: Periodic Analysis

**Monthly Returns Heatmap**:
- Calendar heatmap
- Each cell = one month
- Color-coded by return
- Years as rows, months as columns
- Hover for exact values

**Yearly Returns**:
- Bar chart by year
- Portfolio vs Benchmark
- Annual performance comparison
- Trend analysis

**Quarterly Returns**:
- Bar chart by quarter
- Q1, Q2, Q3, Q4 for each year
- Seasonal patterns visible

**Period Comparison Table**:
- All periods in one table
- Sortable
- Filterable
- Exportable

##### Sub-tab 2.3: Distribution

**Returns Distribution Histogram**:
- Frequency distribution
- Normal distribution overlay
- Statistics:
  - Mean
  - Median
  - Standard deviation
  - Skewness
  - Kurtosis

**Q-Q Plot** (Quantile-Quantile):
- Tests normality
- Portfolio vs Normal distribution
- Deviations indicate non-normality
- Interpretation provided

**Statistical Tests**:
- Normality tests (Shapiro-Wilk, etc.)
- Skewness test
- Kurtosis test
- Results and p-values

**Outlier Analysis**:
- Scatter plot of returns
- Outliers highlighted
- Extreme values identified
- Impact analysis

#### Tab 3: Risk

**Purpose**: Comprehensive risk analysis with 4 sub-tabs

##### Sub-tab 3.1: Key Risk Metrics

**Volatility Metrics**:
- Daily, Weekly, Monthly, Annual
- Portfolio vs Benchmark
- Comparison table
- Trend analysis

**Drawdown Metrics**:
- Max Drawdown (value, %, dates, duration)
- Current Drawdown
- Average Drawdown
- Number of drawdowns
- Recovery time statistics

**VaR Summary**:
- 90%, 95%, 99% VaR
- Multiple methods:
  - Historical
  - Parametric
  - Cornish-Fisher
- Comparison table

**CVaR Summary**:
- 90%, 95%, 99% CVaR
- Expected shortfall
- Tail risk measure

**Complete Risk Metrics Table**:
- All 22 risk metrics
- Sortable, filterable
- Exportable
- Detailed values

##### Sub-tab 3.2: Drawdown Analysis

**Underwater Plot**:
- Full drawdown visualization
- All drawdown periods
- Depth and duration
- Recovery paths

**Drawdown Periods Table**:
- All drawdowns listed
- Start date, end date
- Depth (value and %)
- Duration (days)
- Recovery time

**Drawdown Recovery Analysis**:
- Recovery time statistics
- Average recovery
- Longest recovery
- Recovery patterns

**Drawdown Statistics**:
- Number of drawdowns
- Average drawdown
- Max drawdown
- Drawdown frequency

##### Sub-tab 3.3: VaR & CVaR

**VaR Distribution Chart**:
- Histogram of returns
- VaR levels marked
- Confidence intervals
- Visual risk assessment

**VaR Methods Comparison**:
- Historical VaR
- Parametric VaR
- Cornish-Fisher VaR
- Monte Carlo VaR
- Comparison table
- Method recommendations

**CVaR Analysis**:
- Expected shortfall calculation
- Tail risk visualization
- Beyond-VaR analysis

**Rolling VaR**:
- VaR over time
- Time series chart
- Risk evolution
- Trend analysis

##### Sub-tab 3.4: Rolling Risk Metrics

**Rolling Volatility**:
- 30-day, 60-day, 90-day windows
- Time series
- Volatility trends
- Comparison with benchmark

**Rolling Sharpe Ratio**:
- Risk-adjusted return over time
- Performance consistency
- Trend analysis

**Rolling Sortino Ratio**:
- Downside risk-adjusted return
- Time series
- Consistency measure

**Rolling Beta**:
- Market sensitivity over time
- Time series
- Stability assessment

#### Tab 4: Assets & Correlations

**Purpose**: Individual asset analysis and correlation studies

##### Sub-tab 4.1: Asset Overview & Impact

**Asset Metrics Table**:
- Individual asset performance
- Returns, volatility
- Contribution to portfolio return
- Contribution to portfolio risk
- Weight vs Risk comparison

**Asset Impact on Returns**:
- Bar chart
- Each asset's contribution
- Positive/negative contributions
- Relative importance

**Asset Impact on Risk**:
- Bar chart
- Risk contribution
- Diversification effect
- Concentration risk

**Risk vs Weight Comparison**:
- Scatter plot
- Weight (X-axis) vs Risk (Y-axis)
- Identifies over/under-weighted positions
- Optimization opportunities

##### Sub-tab 4.2: Correlation Analysis

**Correlation Matrix Heatmap**:
- All assets vs all assets
- Color-coded (-1 to +1)
- Interactive (hover for values)
- Clustered version available

**Correlation with Benchmark**:
- Each asset vs benchmark
- Bar chart
- High/low correlation identification
- Diversification assessment

**Correlation Statistics**:
- Average correlation
- Min/Max correlation
- Correlation distribution
- Diversification metrics

**Clustered Correlation Matrix**:
- Hierarchical clustering
- Grouped by correlation
- Dendrogram visualization
- Asset grouping insights

##### Sub-tab 4.3: Asset Dynamics

**Asset Price Charts**:
- Individual price charts
- All assets in portfolio
- Time series
- Interactive

**Volume Analysis**:
- Trading volume charts
- Volume trends
- Price-volume correlation

**Price vs Volume**:
- Scatter plots
- Relationship analysis
- Anomaly detection

**Detailed Asset Analysis**:
- Comprehensive asset view
- Price, volume, returns
- Technical indicators
- Performance metrics

#### Tab 5: Export & Reports

**Purpose**: Export data and generate reports

##### CSV Export

**Export Options**:
- Portfolio data
- Metrics (all or selected)
- Returns data
- Positions data

**Format**:
- Standard CSV
- Excel-compatible
- UTF-8 encoding

##### Excel Export

**Multi-Sheet Workbook**:
- **Summary Sheet**: Overview metrics
- **Holdings Sheet**: All positions
- **Metrics Sheet**: All calculated metrics
- **Returns Sheet**: Daily returns
- **History Sheet**: Historical data

**Features**:
- Formatted tables
- Charts (if supported)
- Auto-sized columns
- Professional layout

##### PDF Report Generation

**Screenshot-Based PDF**:
- Captures actual Streamlit pages
- Full page screenshots
- High quality
- Dark theme preserved

**Report Sections** (selectable):
- Overview
- Performance Analysis
- Risk Analysis
- Assets & Correlations

**Generation Process**:
1. Select tabs to include
2. Enter report name
3. Click "Generate PDF"
4. System:
   - Opens browser
   - Navigates through tabs
   - Takes screenshots
   - Combines into PDF
5. Download PDF

**Features**:
- Real visual representation
- Includes all charts
- Full page capture
- Automatic tab switching

**Requirements**:
- Streamlit app running
- Default URL: http://localhost:8501
- Browser automation (Playwright)

---

### 5. Portfolio Optimization

**Location**: Sidebar → "Portfolio Optimization"  
**Purpose**: Optimize portfolio weights using 17 different methods

#### Page Overview

This page allows you to:
- Select optimization method
- Configure constraints
- Generate optimized weights
- Compare before/after
- Generate efficient frontier
- Run sensitivity analysis

#### Portfolio Selection

**Current Portfolio Display**:
- Portfolio name
- Number of positions
- Current weights table:
  - Ticker
  - Shares
  - Current Price
  - Current Value
  - Current Weight

**Expandable Section**: Click to view full current portfolio details

#### Optimization Parameters

##### Date Range

**Purpose**: Historical period for optimization calculations

- **Start Date**: 
  - Default: 1 year ago
  - Used for historical data
  - Affects return/covariance estimates

- **End Date**:
  - Default: Today
  - Must be after start date
  - Latest data point

**Recommendations**:
- **Short-term**: 3-6 months (for active strategies)
- **Medium-term**: 1 year (most common)
- **Long-term**: 3-5 years (for stable allocations)

##### Out-of-Sample Testing

**Purpose**: Validate optimization on unseen data

**How It Works**:
- **Training Period**: Before specified dates (for optimization)
- **Validation Period**: Specified dates (for testing)
- Prevents overfitting
- Shows real-world performance

**Training Window Options**:
- **30% (Recommended)**: 
  - Balance between freshness and reliability
  - Suitable for most cases
  - Description provided

- **50%**: 
  - More data for training
  - Suitable for stable markets
  - Long-term strategies

- **60%**: 
  - Maximum statistical reliability
  - Stable assets
  - Conservative portfolios

**Period Display**:
- Shows training period dates
- Shows validation period dates
- Day counts for each

##### Benchmark Selection

**Purpose**: Compare optimized portfolio to benchmark

**Options**:
- **None**: No benchmark comparison
- **SPY**: S&P 500 ETF
- **QQQ**: NASDAQ-100 ETF
- **VTI**: Total Stock Market ETF
- **Custom**: Enter any ticker

**Use**: For visualization and comparison metrics

#### Optimization Methods

**17 Available Methods**:

##### 1. Mean-Variance (Markowitz)

**Classic Portfolio Theory**:
- Maximizes risk-adjusted return
- Efficient frontier generation
- Foundation of modern portfolio theory

**Objectives**:
- **Max Sharpe**: Maximum Sharpe ratio
- **Min Variance**: Minimum volatility
- **Max Return**: Maximum expected return

**Parameters**:
- Risk aversion (for max return)
- Expected returns method
- Covariance estimation method

**Best For**: 
- General optimization
- Risk-return trade-off
- Standard portfolio construction

##### 2. Black-Litterman

**Bayesian Approach**:
- Incorporates market views
- Combines market equilibrium with investor views
- More stable than pure Markowitz

**Parameters**:
- Market views (optional)
- Confidence in views
- Risk aversion parameter
- Market capitalization weights

**Best For**:
- Incorporating expert opinions
- More stable optimizations
- Professional portfolio management

##### 3. Risk Parity

**Equal Risk Contribution**:
- Each asset contributes equally to risk
- Not equal weights, but equal risk
- Balanced risk allocation

**Parameters**:
- Risk measure (volatility, CVaR)
- Target risk level

**Best For**:
- Risk-balanced portfolios
- Reducing concentration risk
- Stable allocations

##### 4. Hierarchical Risk Parity (HRP)

**Clustering-Based**:
- No covariance matrix inversion
- More robust to estimation errors
- Uses hierarchical clustering

**Parameters**:
- Linkage method
- Distance metric

**Best For**:
- Many assets (>20)
- Unstable covariance estimates
- Robust optimization

##### 5. CVaR Optimization

**Tail Risk Focus**:
- Optimizes Conditional Value at Risk
- Focuses on extreme losses
- More conservative

**Parameters**:
- Confidence level (90%, 95%, 99%)
- Risk aversion

**Best For**:
- Risk-averse investors
- Tail risk management
- Downside protection

##### 6. Mean-CVaR

**Return vs Tail Risk**:
- Balances return and CVaR
- Efficient frontier in return-CVaR space
- Risk-return trade-off

**Parameters**:
- Confidence level
- Risk aversion parameter

**Best For**:
- Balancing return and tail risk
- Conservative strategies
- Downside protection

##### 7. Robust Optimization

**Uncertainty Sets**:
- Accounts for parameter uncertainty
- Worst-case optimization
- More conservative results

**Parameters**:
- Uncertainty radius (returns)
- Uncertainty radius (covariance)
- Objective (max return, min risk)

**Best For**:
- Parameter uncertainty
- Robust allocations
- Conservative strategies

##### 8. Maximum Diversification

**Diversification Ratio**:
- Maximizes diversification ratio
- Reduces concentration
- Better risk distribution

**Parameters**:
- Diversification measure

**Best For**:
- Reducing concentration
- Improving diversification
- Many assets

##### 9. Minimum Correlation

**Low Correlation Assets**:
- Minimizes average correlation
- Prefers uncorrelated assets
- Better diversification

**Parameters**:
- Correlation threshold

**Best For**:
- Low correlation strategies
- Diversification focus
- Alternative assets

##### 10. Inverse Correlation Weighting

**Anti-Correlation Focus**:
- Weights inversely to correlation
- Prefers negatively correlated assets
- Hedging strategies

**Parameters**:
- Correlation adjustment factor

**Best For**:
- Hedging
- Negative correlation strategies
- Risk reduction

**Additional Methods** (11-17):
- Equal Weight
- Min Variance
- Max Sharpe
- Max Return
- Kelly Criterion
- Min Tracking Error
- Max Alpha
- Market Cap Weighting

#### Constraints Configuration

##### Weight Constraints

**Individual Asset Limits**:
- **Min Weight**: Minimum allocation per asset
  - Default: 0% (allows exclusion)
  - Can set higher (e.g., 5% minimum)
  - Prevents tiny positions

- **Max Weight**: Maximum allocation per asset
  - Default: 100% (no limit)
  - Common: 20-40% maximum
  - Prevents concentration

**Portfolio-Level Limits**:
- **Long-Only**: No shorting (weights ≥ 0)
- **Allow Shorting**: Negative weights allowed
- **Sum to 100%**: Weights must sum to exactly 1.0

##### Group Constraints

**Sector Limits**:
- Min/Max weight per sector
- Example: Tech max 30%, Finance min 10%
- Requires sector data

**Geographic Limits**:
- Min/Max weight per region
- Example: US min 50%, International max 30%
- Requires geographic data

**Custom Groups**:
- Define custom groups
- Set limits per group
- Flexible grouping

##### Turnover Constraints

**Maximum Turnover**:
- Limit changes from current portfolio
- Reduces transaction costs
- Practical constraint

**Transaction Costs**:
- Model transaction costs
- Affects optimization
- More realistic

##### Cardinality Constraints

**Position Limits**:
- **Min Positions**: Minimum number of assets
- **Max Positions**: Maximum number of assets
- Example: Between 10-20 positions

**Use Cases**:
- Limit complexity
- Reduce transaction costs
- Maintain diversification

#### Optimization Results

##### Optimized Weights Table

**Display**:
- Ticker
- Current Weight
- Optimized Weight
- Change (difference)
- Change % (percentage change)
- New Shares (if applicable)

**Sorting**:
- By ticker
- By current weight
- By optimized weight
- By change

##### Trade List

**Required Transactions**:
- Buy/Sell for each asset
- Shares to buy/sell
- Estimated cost
- Transaction type

**Summary**:
- Total buys
- Total sells
- Net cash flow
- Transaction costs

##### Before/After Comparison

**Metrics Comparison**:
- Expected Return
- Volatility
- Sharpe Ratio
- Max Drawdown
- Beta
- Other key metrics

**Visual Comparison**:
- Side-by-side metrics
- Improvement indicators
- Color coding

##### Efficient Frontier

**Generation**:
- Risk-return curve
- Multiple efficient portfolios
- Optimal trade-offs

**Visualization**:
- Scatter plot
- Current portfolio marked
- Optimized portfolio marked
- Efficient frontier line

**Interaction**:
- Click points to see weights
- Select optimal point
- Generate weights for selected point

#### Sensitivity Analysis

**Purpose**: Test robustness of optimization

**Parameters to Vary**:
- Risk aversion
- Expected returns
- Covariance estimates
- Constraints

**Results**:
- Weight stability
- Parameter sensitivity
- Robustness assessment
- Recommendations

---

### 6. Risk Analysis

**Location**: Sidebar → "Risk Analysis"  
**Purpose**: Advanced risk analysis with VaR, Monte Carlo, and stress testing

#### Page Overview

**5 Main Tabs**:
1. VaR Analysis
2. Monte Carlo Simulation
3. Historical Scenarios
4. Custom Scenario
5. Scenario Chain

#### Tab 1: VaR Analysis

**Value at Risk (VaR)**: Maximum expected loss at given confidence level

##### Configuration

**Confidence Level**:
- Slider: 90% to 99%
- Default: 95%
- Common: 90%, 95%, 99%
- Higher = more conservative

**Time Horizon**:
- Days forward
- Range: 1-30 days
- Default: 1 day
- Longer = higher VaR

##### Calculation Methods

**1. Historical VaR**:
- Based on historical returns
- No distribution assumptions
- Most intuitive
- Requires sufficient history

**2. Parametric VaR**:
- Assumes normal distribution
- Uses mean and volatility
- Fast calculation
- May underestimate tail risk

**3. Cornish-Fisher VaR**:
- Adjusts for skewness/kurtosis
- More accurate for non-normal
- Accounts for fat tails
- Better for real markets

**4. Monte Carlo VaR**:
- Simulation-based
- 10,000+ simulations
- Most flexible
- Computationally intensive

##### Results Display

**Key Metrics Cards**:
- VaR (95%): Maximum loss at 95% confidence
- CVaR (95%): Expected loss beyond VaR
- VaR (99%): Maximum loss at 99% confidence
- CVaR (99%): Expected loss beyond VaR

**Methods Comparison Table**:
- All methods side-by-side
- Values in absolute and percentage
- Method recommendations
- Interpretation

**VaR Distribution Chart**:
- Histogram of returns
- VaR levels marked
- Visual risk assessment
- Confidence intervals

**Rolling VaR**:
- VaR over time
- Time series chart
- Risk evolution
- Trend analysis

**VaR Sensitivity Analysis**:
- Parameter variation
- Sensitivity to:
  - Confidence level
  - Time horizon
  - Historical period
- Robustness assessment

**Portfolio VaR vs Component VaR**:
- Total portfolio VaR
- Individual asset contributions
- Risk decomposition
- Concentration analysis

#### Tab 2: Monte Carlo Simulation

**Purpose**: Simulate thousands of possible future scenarios

##### Configuration

**Number of Paths**:
- Default: 10,000
- Range: 1,000 - 100,000
- More paths = more accuracy
- Higher computational cost

**Time Horizon**:
- Days forward
- Default: 30 days
- Range: 1-365 days
- Longer = more uncertainty

**Simulation Method**:
- Geometric Brownian Motion (GBM)
- With/without drift
- Volatility modeling

##### Results

**Distribution of Outcomes**:
- Histogram of final values
- Probability distribution
- Percentiles (5th, 25th, 50th, 75th, 95th)
- Statistics (mean, median, std dev)

**Probability Metrics**:
- Probability of loss
- Probability of gain
- Probability of >X% loss
- Probability of >X% gain

**Confidence Intervals**:
- 90% confidence interval
- 95% confidence interval
- 99% confidence interval
- Range of likely outcomes

**Path Visualization**:
- Sample paths shown
- Representative scenarios
- Visual uncertainty
- Trend patterns

**Analysis**:
- Worst-case scenarios
- Best-case scenarios
- Expected value
- Risk assessment

#### Tab 3: Historical Scenarios

**Purpose**: Test portfolio against past market crises

##### Available Scenarios

**1. 2008 Financial Crisis**:
- Period: Sep 2008 - Mar 2009
- Market crash
- Credit crisis
- Bank failures

**2. 2020 COVID-19 Crash**:
- Period: Feb 2020 - Apr 2020
- Pandemic impact
- Market volatility
- Recovery patterns

**3. 2000 Dot-com Bubble**:
- Period: Mar 2000 - Oct 2002
- Tech crash
- Bubble burst
- Sector-specific

**4. 1987 Black Monday**:
- Period: Oct 1987
- Single-day crash
- Market crash
- Recovery analysis

**5. Custom Historical Period**:
- Select any date range
- Define custom scenario
- Test specific periods

##### Results for Each Scenario

**Portfolio Impact**:
- Total portfolio loss/gain
- Percentage change
- Absolute value change
- Recovery time

**Position-Level Impacts**:
- Each asset's contribution
- Worst hit positions
- Best performing positions
- Sector analysis

**Comparison Across Scenarios**:
- All scenarios in one table
- Relative impact
- Worst scenario identification
- Resilience assessment

**Recovery Analysis**:
- Time to recover
- Recovery path
- Comparison with market
- Lessons learned

#### Tab 4: Custom Scenario

**Purpose**: Create and test custom stress scenarios

##### Scenario Creation

**Scenario Name**:
- Descriptive name
- Examples: "Tech Crash", "Inflation Spike"
- Required field

**Asset Shocks**:
- Define % change for each asset
- Positive or negative
- Examples: AAPL -20%, MSFT -15%
- Can leave some unchanged

**Correlation Changes** (Advanced):
- Modify correlations
- Stress correlation assumptions
- Test correlation breakdown

##### Results

**Portfolio Impact**:
- Total impact percentage
- Total impact value
- Comparison with baseline

**Position Breakdown**:
- Each position's impact
- Worst position (largest loss)
- Best position (smallest loss/gain)
- Detailed breakdown

**Interpretation**:
- Automatic analysis
- Risk assessment
- Recommendations
- Mitigation strategies

#### Tab 5: Scenario Chain

**Purpose**: Test sequential scenarios

##### Chain Configuration

**Multiple Scenarios**:
- Define sequence
- Example: Crisis → Recovery → Growth
- Cumulative impact
- Recovery paths

##### Results

**Cumulative Impact**:
- Total impact across all scenarios
- Step-by-step breakdown
- Recovery analysis

**Path Visualization**:
- Portfolio value over time
- Scenario transitions
- Recovery periods
- Trend analysis

---

### 7. Forecasting

**Location**: Sidebar → "Forecasting"  
**Purpose**: Forecast future prices and returns using multiple methods

#### Page Overview

**Two Forecast Types**:
1. **Single Asset**: Forecast individual ticker
2. **Portfolio**: Forecast entire portfolio

#### Asset/Portfolio Selection

##### Single Asset Forecasting

**Ticker Selection**:
- Dropdown of all tickers in portfolios
- Or enter custom ticker
- Real-time validation

**Use Cases**:
- Individual stock analysis
- Asset-specific forecasts
- Component analysis

##### Portfolio Forecasting

**Portfolio Selection**:
- Choose from portfolio list
- Forecasts aggregate portfolio
- Weighted forecast

**Use Cases**:
- Overall portfolio outlook
- Strategic planning
- Risk assessment

#### Forecasting Parameters

##### Training Period

**Start Date**:
- Historical data start
- Default: 1 year ago
- More data = better model (usually)

**End Date**:
- Training data end
- Default: Today
- Recent data important

**Recommendations**:
- **Short-term models**: 1-2 years
- **Long-term models**: 3-5 years
- **Stable assets**: Longer periods
- **Volatile assets**: Shorter periods

##### Out-of-Sample Testing

**Purpose**: Validate forecast accuracy

**How It Works**:
- **Training Period**: Before validation dates
- **Validation Period**: Specified dates
- Model trained on training data
- Forecast evaluated on validation data

**Training Window Options**:
- **30% (Recommended)**: Balanced
- **50%**: More training data
- **60%**: Maximum reliability

**Validation Metrics**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy

##### Forecast Horizon

**Options**:
- **1 Day**: Next day forecast
- **1 Week (5 days)**: Short-term
- **2 Weeks (10 days)**: Medium-term
- **1 Month (21 days)**: Monthly
- **3 Months (63 days)**: Quarterly
- **6 Months (126 days)**: Semi-annual
- **1 Year (252 days)**: Annual
- **Custom**: Any number of days

**Selection**: Based on investment horizon

#### Forecasting Methods

##### Classical Methods

**1. ARIMA** (Auto-Regressive Integrated Moving Average):
- Time series model
- Handles trends and seasonality
- Auto parameter selection
- Good for stable series

**2. GARCH** (Generalized Autoregressive Conditional Heteroskedasticity):
- Models volatility clustering
- Accounts for changing volatility
- Good for financial data
- Risk-focused

**3. ARIMA-GARCH**:
- Combines ARIMA and GARCH
- Models returns and volatility
- Most comprehensive classical
- Best for financial time series

##### Machine Learning Methods

**4. Random Forest**:
- Ensemble tree-based
- Handles non-linearities
- Feature importance
- Robust to outliers

**5. SVM** (Support Vector Machine):
- Kernel-based
- Good generalization
- Handles complex patterns
- Regularization

**6. XGBoost**:
- Gradient boosting
- State-of-the-art performance
- Feature importance
- Fast training

##### Deep Learning Methods

**7. LSTM** (Long Short-Term Memory):
- Recurrent neural network
- Captures long-term dependencies
- Good for sequences
- Can be bidirectional

**8. TCN** (Temporal Convolutional Network):
- Convolutional approach
- Parallel processing
- Efficient training
- Good performance

**9. SSA-MAEMD-TCN**:
- Advanced decomposition
- Empirical Mode Decomposition
- Combined with TCN
- Most sophisticated
- Best for complex patterns

##### Time Series Methods

**10. Prophet**:
- Facebook's forecasting tool
- Handles seasonality well
- Holiday effects
- User-friendly

##### Ensemble Methods

**11. Weighted Average**:
- Combines multiple forecasts
- Weighted by performance
- More stable
- Better accuracy

**12. Model Selection**:
- Automatically selects best
- Based on validation metrics
- Single best forecast
- Simpler output

#### Forecast Results

##### Forecast Chart

**Display**:
- Historical data (training period)
- Validation period (if enabled)
- Forecast period
- Confidence intervals

**Features**:
- Interactive (zoom, pan)
- Hover for values
- Show/hide components
- Export chart

##### Validation Metrics

**If Out-of-Sample Enabled**:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: % correct direction

**Interpretation**:
- Lower errors = better
- Higher directional accuracy = better
- Comparison across methods

##### Model Information

**For Each Method**:
- Model parameters
- Training time
- Model complexity
- Diagnostics

**Examples**:
- ARIMA: (p, d, q) parameters
- LSTM: Layers, neurons, epochs
- Random Forest: Trees, depth

##### Forecast Statistics

**Point Forecast**:
- Expected value
- Most likely outcome

**Confidence Intervals**:
- 80% confidence interval
- 95% confidence interval
- Uncertainty range

**Percentiles**:
- 5th percentile (pessimistic)
- 50th percentile (median)
- 95th percentile (optimistic)

#### Multiple Methods Comparison

**Run Multiple Forecasts**:
- Select multiple methods
- Compare results
- Side-by-side charts
- Performance comparison

**Best Model Selection**:
- Automatic selection
- Based on validation metrics
- Recommended method
- Explanation

**Ensemble Forecast**:
- Weighted combination
- More stable
- Better accuracy
- Recommended for production

---

## Complete Features Reference

### Metrics Reference

#### Performance Metrics (18 total)

1. **Total Return**: Absolute return over period
2. **CAGR**: Compound Annual Growth Rate
3. **Annualized Return**: Return per year
4. **YTD**: Year-to-Date return
5. **MTD**: Month-to-Date return
6. **QTD**: Quarter-to-Date return
7. **1M Return**: Last 1 month
8. **3M Return**: Last 3 months
9. **6M Return**: Last 6 months
10. **1Y Return**: Last 1 year
11. **3Y Return**: Last 3 years
12. **5Y Return**: Last 5 years
13. **Best Month**: Highest monthly return
14. **Worst Month**: Lowest monthly return
15. **Win Rate**: Percentage of positive periods
16. **Payoff Ratio**: Average win / Average loss
17. **Profit Factor**: Gross profit / Gross loss
18. **Expectancy**: Expected value per trade

#### Risk Metrics (22 total)

1. **Daily Volatility**: Std dev of daily returns
2. **Weekly Volatility**: Std dev of weekly returns
3. **Monthly Volatility**: Std dev of monthly returns
4. **Annual Volatility**: Annualized volatility
5. **Max Drawdown**: Largest peak-to-trough decline
6. **Max Drawdown Date**: When max drawdown occurred
7. **Max Drawdown Duration**: Days in max drawdown
8. **Current Drawdown**: Current decline from peak
9. **Average Drawdown**: Mean of all drawdowns
10. **Drawdown Duration (Max)**: Longest drawdown period
11. **Drawdown Duration (Avg)**: Average drawdown length
12. **Recovery Time**: Time to recover from drawdown
13. **Ulcer Index**: Drawdown severity measure
14. **Pain Index**: Average drawdown
15. **VaR (90%)**: Value at Risk at 90% confidence
16. **VaR (95%)**: Value at Risk at 95% confidence
17. **VaR (99%)**: Value at Risk at 99% confidence
18. **CVaR (90%)**: Conditional Value at Risk at 90% confidence
19. **CVaR (95%)**: Conditional Value at Risk at 95% confidence
20. **CVaR (99%)**: Conditional Value at Risk at 99% confidence
21. **Tail Ratio**: Ratio of 95th percentile to 5th percentile returns
22. **Skewness**: Measure of return distribution asymmetry

#### Risk-Adjusted Ratios (10 total)

1. **Sharpe Ratio**: (Return - Risk-free) / Volatility
2. **Sortino Ratio**: (Return - Risk-free) / Downside Deviation
3. **Calmar Ratio**: Annual Return / Max Drawdown
4. **Information Ratio**: Active Return / Tracking Error
5. **Treynor Ratio**: (Return - Risk-free) / Beta
6. **Jensen's Alpha**: Excess return over CAPM expected return
7. **Omega Ratio**: Probability-weighted ratio of gains to losses
8. **Kappa 3**: Downside risk-adjusted return measure
9. **Gain-to-Pain Ratio**: Total return / Sum of drawdowns
10. **Sterling Ratio**: Average annual return / Average annual max drawdown

#### Market Metrics (15 total)

1. **Beta**: Sensitivity to market movements
2. **Alpha (CAPM)**: Excess return vs expected return
3. **R² (R-squared)**: Correlation with benchmark squared
4. **Correlation**: Correlation coefficient with benchmark
5. **Tracking Error**: Standard deviation of active returns
6. **Up Capture**: Performance in up markets
7. **Down Capture**: Performance in down markets
8. **Up/Down Capture Ratio**: Up Capture / Down Capture
9. **Active Share**: Percentage of portfolio different from benchmark
10. **Information Coefficient**: Correlation of forecasts with returns
11. **Relative Volatility**: Portfolio volatility / Benchmark volatility
12. **Relative Return**: Portfolio return - Benchmark return
13. **Excess Return**: Return above risk-free rate
14. **Market Correlation**: Correlation with overall market
15. **Sector Correlation**: Correlation with sector indices

---

## Workflow Examples

### Example 1: Creating and Analyzing a Tech Portfolio

**Scenario**: Create a technology-focused portfolio and analyze its performance

**Steps**:

1. **Create Portfolio**:
   - Navigate to "Create Portfolio"
   - Choose "Text Input" method
   - Enter: `AAPL 25%, MSFT 25%, GOOGL 20%, NVDA 15%, META 15%`
   - Set starting capital: $100,000
   - Complete wizard

2. **Initial Analysis**:
   - Go to "Portfolio Analysis"
   - Select your portfolio
   - Set date range: 1 year
   - Click "Calculate"
   - Review Overview tab:
     - Check Total Return
     - Review Sharpe Ratio
     - Examine Max Drawdown
     - Compare with SPY benchmark

3. **Deep Dive**:
   - Switch to "Performance" tab
   - Review monthly returns heatmap
   - Check best/worst periods
   - Analyze returns distribution

4. **Risk Assessment**:
   - Go to "Risk Analysis" tab
   - Review VaR metrics
   - Check drawdown analysis
   - Examine rolling volatility

5. **Optimization**:
   - Navigate to "Portfolio Optimization"
   - Select "Max Sharpe" method
   - Set constraints (max 30% per asset)
   - Generate optimized weights
   - Compare before/after metrics

**Expected Outcomes**:
- Portfolio created successfully
- Performance metrics calculated
- Risk metrics displayed
- Optimization suggestions provided

### Example 2: Stress Testing a Conservative Portfolio

**Scenario**: Test how a conservative portfolio would perform during market crashes

**Steps**:

1. **Select Portfolio**:
   - Go to "Portfolio List"
   - Select your conservative portfolio
   - Or create one with: `JNJ 20%, PG 20%, KO 20%, XOM 20%, T 20%`

2. **Run Historical Stress Tests**:
   - Navigate to "Risk Analysis"
   - Go to "Historical Scenarios" tab
   - Select scenarios:
     - 2008 Financial Crisis
     - 2020 COVID-19 Crash
     - 2000 Dot-com Bubble
   - Review impact for each scenario

3. **Custom Scenario**:
   - Go to "Custom Scenario" tab
   - Create scenario: "Inflation Spike"
   - Set asset shocks:
     - Consumer staples: -10%
     - Energy: +15%
     - Utilities: -5%
   - Analyze portfolio impact

4. **Monte Carlo Simulation**:
   - Go to "Monte Carlo Simulation" tab
   - Set: 10,000 paths, 30-day horizon
   - Review probability distributions
   - Check confidence intervals

5. **Interpret Results**:
   - Compare scenario impacts
   - Identify worst-case losses
   - Assess recovery potential
   - Make risk management decisions

**Expected Outcomes**:
- Portfolio resilience assessed
- Worst-case scenarios identified
- Risk mitigation strategies suggested

### Example 3: Forecasting Portfolio Returns

**Scenario**: Forecast future portfolio performance for strategic planning

**Steps**:

1. **Select Portfolio**:
   - Go to "Forecasting" page
   - Select portfolio from dropdown

2. **Configure Parameters**:
   - Set training period: 2 years
   - Enable out-of-sample testing (30% window)
   - Set forecast horizon: 3 months (63 days)

3. **Run Multiple Forecasts**:
   - Select methods:
     - ARIMA-GARCH (classical)
     - LSTM (deep learning)
     - XGBoost (machine learning)
     - Ensemble (weighted average)
   - Click "Generate Forecasts"

4. **Review Results**:
   - Examine forecast chart
   - Check validation metrics (MAE, RMSE, MAPE)
   - Compare method performance
   - Review confidence intervals

5. **Make Decisions**:
   - Use ensemble forecast (most stable)
   - Consider confidence intervals
   - Plan rebalancing if needed
   - Adjust strategy based on outlook

**Expected Outcomes**:
- Multiple forecast methods compared
- Best model automatically selected
- Confidence intervals provided
- Strategic insights generated

### Example 4: Optimizing Portfolio with Constraints

**Scenario**: Optimize portfolio while respecting real-world constraints

**Steps**:

1. **Select Portfolio**:
   - Go to "Portfolio Optimization"
   - Choose portfolio to optimize

2. **Set Date Range**:
   - Start: 1 year ago
   - End: Today
   - Enable out-of-sample testing

3. **Choose Method**:
   - Select "Black-Litterman" (stable optimization)
   - Or "Risk Parity" (balanced risk)

4. **Configure Constraints**:
   - **Weight Constraints**:
     - Min weight: 2% (prevent tiny positions)
     - Max weight: 25% (prevent concentration)
   - **Portfolio Constraints**:
     - Long-only (no shorting)
     - Sum to 100%
   - **Cardinality**:
     - Min positions: 10
     - Max positions: 20

5. **Run Optimization**:
   - Click "Optimize"
   - Review optimized weights
   - Check trade list
   - Compare metrics (before/after)

6. **Sensitivity Analysis**:
   - Vary risk aversion parameter
   - Test constraint changes
   - Assess robustness

7. **Implement Changes**:
   - Review trade list
   - Consider transaction costs
   - Execute trades gradually

**Expected Outcomes**:
- Optimized weights generated
- Constraints respected
- Metrics improved
- Trade list provided

---

## Tips & Best Practices

### Portfolio Creation

**Best Practices**:
- **Start Simple**: Begin with 5-10 positions, expand gradually
- **Diversify**: Don't put all eggs in one basket (max 20-30% per asset)
- **Use Templates**: Leverage pre-built strategies for quick start
- **Validate Tickers**: Always verify ticker symbols before creating
- **Set Realistic Capital**: Use actual investment amounts for accurate analysis

**Common Mistakes to Avoid**:
- Creating portfolios with duplicate tickers
- Setting weights that don't sum to 100%
- Using invalid or delisted tickers
- Forgetting to set starting capital
- Not reviewing portfolio before saving

### Analysis Best Practices

**Date Range Selection**:
- **Short-term analysis**: 3-6 months (for active trading)
- **Medium-term analysis**: 1 year (most common, balanced view)
- **Long-term analysis**: 3-5 years (for strategic planning)
- **Avoid**: Very short periods (<1 month) - insufficient data

**Benchmark Selection**:
- **Tech portfolios**: Use QQQ (NASDAQ-100)
- **Broad market**: Use SPY (S&P 500) or VTI (Total Market)
- **Sector-specific**: Use relevant sector ETF
- **International**: Use appropriate international index

**Metric Interpretation**:
- **Sharpe Ratio**: >1 is good, >2 is excellent
- **Max Drawdown**: Consider your risk tolerance
- **Beta**: <1 = less volatile, >1 = more volatile
- **Alpha**: Positive = outperformance, negative = underperformance

### Optimization Tips

**Method Selection**:
- **General use**: Mean-Variance (Max Sharpe)
- **Stable results**: Black-Litterman
- **Risk-focused**: Risk Parity or CVaR Optimization
- **Many assets**: Hierarchical Risk Parity (HRP)
- **Conservative**: Mean-CVaR or Robust Optimization

**Constraint Guidelines**:
- **Min weight**: 2-5% (prevents tiny positions)
- **Max weight**: 20-30% (prevents concentration)
- **Cardinality**: 10-20 positions (balance diversification and complexity)
- **Turnover**: Limit to 20-30% (reduce transaction costs)

**Out-of-Sample Testing**:
- Always enable for validation
- Use 30% window (balanced)
- Compare training vs validation performance
- Trust validation results more than training

### Risk Management

**VaR Usage**:
- Use multiple methods (Historical + Parametric + Cornish-Fisher)
- Compare results across methods
- Consider 95% and 99% confidence levels
- Review rolling VaR for trend analysis

**Stress Testing**:
- Test against historical crises
- Create custom scenarios for your specific risks
- Run Monte Carlo for probabilistic analysis
- Use scenario chains for sequential events

**Drawdown Management**:
- Monitor current drawdown regularly
- Set drawdown limits (e.g., -20% trigger)
- Review recovery times
- Consider stop-loss strategies

### Forecasting Best Practices

**Model Selection**:
- **Short-term** (1-5 days): ARIMA or LSTM
- **Medium-term** (1-3 months): ARIMA-GARCH or XGBoost
- **Long-term** (6-12 months): Prophet or Ensemble
- **Always**: Use ensemble for production decisions

**Training Data**:
- Use sufficient history (at least 1 year, preferably 2-3 years)
- Include recent data (last 6 months important)
- Avoid periods with structural breaks
- Consider market regime changes

**Validation**:
- Always enable out-of-sample testing
- Review MAE, RMSE, MAPE metrics
- Check directional accuracy
- Compare multiple methods

**Interpretation**:
- Trust confidence intervals, not just point forecasts
- Consider worst-case scenarios (5th percentile)
- Update forecasts regularly (monthly/quarterly)
- Combine with fundamental analysis

### Data Management

**Cache Strategy**:
- Prices cached for 5 minutes (current prices)
- Historical data cached for 24 hours
- Clear cache if data seems stale
- Refresh manually if needed

**API Usage**:
- Monitor API status in sidebar
- Check response times (<100ms is good)
- Be aware of rate limits
- Use cached data when possible

**Data Quality**:
- Verify ticker symbols before use
- Check for missing data periods
- Review data gaps in charts
- Report data issues if found

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Portfolio not found" or "Portfolio does not exist"

**Possible Causes**:
- Portfolio was deleted
- Database connection issue
- Session expired

**Solutions**:
1. Check "Portfolio List" to verify portfolio exists
2. Refresh the page
3. Restart the application if persistent
4. Check database connection status

#### Issue: "Ticker not found" or "Invalid ticker symbol"

**Possible Causes**:
- Typo in ticker symbol
- Ticker delisted or changed
- Market data API issue
- Ticker not supported by data provider

**Solutions**:
1. Verify ticker symbol spelling
2. Check if ticker is still active
3. Try alternative ticker format (e.g., BRK.B vs BRK-B)
4. Wait a few minutes and retry (API might be temporarily unavailable)
5. Check system status in sidebar

#### Issue: "Insufficient data" or "Not enough historical data"

**Possible Causes**:
- Date range too long for new ticker
- Ticker recently listed
- Data gaps in historical data
- API limitations

**Solutions**:
1. Reduce date range (try 6 months instead of 1 year)
2. Check ticker listing date
3. Verify data availability in price charts
4. Use different ticker if data insufficient
5. Contact support if persistent

#### Issue: "Optimization failed" or "No solution found"

**Possible Causes**:
- Constraints too restrictive
- Insufficient data
- Numerical issues
- Conflicting constraints

**Solutions**:
1. Relax constraints (increase max weight, reduce min weight)
2. Increase date range for more data
3. Try different optimization method
4. Check if constraints are feasible (e.g., min weights sum to >100%)
5. Remove cardinality constraints if too restrictive

#### Issue: "Forecast error" or "Model training failed"

**Possible Causes**:
- Insufficient training data
- Data quality issues
- Model complexity too high
- Numerical instability

**Solutions**:
1. Increase training period (more historical data)
2. Check data quality (no missing values, no outliers)
3. Try simpler model (ARIMA instead of LSTM)
4. Reduce forecast horizon
5. Enable out-of-sample testing to validate

#### Issue: "Slow performance" or "Calculation taking too long"

**Possible Causes**:
- Large portfolio (many assets)
- Long date range
- Complex optimization method
- Multiple forecasts running
- API rate limiting

**Solutions**:
1. Reduce date range
2. Use simpler optimization method
3. Run forecasts one at a time
4. Check API status (might be slow)
5. Clear cache and retry
6. Restart application if needed

#### Issue: "Chart not displaying" or "Empty chart"

**Possible Causes**:
- No data for selected period
- All values are zero
- Data filtering issue
- Browser compatibility

**Solutions**:
1. Check date range (ensure data exists)
2. Verify portfolio has positions
3. Check if benchmark data available
4. Try different browser
5. Refresh page
6. Clear browser cache

#### Issue: "Export failed" or "PDF generation error"

**Possible Causes**:
- Browser automation not available
- Streamlit app not running
- Permission issues
- File system errors

**Solutions**:
1. Ensure Streamlit app is running
2. Check browser automation tools installed
3. Verify write permissions
4. Try CSV export instead of PDF
5. Check available disk space

### Performance Optimization

**For Large Portfolios** (20+ assets):
- Use shorter date ranges for faster calculations
- Prefer simpler optimization methods (Equal Weight, Min Variance)
- Disable out-of-sample testing for faster results
- Use cached data when possible

**For Long Date Ranges** (5+ years):
- Calculate metrics in batches
- Use sampling for Monte Carlo (fewer paths)
- Simplify forecast models
- Export data for external analysis if needed

**For Multiple Forecasts**:
- Run forecasts sequentially, not simultaneously
- Use faster methods (ARIMA, Prophet) for quick results
- Save results before running next forecast
- Use ensemble only for final decision

### Getting Help

**Self-Help Resources**:
- Review this User Guide
- Check [Architecture Documentation](ARCHITECTURE.md) for technical details
- Review error messages carefully
- Check system status in sidebar

**When to Contact Support**:
- Persistent errors after troubleshooting
- Data quality issues
- Feature requests
- Bug reports

**Information to Provide**:
- Error message (exact text)
- Steps to reproduce
- Portfolio details (if relevant)
- Date range used
- Browser and OS information

---

## Keyboard Shortcuts & UI Tips

### Navigation Shortcuts

**General Navigation**:
- **Sidebar Toggle**: Click hamburger menu (☰) or use `Ctrl+B` (if implemented)
- **Page Refresh**: `F5` or `Ctrl+R`
- **Back Button**: Browser back button or `Alt+Left Arrow`

**Within Pages**:
- **Tab Navigation**: `Tab` key to move between fields
- **Enter**: Submit forms
- **Escape**: Close dialogs/modals

### Chart Interactions

**Plotly Charts** (Most charts in the app):

**Zooming**:
- **Mouse Wheel**: Zoom in/out
- **Double-Click**: Reset zoom
- **Box Select**: Drag to select area and zoom

**Panning**:
- **Click and Drag**: Pan around chart
- **Arrow Keys**: Pan in direction (if enabled)

**Other Interactions**:
- **Hover**: See exact values
- **Click Legend**: Show/hide series
- **Right-Click**: Context menu (reset, export, etc.)

### Form Tips

**Text Input**:
- **Auto-complete**: Use for ticker entry
- **Tab**: Move to next field
- **Enter**: Submit (if single field)
- **Escape**: Clear/cancel

**Date Pickers**:
- **Click Calendar Icon**: Open date picker
- **Keyboard Entry**: Type dates directly (YYYY-MM-DD)
- **Arrow Keys**: Navigate calendar
- **Enter**: Confirm selection

**Dropdowns**:
- **Type to Search**: Filter options
- **Arrow Keys**: Navigate options
- **Enter**: Select option
- **Escape**: Close dropdown

### Table Interactions

**Sortable Tables**:
- **Click Column Header**: Sort by that column
- **Click Again**: Reverse sort order
- **Multiple Clicks**: Cycle through sort options

**Searchable Tables**:
- **Search Box**: Type to filter rows
- **Real-time Filtering**: Results update as you type
- **Clear**: Click X or delete all text

**Selectable Rows**:
- **Checkbox**: Click to select/deselect
- **Select All**: Use header checkbox
- **Bulk Actions**: Available after selection

### UI Efficiency Tips

**Quick Actions**:
- **Dashboard Navigation**: Use quick action buttons for faster access
- **Portfolio Selection**: Use search in dropdowns
- **Date Range**: Use preset buttons (1M, 3M, 1Y, etc.) when available

**View Management**:
- **Expand/Collapse**: Click section headers to expand/collapse
- **Tab Switching**: Click tabs or use keyboard if implemented
- **Full Screen**: Use browser fullscreen (F11) for more space

**Data Entry**:
- **Copy-Paste**: Use for bulk entry in text input method
- **CSV Import**: Faster for many positions
- **Templates**: Quick start for common strategies

### Browser Tips

**Recommended Browsers**:
- **Chrome**: Best compatibility
- **Firefox**: Good alternative
- **Edge**: Windows default, works well
- **Safari**: Mac default, generally compatible

**Browser Settings**:
- **Enable JavaScript**: Required for all features
- **Allow Pop-ups**: For PDF generation
- **Clear Cache**: If data seems stale
- **Disable Extensions**: If experiencing issues

**Performance**:
- **Close Unused Tabs**: Free up memory
- **Disable Extensions**: Reduce resource usage
- **Update Browser**: Latest version for best performance
- **Hard Refresh**: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac) to clear cache

### Accessibility

**Keyboard Navigation**:
- Most features accessible via keyboard
- Tab order follows logical flow
- Enter key submits forms
- Escape key cancels/closes

**Screen Readers**:
- Alt text on images
- Proper heading structure
- Form labels associated
- ARIA labels where needed

**Visual Adjustments**:
- Dark theme available (if implemented)
- Font size adjustable via browser
- High contrast mode (browser setting)

---

## Appendix

### Glossary of Terms

**Alpha**: Excess return above expected return (from CAPM model)

**Beta**: Measure of portfolio sensitivity to market movements (1.0 = moves with market)

**CAGR**: Compound Annual Growth Rate - geometric mean annual return

**CVaR**: Conditional Value at Risk - expected loss beyond VaR threshold

**Drawdown**: Decline from peak value to trough

**Efficient Frontier**: Set of optimal portfolios offering highest return for given risk level

**Max Drawdown**: Largest peak-to-trough decline in portfolio value

**Sharpe Ratio**: Risk-adjusted return measure: (Return - Risk-free) / Volatility

**Sortino Ratio**: Like Sharpe, but uses downside deviation instead of total volatility

**Tracking Error**: Standard deviation of active returns (portfolio return - benchmark return)

**VaR**: Value at Risk - maximum expected loss at given confidence level

**Volatility**: Standard deviation of returns - measure of price variability

### Data Sources

**Price Data**:
- Real-time and historical prices from market data providers
- Cached for performance
- Updated regularly

**Market Indices**:
- S&P 500 (^GSPC)
- NASDAQ (^NDX)
- Dow Jones (^DJI)
- Russell 2000 (^RUT)

**Benchmark ETFs**:
- SPY: S&P 500 ETF
- QQQ: NASDAQ-100 ETF
- VTI: Total Stock Market ETF
- DIA: Dow Jones ETF
- IWM: Russell 2000 ETF

### System Requirements

**Minimum Requirements**:
- Modern web browser (Chrome, Firefox, Edge, Safari)
- Internet connection (for market data)
- JavaScript enabled
- 4GB RAM recommended

**Recommended**:
- 8GB+ RAM for large portfolios
- Fast internet connection (<100ms latency ideal)
- Modern CPU for complex optimizations

### Version Information

**Current Version**: See application sidebar or about page

**Update Frequency**: Regular updates with new features and improvements

**Changelog**: Available in application or repository

---

**End of User Guide**

For technical documentation, see [ARCHITECTURE.md](ARCHITECTURE.md)  
For implementation details, see [PLAN.md](PLAN.md)  
For requirements, see [REQUIREMENTS.md](REQUIREMENTS.md)