# WILD MARKET CAPITAL - Technical Requirements Specification

**Project Name**: Wild Market Capital Portfolio Management System  
**Version**: 1.0  
**Date**: 2025-10-29  
**Status**: Requirements Definition Phase  
**Document Type**: Complete Technical Requirements

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Functional Requirements](#2-functional-requirements)
3. [API Specifications](#3-api-specifications)
4. [Frontend Specifications](#4-frontend-specifications)
5. [Data Models](#5-data-models)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Additional Requirements](#7-additional-requirements)

---

## 1. PROJECT OVERVIEW

### 1.1 Project Description

Wild Market Capital is a professional-grade portfolio management system designed for institutional investors, portfolio managers, and sophisticated individual traders. The system provides comprehensive tools for portfolio construction, performance analysis, risk management, optimization, and scenario testing.

**Core Value Proposition**:
The platform combines institutional-quality analytics (70+ metrics including Sharpe, Sortino, Calmar ratios, VaR, CVaR, factor analysis) with a modern, intuitive interface inspired by TradingView. It enables users to create portfolios through multiple methods (text parsing, CSV upload, manual entry, templates), analyze performance across various dimensions, optimize allocations using 17 different algorithms (Mean-Variance, Risk Parity, Black-Litterman, HRP, etc.), stress-test against 25+ historical scenarios, and generate professional PDF/Excel reports.

**Technical Architecture**:
The system follows a backend-first architecture with clean separation between calculation engine (Python core modules) and user interface. This enables future migration from Streamlit MVP to React-based web application without any backend changes. The core analytics engine is framework-agnostic and can be wrapped with REST API for multi-platform access.

### 1.2 Goals and Objectives

**Primary Goals**:
1. Provide institutional-quality portfolio analytics accessible to professional investors
2. Enable data-driven investment decisions through comprehensive risk and performance metrics
3. Streamline portfolio management workflows from creation to optimization to reporting
4. Deliver accurate, fast calculations (<1 second for typical portfolios with 10-20 positions)
5. Maintain calculation transparency and reproducibility

**Key Objectives**:
1. **Analytics Depth**: Implement 70+ industry-standard metrics covering performance (18), risk (22), risk-adjusted ratios (15), and market-related metrics (15)
2. **Optimization Variety**: Support 17 optimization methods from classic Mean-Variance to advanced Hierarchical Risk Parity
3. **Risk Management**: Provide 25+ historical stress test scenarios plus custom scenario builder with event chains
4. **Flexibility**: Enable portfolio creation through 5 different methods (text, CSV, manual, template, clone)
5. **Professional Output**: Generate client-ready PDF/Excel reports with customizable sections
6. **Scalability**: Maintain backend independence from UI for future multi-platform deployment

**Success Metrics**:
- All 70+ metrics calculating correctly and matching reference implementations
- Page load < 2 seconds, calculations < 1 second for typical portfolios
- Optimization convergence < 5 seconds for 20-position portfolios
- Support for portfolios up to 100 positions with 10+ years of daily data
- Zero data loss, graceful handling of missing market data

### 1.3 Target Audience

**Primary User Segments**:

**Segment 1: Professional Portfolio Managers**
- Profile: Buy-side firms, Registered Investment Advisors (RIAs), wealth management professionals
- Use Cases: Managing 10-20 client portfolios ($50M-$500M each), conducting performance attribution, generating quarterly client reports, monitoring risk across portfolios
- Needs: Quick risk monitoring dashboard, detailed attribution analysis, professional report generation, compliance-ready documentation
- Values: Speed, accuracy, professional presentation, audit trail

**Segment 2: Quantitative Analysts / Researchers**
- Profile: Institutional investors, hedge funds, academic researchers
- Use Cases: Backtesting portfolio strategies, comparing optimization methods, analyzing factor exposures, conducting research
- Needs: Calculation transparency, data export capabilities, multiple optimization algorithms, factor analysis tools
- Values: Analytical depth, methodological rigor, reproducibility, flexibility

**Segment 3: Sophisticated Individual Investors**
- Profile: High-net-worth individuals managing personal portfolios ($100K-$5M)
- Use Cases: Understanding portfolio risk, optimizing allocation, stress-testing against market scenarios, tracking performance
- Needs: Clear visualizations, educational tooltips, actionable insights, simplified workflows
- Values: Ease of use, clarity, practical recommendations, learning resources

### 1.4 Platform

**Current Phase (MVP - Streamlit)**:
- **Frontend**: Streamlit web application (Python framework)
- **Backend**: Core calculation engine (pure Python with NumPy/Pandas/SciPy)
- **Data Sources**: Yahoo Finance (primary, free), Alpha Vantage (optional), IEX Cloud (optional)
- **Deployment**: Local desktop application (python run.py) or single-server web deployment
- **Storage**: SQLite database for portfolio data, file-based cache for price data
- **Users**: Single-user (no authentication required)

**Future Phase (React Web App)**:
- **Frontend**: React/Next.js progressive web application
- **Backend**: RESTful API wrapping core calculation engine (FastAPI/Flask)
- **Authentication**: JWT-based auth with role-based access control
- **Data Sources**: Multiple providers with intelligent fallback (Yahoo → Alpha Vantage → IEX)
- **Deployment**: Cloud-native (AWS/GCP/Azure), containerized (Docker), auto-scaling
- **Storage**: PostgreSQL for relational data, Redis for caching, S3 for reports
- **Users**: Multi-user with team collaboration features

---

## 2. FUNCTIONAL REQUIREMENTS

### 2.1 Portfolio Creation

#### FR-001: Text-Based Portfolio Creation

**Priority**: Critical (MVP Must-Have)  
**Module**: Portfolio Creation  
**Dependencies**: FR-015 (Ticker Validation), FR-016 (Price Fetching)

**Description**:
Allow users to create portfolios by pasting text containing ticker symbols and weights in various formats. The system must intelligently parse different notations (percentages, decimals, ratios) and normalize weights automatically.

**User Stories**:
- **US-001-1**: As a portfolio manager, I want to quickly create a portfolio by copying allocation data from emails or spreadsheets, so I can save time on manual data entry
- **US-001-2**: As a user, I want the system to intelligently parse different weight formats without manual reformatting, so I don't have to worry about exact syntax
- **US-001-3**: As a user, I want to see validation errors immediately with helpful suggestions, so I can quickly correct mistakes

**Supported Input Formats**:

```text
Format 1 - Percentages with symbol:
"AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, NVDA 10%"

Format 2 - Decimal notation:
"AAPL 0.3, MSFT 0.25, GOOGL 0.2, AMZN 0.15, NVDA 0.1"

Format 3 - Colon notation:
"AAPL:30 MSFT:25 GOOGL:20 AMZN:15 NVDA:10"

Format 4 - Equal weights (tickers only):
"AAPL,MSFT,GOOGL,AMZN,NVDA"
→ System assigns 20% to each (1/5 = 0.20)

Format 5 - Mixed with newlines:
AAPL 30%
MSFT 25%
GOOGL 20%
AMZN 15%
NVDA 10%

Format 6 - Tab-separated (from spreadsheet):
AAPL    30%
MSFT    25%
GOOGL   20%
```

**Functional Flow**:

1. User navigates to "Create Portfolio" page
2. User selects "Text Input" method (tab or radio button)
3. System displays large text area with placeholder example
4. User pastes or types portfolio allocation text
5. System parses text in real-time using regex patterns:
   - Extract ticker symbols (1-10 uppercase alphanumeric characters)
   - Extract weights (numbers with optional %, decimal, or ratio notation)
   - Handle various separators (comma, space, tab, newline, colon)
6. System displays parsed results in preview table:
   - Columns: Ticker, Parsed Weight, Validation Status
   - Color coding: Green checkmark (valid), Yellow spinner (validating), Red X (invalid)
7. For each ticker, system validates via API call:
   - Check ticker exists in market data provider
   - Fetch basic info (company name, current price, sector)
   - Store validation result
8. If weights don't sum to 100%, show warning banner with "Auto-Normalize" button
9. User clicks "Auto-Normalize" (optional) → system proportionally adjusts all weights to sum to 1.0
10. User reviews parsed positions in table, can manually edit weights
11. User fills portfolio metadata form:
    - Portfolio name (required)
    - Description (optional)
    - Tags (optional, comma-separated or chip input)
    - Starting capital (optional, default $100,000)
12. User clicks "Create Portfolio" button
13. System performs final validation:
    - All tickers valid
    - Weights sum to 1.0 (±0.0001 tolerance)
    - Name unique for user
14. System creates portfolio record in database
15. System fetches initial prices for all positions
16. System calculates initial position values and shares
17. System displays success message with portfolio ID
18. System redirects to Portfolio Management detail view

**Input Data**:

```typescript
interface TextInputData {
  raw_text: string;              // User-pasted text
  portfolio_name: string;        // Required, 1-100 chars, unique per user
  description?: string;          // Optional, max 500 chars
  tags?: string[];               // Optional array of tag strings
  starting_capital?: number;     // Optional, default 100000.00
  base_currency?: string;        // Optional, default 'USD'
}
```

**Output Data**:

```json
{
  "portfolio_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Tech Leaders",
  "description": "Large-cap technology portfolio",
  "tags": ["tech", "growth"],
  "starting_capital": 100000.00,
  "base_currency": "USD",
  "creation_date": "2025-10-29T10:30:00Z",
  "positions": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "sector": "Technology",
      "weight": 0.30,
      "shares": 171.43,
      "current_price": 175.50,
      "current_value": 30085.07,
      "validation_status": "valid"
    },
    {
      "id": "660e8400-e29b-41d4-a716-446655440002",
      "ticker": "MSFT",
      "name": "Microsoft Corporation",
      "sector": "Technology",
      "weight": 0.25,
      "shares": 74.68,
      "current_price": 335.10,
      "current_value": 25023.59,
      "validation_status": "valid"
    }
  ],
  "total_value": 100000.00,
  "position_count": 5,
  "total_weight": 1.0,
  "validation_summary": {
    "is_valid": true,
    "warnings": [],
    "errors": []
  }
}
```

**Validation Rules**:

1. **Ticker Validation**:
   - Must be 1-10 characters
   - Uppercase alphanumeric only
   - Must exist in market data provider (API call)
   - Must return valid price data (not delisted/suspended)

2. **Weight Validation**:
   - Individual weights: 0.0 < weight <= 1.0
   - Sum of all weights: 1.0 ± 0.0001 (tolerance for floating point)
   - No negative weights (unless short-selling enabled in settings)

3. **Portfolio Validation**:
   - Minimum 1 position, maximum 100 positions
   - No duplicate tickers
   - Portfolio name must be unique per user
   - Name must be 1-100 characters

**Error Handling**:

**Error Type 1: Invalid Ticker**
```json
{
  "field": "positions[2].ticker",
  "code": "INVALID_TICKER",
  "message": "Ticker 'GOGL' not found",
  "suggestion": "Did you mean 'GOOGL'?",
  "severity": "error"
}
```
UI Treatment: Show red X icon next to ticker in preview table, display error message below, offer "Replace with GOOGL" button

**Error Type 2: Weights Don't Sum to 100%**
```json
{
  "field": "positions",
  "code": "WEIGHTS_INVALID_SUM",
  "message": "Weights sum to 98.5%. Total must equal 100%.",
  "current_sum": 0.985,
  "expected_sum": 1.0,
  "severity": "warning"
}
```
UI Treatment: Show yellow warning banner at top with "Auto-Normalize to 100%" button

**Error Type 3: Duplicate Tickers**
```json
{
  "field": "positions",
  "code": "DUPLICATE_TICKERS",
  "message": "Ticker 'AAPL' appears 2 times",
  "duplicates": ["AAPL"],
  "severity": "error"
}
```
UI Treatment: Highlight duplicate rows in red, offer "Merge Duplicates" button (sums weights)

**Error Type 4: API Failure**
```json
{
  "code": "API_UNAVAILABLE",
  "message": "Unable to validate tickers. Market data provider is unavailable.",
  "severity": "warning",
  "action": "Allow user to proceed with manual confirmation or retry"
}
```
UI Treatment: Show warning banner, offer "Retry Validation" and "Proceed Anyway" buttons

**Error Type 5: Name Conflict**
```json
{
  "field": "portfolio_name",
  "code": "NAME_CONFLICT",
  "message": "Portfolio name 'Tech Leaders' already exists",
  "severity": "error"
}
```
UI Treatment: Show inline error below name field, suggest "Tech Leaders (2)"

**Priority Justification**: Critical because text input is the fastest method for professional users to create portfolios from existing allocations. Many users copy data from emails, broker statements, or other tools.

---

#### FR-002: CSV/Excel File Upload

**Priority**: Critical (MVP Must-Have)  
**Module**: Portfolio Creation  
**Dependencies**: FR-015, FR-016

**Description**:
Allow users to upload CSV or Excel files containing portfolio holdings data with flexible column mapping. System must handle various file formats exported by different brokers and portfolio management tools.

**User Stories**:
- **US-002-1**: As a portfolio manager, I want to import existing portfolio holdings from my broker's CSV export, so I can quickly onboard existing portfolios
- **US-002-2**: As a user, I want to map columns flexibly because different sources use different header names (ticker vs symbol vs stock)
- **US-002-3**: As a user, I want to preview parsed data before creating the portfolio, so I can verify correctness

**Functional Flow**:

1. User navigates to "Create Portfolio" → "Upload File" tab
2. System displays drag-and-drop zone with file icon and instructions
3. User drags file or clicks to browse
4. System validates file:
   - Accepts: .csv, .xlsx, .xls
   - Max size: 5MB
   - Max rows: 1000
5. System reads file and auto-detects structure:
   - Header row detection (first row with text headers)
   - Column data type inference (numeric, text, date)
   - Delimiter detection for CSV (comma, semicolon, tab)
6. System displays Column Mapping interface:
   - Left side: Detected columns from file
   - Right side: Required/Optional fields
   - Drag-and-drop or dropdown to map columns
7. System highlights required fields (ticker, weight) in bold/red until mapped
8. For optional fields (name, sector, shares, purchase_price, purchase_date):
   - Show mapping status: Mapped / Not Mapped / Skip
   - Display sample values from first 3 rows
9. System shows Data Preview table (first 10 rows):
   - All columns with mapped field names
   - Color-coded validation:
     - Green: Valid data
     - Yellow: Warning (e.g., unusual values)
     - Red: Error (e.g., invalid ticker)
10. User confirms mapping or adjusts
11. System validates all rows:
    - Ticker existence check (API call per unique ticker)
    - Weight parsing (handle %, decimal, text like "30%")
    - Date parsing (multiple formats: YYYY-MM-DD, MM/DD/YYYY, etc.)
12. System displays validation summary:
    - X positions valid
    - Y positions with warnings
    - Z positions with errors
13. User can:
    - Fix errors inline in preview table
    - Skip rows with errors
    - Download error report
14. User fills portfolio metadata (name, description, tags, starting capital)
15. User clicks "Create Portfolio"
16. System creates portfolio with valid positions
17. System displays success message with summary

**Required File Columns** (flexible header names accepted):

**Ticker Column** - Required:
- Accepted names: ticker, symbol, stock, asset, code, instrument, security
- Data type: Text
- Validation: Must be 1-10 alphanumeric uppercase characters
- Example values: "AAPL", "MSFT", "BRK.B"

**Weight Column** - Required:
- Accepted names: weight, allocation, pct, percentage, %, alloc, position_size
- Data type: Numeric or text with %
- Validation: Must be parseable as number between 0-100 (if %) or 0.0-1.0 (if decimal)
- Example values: "30%", "0.30", "30", "30.0%"
- Parsing logic:
  - If contains "%": parse number and divide by 100
  - If number > 1: assume percentage, divide by 100
  - If number <= 1: assume decimal weight
  - If sum of all weights > 10: assume all percentages, divide by 100

**Optional File Columns**:

**Name Column**:
- Accepted names: name, company, company_name, description, full_name
- Data type: Text
- Example: "Apple Inc.", "Microsoft Corporation"
- If not provided: Fetched from API

**Sector Column**:
- Accepted names: sector, industry, category, classification
- Data type: Text
- Example: "Technology", "Consumer Cyclical"
- If not provided: Fetched from API

**Shares Column**:
- Accepted names: shares, quantity, units, qty, amount
- Data type: Numeric
- Example: 100, 15.5 (fractional shares allowed)
- If provided: System calculates weight from shares (shares * price / total_portfolio_value)
- If not provided: System calculates shares from weight

**Purchase Price Column**:
- Accepted names: purchase_price, cost_basis, buy_price, price, entry_price
- Data type: Numeric (currency)
- Example: 145.32, $145.32
- Optional: Used for P&L calculation
- Parsing: Remove currency symbols ($, €, £)

**Purchase Date Column**:
- Accepted names: purchase_date, buy_date, date, entry_date, transaction_date
- Data type: Date (multiple formats accepted)
- Example: "2024-01-15", "01/15/2024", "15-Jan-2024"
- Parsing: Try multiple date formats sequentially

**Example CSV File**:

```csv
ticker,weight,sector,purchase_price,purchase_date
AAPL,30%,Technology,145.32,2024-01-15
MSFT,25%,Technology,335.10,2024-01-20
GOOGL,20%,Technology,138.21,2024-02-01
AMZN,15%,Consumer Cyclical,142.50,2024-02-10
NVDA,10%,Technology,495.20,2024-03-01
```

**Example Excel File** (multiple formats):

Format 1 - Broker Export:
```
Portfolio Statement as of 2025-10-29
Account: 12345678

Symbol | Name                    | Shares | Price   | Value    | % of Portfolio
AAPL   | Apple Inc.             | 171    | 175.50  | 30000.50 | 30.0%
MSFT   | Microsoft Corporation  | 75     | 335.10  | 25132.50 | 25.1%
```

Format 2 - Simple:
```
Stock | Allocation
AAPL  | 30%
MSFT  | 25%
GOOGL | 20%
```

**Input Data**:

```typescript
interface FileUploadData {
  file: File;                    // File object (.csv, .xlsx, .xls)
  column_mapping: {              // User-defined or auto-detected mapping
    ticker: string;              // Source column name for ticker
    weight?: string;             // Source column name for weight
    shares?: string;             // Alternative to weight
    name?: string;               // Optional
    sector?: string;             // Optional
    purchase_price?: string;     // Optional
    purchase_date?: string;      // Optional
  };
  skip_rows?: number;            // Number of header rows to skip
  portfolio_metadata: {
    name: string;
    description?: string;
    tags?: string[];
    starting_capital?: number;
  };
}
```

**Output Data**: Same format as FR-001 (Portfolio creation response)

**Validation Rules**:

1. **File Validation**:
   - Size < 5MB
   - Extension in [.csv, .xlsx, .xls]
   - Readable/parseable format
   - Max 1000 data rows (excluding headers)

2. **Data Validation** (per row):
   - Required columns (ticker, weight OR shares) must have values
   - No completely empty rows (skip if all fields empty)
   - Ticker must match validation rules from FR-001
   - Weight must be numeric and 0 < weight <= 1.0 after normalization
   - Dates must parse successfully if provided
   - Numbers must be valid (not NaN, not infinity)

3. **Aggregate Validation**:
   - At least 1 valid row
   - Weights sum to 1.0 (±0.0001) after normalization
   - No duplicate tickers (unless merging option enabled)

**Error Handling**:

**File-Level Errors**:

```json
{
  "code": "FILE_TOO_LARGE",
  "message": "File size 7.2MB exceeds maximum 5MB",
  "max_size_mb": 5,
  "actual_size_mb": 7.2
}
```

```json
{
  "code": "UNSUPPORTED_FORMAT",
  "message": "File format '.pdf' not supported",
  "supported_formats": [".csv", ".xlsx", ".xls"]
}
```

```json
{
  "code": "PARSE_ERROR",
  "message": "Unable to parse file. Please ensure it's a valid CSV/Excel file",
  "technical_detail": "pandas.errors.ParserError: Expected 5 columns, found 3"
}
```

**Row-Level Errors** (displayed in preview table):

```json
{
  "row": 5,
  "ticker": "GOOG",
  "error": {
    "code": "INVALID_TICKER",
    "message": "Ticker 'GOOG' not found. Did you mean 'GOOGL'?",
    "severity": "error"
  }
}
```

```json
{
  "row": 12,
  "ticker": "AAPL",
  "error": {
    "code": "INVALID_WEIGHT",
    "message": "Weight '-5%' is negative. Weights must be positive",
    "severity": "error"
  }
}
```

```json
{
  "row": 8,
  "ticker": "BRK.B",
  "warning": {
    "code": "HIGH_CONCENTRATION",
    "message": "Position weight 45% exceeds recommended maximum 35%",
    "severity": "warning"
  }
}
```

**Column Mapping Errors**:

```json
{
  "code": "MISSING_REQUIRED_COLUMN",
  "message": "Required column 'ticker' not mapped",
  "required_columns": ["ticker", "weight or shares"]
}
```

UI Treatment:
- Show error summary banner: "5 errors, 3 warnings. Review issues below."
- In preview table, highlight error rows in red, warning rows in yellow
- Show error icon with tooltip on hover
- Provide "Download Error Report" button (CSV with all errors)
- Offer "Skip Error Rows" checkbox
- Allow inline editing of data in preview table

**Priority Justification**: Critical because most professional users have existing portfolios in CSV/Excel format from brokers or other tools. This is the primary onboarding path for existing portfolios.

---

#### FR-003: Manual Portfolio Entry

**Priority**: Important  
**Module**: Portfolio Creation  
**Dependencies**: FR-015, FR-016, FR-017 (Autocomplete)

**Description**:
Provide an interactive table interface for manually adding positions one by one with live validation, price fetching, and real-time weight calculation feedback.

**User Stories**:
- **US-003-1**: As a user, I want to build a portfolio from scratch with guided input and autocomplete, so I don't make errors in ticker entry
- **US-003-2**: As a user, I want to see live prices as I add tickers, so I know current market values
- **US-003-3**: As a user, I want real-time feedback on total weight allocation, so I know if I've allocated 100%

**Functional Flow**:

1. User navigates to "Create Portfolio" → "Manual Entry" tab
2. System displays empty positions table with "Add Position" button
3. User clicks "Add Position" button
4. System shows inline row in table with input fields:
   - Ticker (autocomplete input field)
   - Weight (%) or Dollar Amount ($) - toggle between modes
5. **Ticker Input Flow**:
   a. User starts typing ticker (e.g., "APP")
   b. System debounces input (300ms delay)
   c. System queries ticker search API with prefix match
   d. System shows dropdown with max 10 suggestions:
      - Format: "AAPL - Apple Inc. (Technology, $2.75T)"
      - Include: ticker, company name, sector, market cap
   e. User selects from dropdown or presses Enter if exact match
   f. System validates ticker via API
   g. System fetches and displays:
      - Company name
      - Current price (with timestamp)
      - Sector
      - Market cap
      - Last update time
6. **Weight Input Flow**:
   - Option A: User enters weight percentage (e.g., "30" or "30%")
     - System calculates dollar amount: weight * total_portfolio_value
     - System calculates shares: dollar_amount / current_price
   - Option B: User enters dollar amount (e.g., "30000" or "$30,000")
     - System calculates weight: dollar_amount / total_portfolio_value
     - System calculates shares: dollar_amount / current_price
   - System displays all three values (weight %, dollar amount, shares)
7. User clicks "Add" button or presses Enter
8. Position appears in positions table below with columns:
   - Ticker
   - Name
   - Sector
   - Current Price (with live update indicator)
   - Weight (editable)
   - Dollar Amount (calculated)
   - Shares (calculated, fractional)
   - Actions (Edit, Delete)
9. System updates running total row at bottom:
   - Total Weight: X.XX% (color-coded)
     - Green if == 100.00%
     - Yellow if 95-99% or 101-105%
     - Red if < 95% or > 105%
   - Total Value: $X,XXX.XX
   - Position Count: N
10. User repeats steps 3-8 for all positions
11. If weights don't sum to 100%, system shows:
    - Warning message: "Total allocation: 98.5%. Add 1.5% more or use Auto-Normalize."
    - "Auto-Normalize" button (proportionally adjusts all weights to sum to 100%)
12. User can edit any position inline:
    - Click weight cell → edit → recalculates shares and $ amount
    - Drag rows to reorder
13. User can delete positions via trash icon
14. User fills portfolio metadata form (name, description, tags, starting capital)
15. User clicks "Create Portfolio" button
16. System validates final state
17. System creates portfolio

**UI Components**:

**1. Ticker Autocomplete Input**:
```typescript
interface AutocompleteOption {
  ticker: string;              // "AAPL"
  name: string;                // "Apple Inc."
  sector: string;              // "Technology"
  market_cap: number;          // 2750000000000
  market_cap_formatted: string; // "$2.75T"
  current_price: number;       // 175.50
  exchange: string;            // "NASDAQ"
}
```

Autocomplete Behavior:
- Minimum 1 character to trigger
- Debounce: 300ms
- Max results: 10
- Sort order: Exact match first, then market cap descending
- Keyboard navigation: Arrow keys to select, Enter to confirm, Esc to close
- Click outside to close

**2. Weight/Dollar Toggle Input**:
```html
<div class="input-group">
  <toggle>
    <option value="weight" selected>Weight (%)</option>
    <option value="dollars">Dollar Amount ($)</option>
  </toggle>
  <input type="number" 
         placeholder="Enter weight (e.g., 30)"
         min="0" 
         max="100" 
         step="0.01" />
</div>
```

**3. Positions Table**:
| Ticker | Name | Sector | Price | Weight (%) | Value ($) | Shares | Actions |
|--------|------|--------|-------|------------|-----------|--------|---------|
| AAPL | Apple Inc. | Technology | $175.50 ⟳ | 30.00 | $30,000.00 | 170.94 | ✎ 🗑 |
| MSFT | Microsoft Corp. | Technology | $335.10 ⟳ | 25.00 | $25,000.00 | 74.59 | ✎ 🗑 |
| **TOTAL** | | | | **55.00%** 🟡 | **$55,000.00** | | |

Table Features:
- Sortable columns (click header)
- Inline editing (click cell with pencil icon)
- Drag-to-reorder rows (drag handle on left)
- Delete with confirmation modal
- Footer row with totals
- Color-coded total weight indicator

**4. Running Total Indicator**:
```
┌─────────────────────────────────────────┐
│ Portfolio Allocation Status             │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ [████████████░░░░░░░░░] 55.0% / 100.0% │
│                                         │
│ ⚠ Add 45% more to reach 100%          │
│ [Auto-Normalize to 100%]               │
└─────────────────────────────────────────┘
```

**Input Data**:

```typescript
interface ManualPosition {
  ticker: string;
  weight?: number;        // 0.0-1.0, exclusive with dollar_amount
  dollar_amount?: number; // Exclusive with weight
  purchase_price?: number; // Optional for P&L tracking
}

interface ManualEntryData {
  positions: ManualPosition[];
  portfolio_metadata: {
    name: string;
    description?: string;
    tags?: string[];
    starting_capital: number; // Required for manual entry
  };
}
```

**Output Data**: Same format as FR-001

**Real-Time Validations**:

1. **Ticker Validation** (on blur or Enter):
   ```json
   {
     "ticker": "AAPL",
     "status": "valid",
     "name": "Apple Inc.",
     "current_price": 175.50,
     "sector": "Technology",
     "last_update": "2025-10-29T20:00:00Z"
   }
   ```

2. **Weight Validation** (on input):
   - If < 0: Show error "Weight must be positive"
   - If > 100: Show error "Weight cannot exceed 100%"
   - If sum > 100: Show warning "Total allocation exceeds 100%"
   - If sum == 100: Show success "Portfolio fully allocated ✓"

3. **Duplicate Ticker Check** (on add):
   ```json
   {
     "code": "DUPLICATE_TICKER",
     "message": "AAPL already exists in portfolio",
     "action": "Replace existing position or cancel"
   }
   ```

**Error Handling**:

**Price Fetch Failure**:
```
UI Display: "AAPL - Price unavailable ⚠ (last updated: 2hrs ago)"
Action: Show stale price with warning, allow user to proceed or retry
```

**API Timeout**:
```
Modal: "Ticker validation is taking longer than usual. 
        Continue waiting or enter ticker manually?"
Buttons: [Wait] [Enter Manually]
```

**Invalid Weight**:
```
Inline Error: "Weight must be between 0% and 100%"
Visual: Red border on input field, shake animation
```

**Priority Justification**: Important (not critical) because power users prefer text or CSV import for speed. However, manual entry is essential for users building small portfolios or learning the system.

---

#### FR-004: Template-Based Portfolio Creation

**Priority**: Desirable  
**Module**: Portfolio Creation  
**Dependencies**: FR-015, FR-016

**Description**:
Offer pre-configured portfolio templates based on common investment strategies (60/40, All Weather, Risk Parity, Sector Rotation, Dividend Growth) with customizable weights and ticker selection.

**User Stories**:
- **US-004-1**: As a beginner investor, I want to start with a proven portfolio allocation, so I don't have to research optimal strategies
- **US-004-2**: As a user, I want to customize a template to fit my risk tolerance, so I can start with best practices but personalize
- **US-004-3**: As a user, I want to see the historical performance of a template, so I can make informed decisions

**Available Templates**:

**Template 1: Classic 60/40 (Stocks/Bonds)**
```json
{
  "template_id": "60-40",
  "name": "Classic 60/40",
  "description": "Traditional balanced portfolio with 60% stocks and 40% bonds. Designed for moderate risk tolerance and long-term wealth preservation.",
  "category": "balanced",
  "risk_level": "moderate",
  "rebalance_frequency": "quarterly",
  "min_investment": 10000,
  "asset_classes": [
    {
      "type": "US Large Cap Stocks",
      "weight": 0.50,
      "default_ticker": "SPY",
      "alternatives": ["VOO", "IVV", "VTI"],
      "rationale": "Core equity exposure with broad diversification"
    },
    {
      "type": "US Small Cap Stocks",
      "weight": 0.10,
      "default_ticker": "IWM",
      "alternatives": ["VB", "IJR", "SCHA"],
      "rationale": "Growth potential from smaller companies"
    },
    {
      "type": "US Aggregate Bonds",
      "weight": 0.30,
      "default_ticker": "AGG",
      "alternatives": ["BND", "VTEB", "SCHZ"],
      "rationale": "Income and stability from investment-grade bonds"
    },
    {
      "type": "International Bonds",
      "weight": 0.10,
      "default_ticker": "BNDX",
      "alternatives": ["IAGG", "IGOV"],
      "rationale": "Geographic diversification in fixed income"
    }
  ],
  "expected_metrics": {
    "annual_return": 0.08,
    "volatility": 0.10,
    "max_drawdown": -0.22,
    "sharpe_ratio": 0.80
  },
  "historical_period": "1990-2025"
}
```

**Template 2: All Weather Portfolio**
```json
{
  "template_id": "all-weather",
  "name": "All Weather Portfolio",
  "description": "Ray Dalio's risk parity approach designed to perform well across different economic environments (growth, recession, inflation, deflation).",
  "category": "risk_parity",
  "risk_level": "moderate-low",
  "rebalance_frequency": "quarterly",
  "min_investment": 10000,
  "asset_classes": [
    {
      "type": "US Stocks",
      "weight": 0.30,
      "default_ticker": "VTI",
      "alternatives": ["ITOT", "SCHB"],
      "rationale": "Growth during prosperity"
    },
    {
      "type": "Long-term Treasuries (20-25yr)",
      "weight": 0.40,
      "default_ticker": "TLT",
      "alternatives": ["VGLT", "SPTL"],
      "rationale": "Protection during deflation, rate cuts"
    },
    {
      "type": "Intermediate Treasuries (7-10yr)",
      "weight": 0.15,
      "default_ticker": "IEF",
      "alternatives": ["VGIT", "SCHR"],
      "rationale": "Moderate duration exposure"
    },
    {
      "type": "Gold",
      "weight": 0.075,
      "default_ticker": "GLD",
      "alternatives": ["IAU", "SGOL", "GLDM"],
      "rationale": "Inflation hedge, currency debasement protection"
    },
    {
      "type": "Commodities",
      "weight": 0.075,
      "default_ticker": "DBC",
      "alternatives": ["PDBC", "GSG", "COMT"],
      "rationale": "Inflation protection, diversification"
    }
  ],
  "expected_metrics": {
    "annual_return": 0.07,
    "volatility": 0.08,
    "max_drawdown": -0.18,
    "sharpe_ratio": 0.90
  }
}
```

**Template 3: Risk Parity**
```json
{
  "template_id": "risk-parity",
  "name": "Risk Parity Portfolio",
  "description": "Equal risk contribution from each asset class. Balances traditional 60/40 bias toward equity risk.",
  "category": "risk_parity",
  "risk_level": "moderate",
  "rebalance_frequency": "monthly",
  "asset_classes": [
    {
      "type": "US Stocks",
      "weight": 0.25,
      "default_ticker": "VTI"
    },
    {
      "type": "International Stocks",
      "weight": 0.15,
      "default_ticker": "VXUS"
    },
    {
      "type": "US Treasuries",
      "weight": 0.30,
      "default_ticker": "IEF"
    },
    {
      "type": "TIPS (Inflation-Protected)",
      "weight": 0.15,
      "default_ticker": "TIP"
    },
    {
      "type": "Commodities",
      "weight": 0.10,
      "default_ticker": "DBC"
    },
    {
      "type": "REITs",
      "weight": 0.05,
      "default_ticker": "VNQ"
    }
  ]
}
```

**Template 4: Sector Rotation**
```json
{
  "template_id": "sector-rotation",
  "name": "Sector Rotation Strategy",
  "description": "Tactical allocation across sectors based on economic cycle positioning.",
  "category": "tactical",
  "risk_level": "high",
  "rebalance_frequency": "monthly",
  "asset_classes": [
    {
      "type": "Technology",
      "weight": 0.25,
      "default_ticker": "XLK"
    },
    {
      "type": "Healthcare",
      "weight": 0.15,
      "default_ticker": "XLV"
    },
    {
      "type": "Financials",
      "weight": 0.15,
      "default_ticker": "XLF"
    },
    {
      "type": "Consumer Discretionary",
      "weight": 0.15,
      "default_ticker": "XLY"
    },
    {
      "type": "Industrials",
      "weight": 0.10,
      "default_ticker": "XLI"
    },
    {
      "type": "Energy",
      "weight": 0.10,
      "default_ticker": "XLE"
    },
    {
      "type": "Utilities",
      "weight": 0.10,
      "default_ticker": "XLU"
    }
  ]
}
```

**Template 5: Dividend Growth**
```json
{
  "template_id": "dividend-growth",
  "name": "Dividend Growth Portfolio",
  "description": "Focus on companies with history of consistent dividend growth. Designed for income generation and capital preservation.",
  "category": "income",
  "risk_level": "moderate-low",
  "rebalance_frequency": "semi-annual",
  "asset_classes": [
    {
      "type": "Dividend Aristocrats",
      "weight": 0.40,
      "default_ticker": "NOBL"
    },
    {
      "type": "High Dividend Yield",
      "weight": 0.25,
      "default_ticker": "VYM"
    },
    {
      "type": "Dividend Growth",
      "weight": 0.20,
      "default_ticker": "VIG"
    },
    {
      "type": "REITs",
      "weight": 0.15,
      "default_ticker": "VNQ"
    }
  ]
}
```

**Functional Flow**:

1. User navigates to "Create Portfolio" → "From Template" tab
2. System displays template cards in grid layout (2-3 per row):
   - Template name
   - Short description (1 sentence)
   - Risk level badge (Low/Moderate/High)
   - Expected annual return (e.g., "~8%")
   - Visual icon/illustration
   - "Learn More" and "Use Template" buttons
3. User clicks "Learn More" on a template
4. System opens modal/drawer with detailed template information:
   - Full description
   - Asset allocation chart (pie or donut)
   - Historical performance chart (if available)
   - Key metrics table (return, volatility, Sharpe, max drawdown)
   - List of holdings with rationale
   - "Use This Template" button
5. User clicks "Use This Template"
6. System displays template customization interface:
   - **Section 1: Asset Allocation (Sliders)**
     - One slider per asset class
     - Default value pre-filled
     - Total allocation bar at bottom (must sum to 100%)
     - Reset to defaults button
   - **Section 2: Ticker Selection (Dropdowns)**
     - For each asset class:
       - Label: "US Large Cap Stocks"
       - Dropdown with default and alternatives
       - Default: SPY (S&P 500 ETF)
       - Alternatives: VOO, IVV, VTI
       - Option to enter custom ticker
   - **Section 3: Additional Positions (Optional)**
     - "Add Custom Position" button
     - Opens mini form to add tickers beyond template
7. User adjusts weights using sliders
8. System validates in real-time:
   - Total must equal 100%
   - Show warning if != 100%
   - Disable "Create" button until valid
9. User selects preferred tickers from dropdowns
10. System fetches current prices for selected tickers
11. System displays preview summary:
    - Portfolio allocation pie chart
    - Expected metrics (calculated from historical data)
    - Estimated initial value breakdown
12. User fills portfolio metadata (name, description, tags, starting capital)
13. User clicks "Create Portfolio"
14. System creates portfolio with selected tickers and weights

**UI Components**:

**Template Card**:
```
┌────────────────────────────────┐
│  [ICON]  Classic 60/40         │
│                                │
│  Traditional stocks/bonds      │
│  balanced allocation           │
│                                │
│  Risk: ● Moderate              │
│  Return: ~8% annually          │
│                                │
│  [Learn More] [Use Template]   │
└────────────────────────────────┘
```

**Weight Adjustment Sliders**:
```
US Large Cap Stocks: 50%
[━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 
Min: 0%    Default: 50%    Max: 100%

US Small Cap Stocks: 10%
[━━━━━━░░░░░░░░░░░░░░░░░░░░░░░░░░]

Total Allocation: 100% ✓
[━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
```

**Input Data**:

```typescript
interface TemplateSelectionData {
  template_id: string;           // "60-40", "all-weather", etc.
  weight_overrides?: {           // Optional weight adjustments
    [asset_class: string]: number;
  };
  ticker_selections: {           // User-selected tickers
    [asset_class: string]: string;
  };
  additional_positions?: Array<{ // Optional custom additions
    ticker: string;
    weight: number;
  }>;
  portfolio_metadata: {
    name: string;
    description?: string;
    tags?: string[];
    starting_capital: number;
  };
}
```

**Example Request**:
```json
{
  "template_id": "60-40",
  "weight_overrides": {
    "US Large Cap Stocks": 0.55,
    "US Small Cap Stocks": 0.05
  },
  "ticker_selections": {
    "US Large Cap Stocks": "VOO",
    "US Small Cap Stocks": "VB",
    "US Aggregate Bonds": "AGG",
    "International Bonds": "BNDX"
  },
  "portfolio_metadata": {
    "name": "My 60/40 Portfolio",
    "description": "Modified 60/40 with higher large-cap allocation",
    "tags": ["template", "balanced"],
    "starting_capital": 50000
  }
}
```

**Output Data**: Same format as FR-001

**Validation Rules**:

1. **Weight Validation**:
   - Sum must equal 1.0 (100%)
   - Individual weights: 0.0 <= weight <= 1.0
   - If user overrides weights, must still sum to 100%

2. **Ticker Validation**:
   - All selected tickers must be valid
   - Must return current price data
   - Cannot have duplicates across asset classes

**Error Handling**:

```json
{
  "code": "WEIGHTS_INVALID_SUM",
  "message": "Total allocation is 105%. Adjust weights to equal 100%.",
  "current_total": 1.05,
  "difference": 0.05
}
```

UI: Show warning banner, highlight total bar in red, disable "Create" button

```json
{
  "code": "INVALID_TICKER_SELECTION",
  "message": "Selected ticker 'XYZ' for US Bonds is invalid",
  "asset_class": "US Aggregate Bonds",
  "selected_ticker": "XYZ"
}
```

UI: Show error inline below dropdown, highlight in red

**Priority Justification**: Desirable (not critical) because templates are helpful for beginners and save time, but experienced users typically have their own allocations. Nice-to-have for MVP, essential for broader adoption.

---

#### FR-005: Clone Existing Portfolio

**Priority**: Important  
**Module**: Portfolio Management  
**Dependencies**: FR-006 (Portfolio List), FR-002

**Description**:
Allow users to duplicate an existing portfolio with option to modify name, weights, and settings. Enables quick creation of variations for testing or client management.

**User Stories**:
- **US-005-1**: As a portfolio manager, I want to create variations of existing portfolios for testing different strategies, without rebuilding from scratch
- **US-005-2**: As a user, I want to clone a portfolio to create a similar one for a different account with slight modifications
- **US-005-3**: As a user, I want to preserve the original portfolio while experimenting with changes

**Functional Flow**:

1. **Entry Points** (multiple ways to initiate cloning):
   - Method A: From Portfolio List view
     - User clicks "..." menu button on portfolio card
     - Dropdown menu shows "Clone Portfolio" option
     - User clicks "Clone Portfolio"
   - Method B: From Portfolio Detail view
     - User clicks "Actions" dropdown in header
     - Selects "Clone Portfolio"
   - Method C: From Portfolio Analysis view
     - After reviewing performance/risk
     - "Clone This Portfolio" button in toolbar

2. System opens "Clone Portfolio" modal/page with:
   - Source portfolio information (read-only section):
     - Name
     - Current value
     - Position count
     - Creation date
   - Cloning options (user inputs):
     - New portfolio name (pre-filled: "Copy of {original_name}")
     - Clone mode (radio buttons):
       - ○ Exact Clone: Copy all weights exactly
       - ○ Rebalance to Equal: Reset all positions to equal weights
       - ○ Custom Weights: Copy positions but allow weight editing
     - Settings to clone (checkboxes):
       - ☑ Rebalancing settings
       - ☑ Risk constraints
       - ☑ Position constraints
       - ☑ Tax settings
       - ☑ Filters (ESG, etc.)
     - New starting capital (optional, default: same as original)

3. User selects clone mode:

   **Mode A: Exact Clone**
   - All positions copied with identical weights
   - All settings copied
   - Only name and starting capital editable

   **Mode B: Rebalance to Equal**
   - All positions copied
   - Weights reset: 1/N for N positions
   - Example: 5 positions → each gets 20%
   - Settings optionally copied

   **Mode C: Custom Weights**
   - System shows editable positions table:
     | Ticker | Name | Current Weight | New Weight |
     |--------|------|----------------|------------|
     | AAPL | Apple | 30% | [editable] |
     | MSFT | Microsoft | 25% | [editable] |
   - User can:
     - Edit weights
     - Remove positions (X button)
     - Add new positions (+ button)
   - System validates weights sum to 100%

4. User fills new portfolio metadata:
   - Name (required, must be unique)
   - Description (optional, default: "Cloned from {original_name}")
   - Tags (optional, default: copy original + add "cloned" tag)
   - Starting capital (optional, default: match original)

5. User clicks "Clone Portfolio" button

6. System validates:
   - Name unique
   - Weights sum to 100% (if custom mode)
   - All tickers still valid (prices available)

7. System creates new portfolio:
   - Generates new UUID
   - Copies positions with selected weights
   - Copies settings if selected
   - Sets metadata fields:
     - `cloned_from`: original portfolio ID
     - `clone_date`: timestamp
     - `clone_mode`: exact|equal|custom

8. System displays success message:
   - "Portfolio cloned successfully!"
   - Shows comparison summary:
     - Original: {name} - $X,XXX
     - New: {name} - $Y,YYY
   - Buttons: [View New Portfolio] [Clone Another] [Close]

9. User clicks "View New Portfolio"

10. System redirects to Portfolio Detail view for new portfolio

**Cloning Options**:

**Option 1: Clone Exact Weights (Default)**
```json
{
  "mode": "exact",
  "preserve_weights": true,
  "preserve_settings": true
}
```
Result: Perfect copy, different ID and name

**Option 2: Clone and Rebalance to Equal**
```json
{
  "mode": "equal",
  "preserve_weights": false,
  "preserve_settings": true
}
```
Result: Same tickers, equal weights (1/N)

**Option 3: Clone Structure Only**
```json
{
  "mode": "structure",
  "preserve_weights": false,
  "preserve_settings": false,
  "custom_weights": {
    "AAPL": 0.35,
    "MSFT": 0.30,
    "GOOGL": 0.20,
    "AMZN": 0.15
  }
}
```
Result: Same tickers, custom weights, default settings

**Input Data**:

```typescript
interface ClonePortfolioRequest {
  source_portfolio_id: string;   // UUID of portfolio to clone
  clone_mode: "exact" | "equal" | "custom";
  new_metadata: {
    name: string;                // Required, must be unique
    description?: string;
    tags?: string[];
    starting_capital?: number;   // If null, copies original
  };
  settings_to_clone: {
    rebalancing: boolean;
    risk_management: boolean;
    position_constraints: boolean;
    tax_settings: boolean;
    filters: boolean;
  };
  custom_weights?: {             // Only if mode == "custom"
    [ticker: string]: number;
  };
  positions_to_exclude?: string[]; // Optional: tickers to not clone
  positions_to_add?: Array<{       // Optional: new tickers to add
    ticker: string;
    weight: number;
  }>;
}
```

**Example Request (Exact Clone)**:
```json
{
  "source_portfolio_id": "550e8400-e29b-41d4-a716-446655440000",
  "clone_mode": "exact",
  "new_metadata": {
    "name": "Tech Leaders - Copy",
    "description": "Cloned from Tech Leaders for testing",
    "tags": ["tech", "growth", "cloned"],
    "starting_capital": 100000
  },
  "settings_to_clone": {
    "rebalancing": true,
    "risk_management": true,
    "position_constraints": true,
    "tax_settings": false,
    "filters": true
  }
}
```

**Example Request (Custom Weights)**:
```json
{
  "source_portfolio_id": "550e8400-e29b-41d4-a716-446655440000",
  "clone_mode": "custom",
  "new_metadata": {
    "name": "Tech Leaders - Hedged",
    "starting_capital": 100000
  },
  "custom_weights": {
    "AAPL": 0.25,
    "MSFT": 0.25,
    "GOOGL": 0.20,
    "AMZN": 0.15,
    "NVDA": 0.05,
    "SPY": 0.10  // Added hedge
  },
  "positions_to_add": [
    {
      "ticker": "SPY",
      "weight": 0.10
    }
  ],
  "settings_to_clone": {
    "rebalancing": true,
    "risk_management": false,
    "position_constraints": true,
    "tax_settings": false,
    "filters": false
  }
}
```

**Output Data**:

```json
{
  "new_portfolio_id": "770e8400-e29b-41d4-a716-446655440002",
  "name": "Tech Leaders - Copy",
  "cloned_from": "550e8400-e29b-41d4-a716-446655440000",
  "clone_date": "2025-10-29T17:00:00Z",
  "clone_mode": "exact",
  "position_count": 5,
  "starting_value": 100000.00,
  "comparison": {
    "original": {
      "name": "Tech Leaders",
      "value": 125340.50,
      "positions": 5
    },
    "new": {
      "name": "Tech Leaders - Copy",
      "value": 100000.00,
      "positions": 5
    },
    "differences": {
      "weight_changes": [],      // Empty for exact clone
      "positions_added": [],
      "positions_removed": []
    }
  },
  "message": "Portfolio cloned successfully"
}
```

**Validation Rules**:

1. **Source Portfolio**:
   - Must exist
   - User must have permission to view
   - Cannot clone archived portfolios

2. **New Name**:
   - Required
   - 1-100 characters
   - Must be unique per user
   - Cannot match source portfolio name

3. **Weights** (if custom mode):
   - Must sum to 1.0 (±0.0001)
   - Individual weights: 0.0 < weight <= 1.0
   - At least 1 position

4. **Tickers** (if adding new):
   - Must be valid
   - Must return current price
   - Cannot duplicate existing tickers

**Error Handling**:

```json
{
  "code": "SOURCE_NOT_FOUND",
  "message": "Portfolio with ID '550e84...' not found or you don't have permission to access it"
}
```

```json
{
  "code": "NAME_CONFLICT",
  "message": "Portfolio name 'Tech Leaders - Copy' already exists",
  "suggestion": "Tech Leaders - Copy (2)"
}
```

```json
{
  "code": "INVALID_WEIGHTS",
  "message": "Custom weights sum to 0.98. Total must equal 1.0",
  "current_sum": 0.98,
  "expected_sum": 1.0
}
```

```json
{
  "code": "TICKER_NO_LONGER_VALID",
  "message": "Ticker 'XYZ' from source portfolio is no longer available (delisted)",
  "affected_ticker": "XYZ",
  "action": "Remove ticker or proceed without it?"
}
```

UI Treatment:
- Show warning modal: "Ticker XYZ is no longer valid. Remove it to proceed?"
- Buttons: [Remove and Continue] [Cancel Clone]

**Priority Justification**: Important because portfolio managers often need to create variations for testing or client customization. Saves significant time vs manual recreation.

---

---

### 2.2 Portfolio Management

#### FR-006: Portfolio List View

**Priority**: Critical  
**Module**: Portfolio Management  
**Dependencies**: None

**Description**:
Display all user portfolios in sortable/filterable table or card grid format with key metrics, search functionality, and bulk actions.

**User Stories**:
- **US-006-1**: As a portfolio manager, I want to see all my portfolios at a glance with current performance
- **US-006-2**: As a user, I want to quickly find specific portfolios using search and filters
- **US-006-3**: As a user, I want to sort portfolios by performance to identify top/bottom performers

**Display Formats**:

**Format A: Table View** (Default for many portfolios)
| Name | Value | Positions | Daily Δ | MTD | YTD | Risk | Updated | Actions |
|------|-------|-----------|---------|-----|-----|------|---------|---------|
| Tech Leaders | $125,340 | 5 | +0.99% | +5.23% | +18.45% | High | 2h ago | ⋮ |
| Balanced 60/40 | $78,230 | 8 | +0.34% | +2.45% | +8.92% | Moderate | 2h ago | ⋮ |

**Format B: Card Grid View** (Default for few portfolios)
```
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ Tech Leaders           High │  │ Balanced 60/40      Moderate│
│ $125,340    ↑ +$1,234 (0.99%)│  │ $78,230     ↑ +$265 (0.34%) │
│                             │  │                             │
│ 5 positions                 │  │ 8 positions                 │
│ MTD: +5.23%  YTD: +18.45%  │  │ MTD: +2.45%  YTD: +8.92%   │
│ ──────────── sparkline ──── │  │ ──────────── sparkline ──── │
│ [Analyze] [Edit] [⋮]       │  │ [Analyze] [Edit] [⋮]       │
└─────────────────────────────┘  └─────────────────────────────┘
```

**Display Columns** (Table View):
1. **Portfolio Name**: Sortable, searchable, clickable to open detail
2. **Current Value**: Total portfolio value in base currency
3. **# Positions**: Count of holdings
4. **Daily Change**: Absolute ($) and percentage (%), color-coded
5. **MTD Return**: Month-to-date return percentage
6. **YTD Return**: Year-to-date return percentage
7. **Risk Level**: Badge (Low/Moderate/High/Extreme) based on volatility
8. **Last Updated**: Relative timestamp (2h ago, yesterday, etc.)
9. **Actions**: Dropdown menu with options

**Functional Features**:

**1. Sorting**:
- Click column header to sort
- First click: Descending
- Second click: Ascending
- Third click: Reset to default (creation date desc)
- Visual indicator: ▼ or ▲ arrow on active column
- Default sort: Last Updated (newest first)

**2. Search**:
- Search box in header: "Search portfolios..."
- Searches in: portfolio name, description, tags
- Real-time filtering (debounced 300ms)
- Clear button (X) to reset search
- Shows result count: "Showing 3 of 15 portfolios"

**3. Filters**:
- **Risk Level**: Checkboxes for Low/Moderate/High/Extreme
- **Performance**: 
  - Positive YTD (checkbox)
  - Negative YTD (checkbox)
  - Custom range slider: -50% to +100%
- **Tags**: Multi-select dropdown
- **Date Range**: Created between [date] and [date]
- **Position Count**: Min/max slider
- **Apply Filters** and **Reset** buttons
- Active filters shown as chips above table

**4. Bulk Actions**:
- Checkbox column (first column)
- "Select All" checkbox in header
- Selected count indicator: "3 portfolios selected"
- Bulk action buttons (enabled when ≥1 selected):
  - Export Multiple (combined Excel)
  - Delete Multiple (confirmation modal)
  - Archive Multiple
  - Tag Multiple (add tags in bulk)

**5. Pagination**:
- 50 portfolios per page default
- Page size selector: 25/50/100 per page
- Navigation: ‹ 1 2 3 4 5 › buttons
- Jump to page input
- "Showing 1-50 of 234 portfolios"

**6. View Toggle**:
- Icon buttons: [☷ Table] [⊞ Cards]
- Persists preference in user settings

**Actions Per Portfolio** (Dropdown Menu):
- 📊 **Analyze**: Navigate to Portfolio Analysis page
- ✏️ **Edit**: Open Portfolio Management detail view
- 📋 **Clone**: Create copy (FR-005)
- 📤 **Export**: Download holdings
  - Submenu: CSV / Excel / JSON
- 📄 **Generate Report**: Navigate to report builder
- 📁 **Archive**: Move to archived (soft delete)
- 🗑️ **Delete**: Permanent delete (confirmation modal)

**API Endpoint**:

```
GET /api/portfolios

Query Parameters:
- page (integer, default 1)
- limit (integer, default 50, max 100)
- sort (string, default "last_updated")
  Options: name, value, position_count, daily_change, 
           mtd_return, ytd_return, risk_level, 
           creation_date, last_updated
- order (string, default "desc"): asc|desc
- search (string): Search query
- risk_level (array of strings): ["low", "moderate"]
- tags (array of strings): ["tech", "growth"]
- ytd_min (float): Min YTD return filter
- ytd_max (float): Max YTD return filter
- created_after (date): YYYY-MM-DD
- created_before (date): YYYY-MM-DD
```

**Response Format**:
```json
{
  "portfolios": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Tech Leaders",
      "description": "Large-cap technology portfolio",
      "tags": ["tech", "growth"],
      "value": 125340.50,
      "starting_capital": 100000.00,
      "total_return": 0.2534,
      "position_count": 5,
      "daily_change_dollars": 1234.56,
      "daily_change_pct": 0.0099,
      "mtd_return": 0.0523,
      "ytd_return": 0.1845,
      "risk_level": "high",
      "volatility_annual": 0.2301,
      "sharpe_ratio": 1.35,
      "creation_date": "2025-01-15T10:00:00Z",
      "last_modified": "2025-10-29T14:30:00Z",
      "last_price_update": "2025-10-29T20:00:00Z",
      "sparkline_30d": [1.0, 1.01, 0.99, "..."]
    }
  ],
  "pagination": {
    "current_page": 1,
    "total_pages": 5,
    "total_items": 234,
    "items_per_page": 50,
    "has_next": true,
    "has_prev": false
  },
  "filters_applied": {
    "search": "tech",
    "risk_level": ["high"],
    "tags": []
  }
}
```

**Output States**:

**Empty State** (No portfolios):
```
┌────────────────────────────────────┐
│         [Folder Icon]              │
│   No portfolios yet                │
│                                    │
│   Create your first portfolio to   │
│   start tracking performance       │
│                                    │
│   [+ Create Portfolio]             │
└────────────────────────────────────┘
```

**No Results State** (Search/filter no matches):
```
┌────────────────────────────────────┐
│         [Search Icon]              │
│   No portfolios found              │
│                                    │
│   Try adjusting your search or     │
│   filters to find portfolios       │
│                                    │
│   [Clear Filters]                  │
└────────────────────────────────────┘
```

**Loading State**:
```
┌────────────────────────────────────┐
│   [Spinner]  Loading portfolios... │
│   ▢▢▢▢▢▢▢▢ Skeleton cards          │
└────────────────────────────────────┘
```

**Error State**:
```
┌────────────────────────────────────┐
│         [Error Icon]               │
│   Failed to load portfolios        │
│                                    │
│   Unable to connect to server.     │
│   Please try again.                │
│                                    │
│   [Retry]                          │
└────────────────────────────────────┘
```

**Priority Justification**: Critical because this is the primary navigation hub for all portfolio management activities.

---

#### FR-007: Portfolio Detail View and Editing

**Priority**: Critical  
**Module**: Portfolio Management  
**Dependencies**: FR-015, FR-016

**Description**:
Provide detailed view of single portfolio with all holdings, metadata, and inline editing capabilities for weights, shares, prices, and settings.

**User Stories**:
- **US-007-1**: As a user, I want to see all my holdings with current prices and P&L in one place
- **US-007-2**: As a user, I want to edit position sizes without deleting and recreating
- **US-007-3**: As a user, I want to update portfolio settings like rebalancing frequency

**Page Layout**:

```
┌─────────────────────────────────────────────────────────────┐
│ ← Back to Portfolios                          [⚙ Settings]  │
├─────────────────────────────────────────────────────────────┤
│ HEADER SECTION                                              │
│ Tech Leaders ✎                                              │
│ Large-cap technology portfolio ✎                            │
│ [tech] [growth] [+]                                         │
│                                                             │
│ Created: Jan 15, 2025  |  Modified: Oct 29, 2025          │
├─────────────────────────────────────────────────────────────┤
│ VALUATION SUMMARY                                           │
│ ┌──────────┬──────────┬──────────┬──────────┐            │
│ │ Value    │ Return   │ Daily Δ  │ Risk     │            │
│ │ $125,340 │ +25.34%  │ +$1,234  │ High     │            │
│ └──────────┴──────────┴──────────┴──────────┘            │
├─────────────────────────────────────────────────────────────┤
│ HOLDINGS TABLE                                              │
│ [Add Position] [Rebalance] [Update Prices] [⋮ More]       │
│                                                             │
│ Ticker  Name     Sector  Shares  Price   Value  Weight ... │
│ AAPL    Apple    Tech    171.4   $175.50 $30,085 30.0% ... │
│ MSFT    Microsoft Tech   74.7    $335.10 $25,023 25.0% ... │
│ ...                                                         │
│                                                             │
│ TOTAL                                   $125,340  100.0%   │
├─────────────────────────────────────────────────────────────┤
│ WEIGHT ADJUSTMENT PANEL (collapsible)                       │
│ [Interactive pie chart] [Sliders] [Auto-normalize]         │
└─────────────────────────────────────────────────────────────┘
```

**Header Section**:

**Editable Fields** (inline editing):
- **Portfolio Name**: Click to edit, Enter to save, Esc to cancel
  - Validation: 1-100 chars, unique per user
  - Shows save indicator: "Saving..." → "Saved ✓"
- **Description**: Click to edit (expands to textarea)
  - Validation: max 500 chars
  - Character counter shown during edit
- **Tags**: Chip input
  - Click (+) to add new tag
  - Click (X) on chip to remove
  - Autocomplete from existing tags
  - Max 10 tags

**Read-Only Metadata**:
- Created date (formatted: "Jan 15, 2025")
- Last modified date (relative: "Updated 2 hours ago")
- Portfolio ID (expandable technical details)

**Valuation Summary Cards**:

Card 1: **Current Value**
- Large display: $125,340.50
- Subtitle: vs Starting $100,000.00
- Color: Neutral

Card 2: **Total Return**
- Large display: +25.34% ($25,340)
- Since inception
- Color: Green (positive) or Red (negative)

Card 3: **Daily Change**
- Large display: +$1,234.56 (+0.99%)
- Since previous close
- Color: Green/Red
- Small sparkline (last 7 days)

Card 4: **Risk Level**
- Badge: High (color-coded)
- Volatility: 23.01% annual
- Max Drawdown: -15.23%

**Holdings Table**:

**Columns**:
1. **Ticker**: Read-only, clickable for position detail
2. **Company Name**: Read-only
3. **Sector**: Read-only
4. **Shares**: Editable (click to edit)
5. **Current Price**: Read-only, with timestamp tooltip
6. **Purchase Price**: Editable (optional, for P&L)
7. **Current Value**: Calculated (shares × current_price)
8. **Weight (%)**: Editable
9. **Unrealized P&L ($)**: Calculated (if purchase price exists)
10. **Unrealized P&L (%)**: Calculated
11. **Daily Change ($)**: Calculated
12. **Daily Change (%)**: Calculated
13. **Actions**: [✎ Edit] [🗑 Delete]

**Table Features**:
- **Sortable**: Click any column header
- **Inline Editing**:
  - Click shares cell → input appears → type → Enter to save
  - Click weight cell → input appears → auto-recalculates shares
  - Changes saved immediately with optimistic UI update
- **Color Coding**:
  - Positive P&L: Green background (light) or green text
  - Negative P&L: Red background (light) or red text
- **Footer Row**: Shows totals
  - Total Value: sum of all positions
  - Total Weight: should always be 100.00%
- **Expandable Rows**: Click row to expand additional details
  - 52-week range
  - P/E ratio
  - Dividend yield
  - Beta
  - Purchase date
  - Notes field (editable)

**Action Buttons** (above table):

**Primary Actions**:
- **Add Position**: Opens modal/drawer to add new ticker
  - Uses same flow as FR-003 (manual entry)
- **Rebalance to Target**: Calculates trades to restore target weights
  - Shows preview modal with trade list
  - Option to execute or cancel
- **Update Prices**: Manual refresh from data provider
  - Shows loading indicator
  - Displays timestamp of last update

**Secondary Actions** (dropdown "More"):
- **Export Holdings**:
  - CSV (simple: ticker, shares, value)
  - Excel (formatted with formulas)
  - JSON (full data structure)
- **Generate Report**: Navigate to report builder with portfolio pre-selected
- **Optimize**: Navigate to optimization page with portfolio loaded
- **Run Stress Test**: Navigate to scenario analysis
- **View Analytics**: Navigate to Portfolio Analysis page
- **Settings**: Open settings panel
- **Duplicate**: Clone portfolio (FR-005)
- **Archive**: Soft delete
- **Delete**: Permanent delete with confirmation

**Weight Adjustment Panel** (Expandable):

Toggle: [Collapse ▲] / [Expand ▼]

**Layout when expanded**:

Left Side (60%): **Interactive Pie Chart**
- Draggable slices to adjust weights
- Hover shows ticker, weight, value
- Click slice to select (highlights in table)

Right Side (40%): **Weight Sliders**
```
AAPL  [━━━━━━━━━━━━━━░░░░] 30.0%  ± buttons
MSFT  [━━━━━━━━━━░░░░░░░░░] 25.0%  ± buttons
GOOGL [━━━━━━━░░░░░░░░░░░░] 20.0%  ± buttons
AMZN  [━━━━━░░░░░░░░░░░░░░] 15.0%  ± buttons
NVDA  [━━░░░░░░░░░░░░░░░░░] 10.0%  ± buttons
────────────────────────────────────
TOTAL                        100.0%  ✓
```

**Controls**:
- Each slider adjusts weight 0-100%
- ± buttons for precise adjustments (±0.1%)
- Total indicator at bottom (color-coded)
- **Auto-Normalize** button: Proportionally adjusts all to sum to 100%
- **Calculate Trades** button: Shows what trades needed to reach new weights
- **Apply Changes** button: Saves new target weights
- **Reset** button: Reverts to current weights

**Calculate Trades Preview**:
```
┌──────────────────────────────────────────────┐
│ Required Trades to Rebalance                 │
├──────────────────────────────────────────────┤
│ SELL  14 shares AAPL  ~$2,457  (-2.0 pp)    │
│ BUY    5 shares MSFT  ~$1,676  (+1.5 pp)    │
│ BUY    3 shares GOOGL ~$414    (+0.5 pp)    │
├──────────────────────────────────────────────┤
│ Est. Transaction Costs: $4.55                │
│ Net Cash Flow: $0                            │
│                                              │
│ [Apply Rebalance] [Cancel]                  │
└──────────────────────────────────────────────┘
```

**Inline Editing Behavior**:

**Edit Shares**:
1. User clicks shares cell (e.g., "171.4")
2. Cell becomes input field with current value
3. User types new value (e.g., "200")
4. User presses Enter or clicks outside
5. System validates: must be > 0
6. System recalculates:
   - New value = shares × current_price
   - New weight = value / total_portfolio_value
7. System updates row immediately (optimistic)
8. System saves to backend (async)
9. Shows save indicator: ↻ → ✓

**Edit Weight**:
1. User clicks weight cell (e.g., "30.0%")
2. Cell becomes input with current value
3. User types new percentage (e.g., "35")
4. System validates: 0-100 range
5. System recalculates:
   - New value = weight × total_portfolio_value
   - New shares = value / current_price
6. System checks if sum of all weights != 100%
7. Shows warning if != 100%: "Total allocation now 102%. Normalize?"
8. Updates row, saves to backend

**Edit Purchase Price**:
1. User clicks purchase price cell (e.g., "$145.32")
2. Input appears
3. User enters price
4. System validates: must be > 0
5. System recalculates P&L fields:
   - Unrealized P&L $ = (current - purchase) × shares
   - Unrealized P&L % = (current - purchase) / purchase
6. Saves immediately

**API Endpoints**:

```
GET /api/portfolios/{portfolio_id}
PUT /api/portfolios/{portfolio_id}
PATCH /api/portfolios/{portfolio_id}
DELETE /api/portfolios/{portfolio_id}
PATCH /api/portfolios/{portfolio_id}/positions/{ticker}
POST /api/portfolios/{portfolio_id}/positions
DELETE /api/portfolios/{portfolio_id}/positions/{ticker}
POST /api/portfolios/{portfolio_id}/rebalance
POST /api/portfolios/{portfolio_id}/prices/refresh
```

**Response Example** (GET portfolio):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Tech Leaders",
  "description": "Large-cap technology portfolio",
  "tags": ["tech", "growth"],
  "starting_capital": 100000.00,
  "base_currency": "USD",
  "creation_date": "2025-01-15T10:00:00Z",
  "last_modified": "2025-10-29T14:30:00Z",
  "last_price_update": "2025-10-29T20:00:00Z",
  
  "valuation": {
    "total_value": 125340.50,
    "total_return_dollars": 25340.50,
    "total_return_pct": 0.2534,
    "daily_change_dollars": 1234.56,
    "daily_change_pct": 0.0099
  },
  
  "risk_summary": {
    "level": "high",
    "annual_volatility": 0.2301,
    "max_drawdown": -0.1523,
    "sharpe_ratio": 1.35
  },
  
  "positions": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "sector": "Technology",
      "industry": "Consumer Electronics",
      "asset_class": "stock",
      "shares": 171.43,
      "weight_target": 0.30,
      "weight_current": 0.2998,
      "current_price": 175.50,
      "previous_close": 174.20,
      "purchase_price": 145.32,
      "purchase_date": "2024-01-15",
      "current_value": 30085.07,
      "cost_basis": 24912.58,
      "unrealized_pl_dollars": 5172.49,
      "unrealized_pl_pct": 0.2076,
      "daily_change_dollars": 222.86,
      "daily_change_pct": 0.0075,
      "fundamental_data": {
        "market_cap": 2750000000000,
        "pe_ratio": 28.5,
        "dividend_yield": 0.0045,
        "beta": 1.21,
        "week_52_high": 199.62,
        "week_52_low": 124.17
      },
      "notes": ""
    }
  ],
  
  "settings": {
    "rebalancing": {
      "enabled": true,
      "frequency": "quarterly",
      "threshold_pct": 5.0,
      "auto_rebalance": false
    }
  }
}
```

**Validation Rules**:
- Name: 1-100 chars, unique per user
- Description: max 500 chars
- Tags: max 10 tags, max 30 chars each
- Shares: must be > 0
- Purchase price: must be > 0 if provided
- Weights: sum must equal 1.0 (±0.0001)

**Error Handling**:

**Save Failure** (network error):
```
Toast notification (top-right):
"Failed to save changes. Retry?"
[Retry] [Cancel]
```

**Validation Error** (invalid shares):
```
Inline error below input:
"Shares must be greater than 0"
Input border: red
```

**Concurrent Edit** (another user/session modified):
```
Modal:
"This portfolio was modified by another session.
 Current version: Updated 5 minutes ago
 Your version: Based on 10 minutes ago

 [Reload Latest] [Overwrite with My Changes]"
```

**Priority Justification**: Critical because this is where users spend most time managing their portfolios.

---

### 2.3 Portfolio Analytics

#### FR-008: Performance Metrics Calculation (70+ Metrics)

**Priority**: Critical  
**Module**: Analytics Engine  
**Dependencies**: FR-016 (Price Data)

**Description**:
Calculate comprehensive portfolio analytics including 18 performance metrics, 22 risk metrics, 15 risk-adjusted ratios, and 15 market-related metrics. All calculations must be accurate, transparent, and reproducible.

**User Stories**:
- **US-008-1**: As a portfolio manager, I want to see all key metrics at a glance to assess portfolio health
- **US-008-2**: As an analyst, I want calculation transparency to verify methodology
- **US-008-3**: As a user, I want to compare my portfolio metrics against benchmarks

**Metrics Categories**:

**A. PERFORMANCE METRICS (18)**

1. **Total Return**
   - Formula: (End Value - Start Value) / Start Value
   - Interpretation: Overall gain/loss over period
   - Example: 0.2534 = +25.34%

2. **CAGR (Compound Annual Growth Rate)**
   - Formula: ((End Value / Start Value)^(1/Years) - 1)
   - Interpretation: Annualized return if compounded
   - Example: 0.3125 = 31.25% per year

3. **Annualized Return**
   - Formula: (1 + Total Return)^(252 / Trading Days) - 1
   - Interpretation: Average yearly return (arithmetic)
   - Assumption: 252 trading days per year

4-6. **Period Returns**
   - YTD (Year-to-Date): Return from Jan 1 to today
   - MTD (Month-to-Date): Return from month start
   - QTD (Quarter-to-Date): Return from quarter start

7-12. **Fixed Period Returns**
   - 1M, 3M, 6M, 1Y, 3Y, 5Y returns
   - Looking back N months/years from current date
   - Null if insufficient data

13-14. **Best/Worst Month**
   - Best Month: Highest monthly return in period
   - Worst Month: Lowest monthly return in period
   - Includes month identifier (e.g., "Aug 2024")

15. **Win Rate**
   - Formula: Count(Positive Periods) / Count(Total Periods)
   - Can calculate for daily/weekly/monthly periods
   - Example: 0.6471 = 64.71% of days were positive

16. **Payoff Ratio**
   - Formula: Average Win / Average Loss (absolute value)
   - Interpretation: How much you make vs lose on average
   - Example: 1.85 = wins are 1.85x larger than losses

17. **Profit Factor**
   - Formula: Gross Profits / Gross Losses
   - Interpretation: Total gains vs total losses
   - Example: 2.34 = made $2.34 for every $1 lost

18. **Expectancy**
   - Formula: (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
   - Interpretation: Expected value per period
   - Example: 0.0052 = expect +0.52% per day

**B. RISK METRICS (22)**

19-22. **Volatility (Multiple Timeframes)**
   - Daily: Std dev of daily returns
   - Weekly: Std dev of weekly returns
   - Monthly: Std dev of monthly returns
   - Annual: Daily vol × sqrt(252)
   - Example: 0.2301 = 23.01% annual volatility

23. **Maximum Drawdown**
   - Formula: Max((Peak - Trough) / Peak)
   - Interpretation: Largest peak-to-trough decline
   - Example: -0.1523 = -15.23% worst drawdown
   - Includes: Date of peak, date of trough

24. **Current Drawdown**
   - Formula: (Current Price - Recent Peak) / Recent Peak
   - Interpretation: How far below recent high
   - Example: -0.0234 = -2.34% below peak

25. **Average Drawdown**
   - Formula: Mean of all drawdown periods
   - Interpretation: Typical decline magnitude
   - Example: -0.0456 = -4.56% average

26-27. **Drawdown Duration**
   - Max Duration: Longest time in drawdown (days)
   - Avg Duration: Average time in drawdown (days)
   - Useful for: Understanding recovery time

28. **Recovery Time**
   - Days to recover from max drawdown to new high
   - Null if not yet recovered
   - Example: 28 days to recover from -15% drawdown

29. **Ulcer Index**
   - Formula: sqrt(Sum of Squared Drawdowns / N)
   - Interpretation: Depth × duration of drawdowns
   - Lower is better (less stress)

30. **Pain Index**
   - Formula: Sum of all negative returns
   - Interpretation: Cumulative downside
   - Used in: Gain-Pain Ratio

31-33. **VaR (Value at Risk) - Multiple Confidence Levels**
   - VaR 90%: 10% chance of losing this much in 1 day
   - VaR 95%: 5% chance
   - VaR 99%: 1% chance
   - Methods: Historical, Parametric, Monte Carlo, Cornish-Fisher
   - Example: VaR 95% = -0.0278 = 5% chance of losing > 2.78% in 1 day

34-36. **CVaR (Conditional VaR / Expected Shortfall)**
   - CVaR 90%, 95%, 99%
   - Interpretation: Average loss IF VaR threshold exceeded
   - Example: CVaR 95% = -0.0342 = if in worst 5%, expect to lose 3.42%

37. **Downside Deviation**
   - Formula: Std dev of returns below target (usually 0)
   - Interpretation: Volatility of negative returns only
   - Used in: Sortino Ratio

38. **Semi-Deviation**
   - Similar to downside deviation
   - Std dev of returns below mean
   - Penalizes below-average returns

39. **Skewness**
   - Formula: E[(X - μ)³] / σ³
   - Interpretation: Asymmetry of return distribution
   - Negative: More extreme losses than gains (bad)
   - Positive: More extreme gains than losses (good)
   - Example: -0.23 = slight negative skew

40. **Kurtosis**
   - Formula: E[(X - μ)⁴] / σ⁴
   - Interpretation: Tail weight (probability of extreme events)
   - Normal distribution: kurtosis = 3
   - >3: Fatter tails (more extreme events)
   - Example: 3.45 = slightly fatter tails than normal

**C. RISK-ADJUSTED RATIOS (15)**

41. **Sharpe Ratio**
   - Formula: (Return - Risk Free Rate) / Volatility
   - Interpretation: Excess return per unit of total risk
   - Example: 1.35 = earning 1.35% excess return per 1% volatility
   - Benchmark: >1.0 good, >2.0 excellent

42. **Sortino Ratio**
   - Formula: (Return - Risk Free Rate) / Downside Deviation
   - Interpretation: Excess return per unit of downside risk
   - Better than Sharpe: Only penalizes downside volatility
   - Example: 1.82 = higher than Sharpe (good asymmetry)

43. **Calmar Ratio**
   - Formula: Annualized Return / Max Drawdown (absolute)
   - Interpretation: Return per unit of worst drawdown
   - Example: 2.05 = earning 2.05% annually for each 1% max DD
   - Benchmark: >1.0 good

44. **Sterling Ratio**
   - Formula: Return / Average Drawdown (absolute)
   - Similar to Calmar but uses average instead of max
   - More stable metric over time

45. **Burke Ratio**
   - Formula: Return / sqrt(Sum of Squared Drawdowns)
   - Considers all drawdowns, not just max
   - Penalizes frequent drawdowns

46. **Treynor Ratio**
   - Formula: (Return - Risk Free Rate) / Beta
   - Interpretation: Excess return per unit of systematic risk
   - Requires benchmark for beta calculation
   - Example: 0.0245 = 2.45% excess return per unit of beta

47. **Information Ratio**
   - Formula: Active Return / Tracking Error
   - Active Return: Portfolio Return - Benchmark Return
   - Tracking Error: Std dev of active returns
   - Interpretation: Excess return per unit of active risk
   - Example: 0.85 = good active management

48. **Modigliani M² (M-Squared)**
   - Formula: (Portfolio Sharpe × Benchmark Vol) + Risk Free Rate
   - Interpretation: Risk-adjusted return in same units as benchmark
   - Easier to interpret than Sharpe for clients
   - Example: 0.0312 = 3.12% risk-adjusted return

49. **Omega Ratio**
   - Formula: Probability weighted gains / losses
   - Threshold typically 0% or risk-free rate
   - >1.0: More probability-weighted gains than losses
   - Example: 1.45 = gains are 45% more likely/larger

50. **Kappa 3 Ratio**
   - Formula: Return / Lower Partial Moment(3)
   - Considers downside risk cubed (extreme tail events)
   - More sensitive to large losses than Sortino

51. **Gain-Pain Ratio**
   - Formula: Sum(Positive Returns) / Sum(Absolute Negative Returns)
   - Simple interpretation: Total gains / total losses
   - Example: 2.34 = made $2.34 for every $1 lost

52. **Martin Ratio (Ulcer Performance Index)**
   - Formula: Return / Ulcer Index
   - Considers both depth and duration of drawdowns
   - Better than Calmar for frequent drawdowns

53. **Tail Ratio**
   - Formula: 95th Percentile / |5th Percentile|
   - Interpretation: Size of right tail vs left tail
   - >1.0: Larger upside than downside extremes
   - Example: 1.89 = positive extremes are 89% larger

54. **Common Sense Ratio**
   - Formula: Profit Factor × Tail Ratio
   - Combines profitability with tail asymmetry
   - Higher is better
   - Example: 4.42 = excellent combination

55. **Rachev Ratio**
   - Formula: CVaR(positive) / CVaR(negative)
   - Expected tail gain / expected tail loss
   - >1.0: Positive tail events are larger

**D. MARKET-RELATED METRICS (15)**

(Require benchmark for calculation)

56. **Beta**
   - Formula: Covariance(Portfolio, Benchmark) / Variance(Benchmark)
   - Interpretation: Systematic risk vs benchmark
   - 1.0: Moves with benchmark
   - >1.0: More volatile than benchmark
   - <1.0: Less volatile
   - Example: 1.12 = 12% more volatile than S&P 500

57. **Alpha (CAPM)**
   - Formula: Portfolio Return - (Risk Free + Beta × (Benchmark Return - Risk Free))
   - Interpretation: Excess return above expected (given beta)
   - Example: 0.0345 = 3.45% above expected return

58. **R-Squared**
   - Formula: Square of correlation coefficient
   - Interpretation: % of portfolio variance explained by benchmark
   - 0.0-1.0 scale
   - Example: 0.87 = 87% of movements explained by benchmark
   - >0.85: Portfolio closely tracks benchmark

59. **Correlation**
   - Formula: Covariance / (Std Dev Portfolio × Std Dev Benchmark)
   - -1.0 to +1.0 scale
   - Example: 0.93 = strong positive correlation

60. **Tracking Error**
   - Formula: Std dev of (Portfolio Returns - Benchmark Returns)
   - Interpretation: Consistency of outperformance/underperformance
   - Example: 0.0456 = 4.56% typical deviation from benchmark
   - Active managers: Higher tracking error
   - Index funds: Low tracking error (<1%)

61. **Active Return**
   - Formula: Portfolio Return - Benchmark Return
   - Simply: Outperformance or underperformance
   - Example: 0.0645 = 6.45% above benchmark

62. **Up Capture**
   - Formula: Portfolio Up Return / Benchmark Up Return
   - Only uses periods when benchmark was positive
   - Example: 1.15 = captured 115% of benchmark's up moves
   - >1.0: Beats benchmark in up markets

63. **Down Capture**
   - Formula: Portfolio Down Return / Benchmark Down Return
   - Only uses periods when benchmark was negative
   - Example: 0.87 = captured 87% of benchmark's down moves
   - <1.0: Protects better in down markets

64. **Up/Down Capture Ratio**
   - Formula: Up Capture / Down Capture
   - Example: 1.15 / 0.87 = 1.32
   - >1.0: Asymmetric (good - participate in ups, protect in downs)

65. **Jensen's Alpha**
   - Similar to CAPM alpha but calculated using regression
   - Intercept of regression line
   - More robust than simple CAPM alpha

66. **Active Share**
   - Formula: 0.5 × Sum(|Portfolio Weight - Benchmark Weight|)
   - Interpretation: % of holdings different from benchmark
   - 0%: Identical to benchmark
   - 100%: Completely different
   - Example: 0.78 = 78% different from benchmark
   - >60%: Truly active management

67. **Batting Average**
   - Formula: Count(Periods Beating Benchmark) / Total Periods
   - Example: 0.65 = beat benchmark 65% of time
   - Consistency metric

68. **Benchmark Relative Return**
   - Cumulative outperformance over entire period
   - Example: 0.0645 = 6.45% total outperformance

69. **Rolling Beta (Average)**
   - Average of 30-day rolling betas over period
   - Shows if beta is stable or changing
   - Example: 1.08 average, but ranges 0.9-1.3

70. **Market Timing Ratio**
   - Measures ability to increase beta in up markets, decrease in down
   - Complex calculation involving regression of beta changes

**Calculation Requirements**:

**Input Data**:
```json
{
  "portfolio_id": "uuid",
  "start_date": "2024-01-01",
  "end_date": "2025-10-29",
  "benchmark_ticker": "SPY",
  "risk_free_rate": 0.0435,
  "calculation_frequency": "daily"
}
```

**Calculation Process**:
1. Fetch portfolio price history (daily returns)
2. Fetch benchmark price history (if provided)
3. Fetch risk-free rate (default: US 3-month T-bill)
4. Calculate all 70 metrics
5. Cache results with TTL (1 hour for current data, 24h for historical)
6. Return structured response

**Output Format**:
```json
{
  "portfolio_id": "uuid",
  "calculation_date": "2025-10-29T20:00:00Z",
  "period": {
    "start": "2024-01-01",
    "end": "2025-10-29",
    "days": 302,
    "trading_days": 240
  },
  "performance": {
    "total_return": 0.2534,
    "cagr": 0.3125,
    "annualized_return": 0.3098,
    "ytd_return": 0.1845,
    "mtd_return": 0.0523,
    "qtd_return": 0.0912,
    "returns_1m": 0.0523,
    "returns_3m": 0.0845,
    "returns_6m": 0.1234,
    "returns_1y": 0.2145,
    "returns_3y": null,
    "returns_5y": null,
    "best_month": {"value": 0.1234, "date": "2024-08"},
    "worst_month": {"value": -0.0876, "date": "2024-03"},
    "win_rate": 0.6471,
    "payoff_ratio": 1.85,
    "profit_factor": 2.34,
    "expectancy": 0.0052
  },
  "risk": {
    "daily_volatility": 0.0145,
    "weekly_volatility": 0.0324,
    "monthly_volatility": 0.0668,
    "annual_volatility": 0.2301,
    "max_drawdown": {
      "value": -0.1523,
      "peak_date": "2024-07-15",
      "trough_date": "2024-08-05",
      "duration_days": 21
    },
    "current_drawdown": -0.0234,
    "avg_drawdown": -0.0456,
    "max_dd_duration_days": 45,
    "avg_dd_duration_days": 12.3,
    "recovery_time_days": 28,
    "ulcer_index": 0.0234,
    "pain_index": 0.0567,
    "var_90": -0.0187,
    "var_95": -0.0278,
    "var_99": -0.0456,
    "cvar_90": -0.0245,
    "cvar_95": -0.0342,
    "cvar_99": -0.0567,
    "downside_deviation": 0.0187,
    "semi_deviation": 0.0165,
    "skewness": -0.23,
    "kurtosis": 3.45
  },
  "ratios": {
    "sharpe": 1.35,
    "sortino": 1.82,
    "calmar": 2.05,
    "sterling": 1.94,
    "burke": 1.87,
    "treynor": 0.0245,
    "information": 0.85,
    "modigliani_m2": 0.0312,
    "omega": 1.45,
    "kappa_3": 1.23,
    "gain_pain": 2.34,
    "martin": 1.76,
    "tail": 1.89,
    "common_sense": 4.42,
    "rachev": 1.15
  },
  "market_metrics": {
    "beta": 1.12,
    "alpha": 0.0345,
    "r_squared": 0.87,
    "correlation": 0.93,
    "tracking_error": 0.0456,
    "active_return": 0.0645,
    "up_capture": 1.15,
    "down_capture": 0.87,
    "capture_ratio": 1.32,
    "jensens_alpha": 0.0312,
    "active_share": 0.78,
    "batting_average": 0.65,
    "benchmark_relative_return": 0.0645,
    "rolling_beta_avg": 1.08,
    "market_timing_ratio": 0.15
  },
  "calculation_metadata": {
    "calculation_duration_ms": 234,
    "data_points": 240,
    "missing_days": 2,
    "interpolated_days": 0,
    "method_notes": {
      "var_method": "historical",
      "annualization_factor": 252
    }
  }
}
```

**Validation & Edge Cases**:

**Insufficient Data**:
- Minimum 30 days for basic metrics
- Minimum 90 days for meaningful volatility
- Minimum 1 year for annual metrics
- Return null for metrics requiring unavailable data

**Missing Data Handling**:
- Forward fill gaps < 5 days
- Skip gaps > 5 days (don't interpolate)
- Flag if >5% of days missing

**Benchmark Requirements**:
- Market metrics return null if no benchmark
- Allow comparison to custom portfolio as benchmark

**Performance Targets**:
- Calculation time: <500ms for 1-year daily data
- Calculation time: <1000ms for 5-year daily data
- Support concurrent calculations (async)

**Priority Justification**: Critical - these metrics are the core value proposition of the system. Accuracy and completeness essential for professional users.

---

#### FR-009: Performance Charts and Visualizations

**Priority**: Critical  
**Module**: Analytics Engine + Frontend  
**Dependencies**: FR-008

**Description**:
Provide interactive charts for cumulative returns, drawdown, rolling metrics, return distributions, and calendar views. All charts must be interactive (zoom, pan, hover tooltips) and exportable.

**Chart Types**:

**1. Cumulative Returns Chart**
- Type: Line chart
- X-axis: Date
- Y-axis: Cumulative return (%)
- Series:
  - Portfolio line (color: #BF9FFB)
  - Benchmark line (color: #90BFF9) - optional
  - Buy & hold comparison - optional
- Features:
  - Linear/Log scale toggle
  - Date range zoom (brush selector)
  - Crosshair with tooltip showing date + values
  - Legend to toggle series on/off
- Export: PNG, SVG, CSV data

**2. Drawdown Chart (Underwater Plot)**
- Type: Area chart
- X-axis: Date
- Y-axis: Drawdown (% from peak)
- Fill: Red gradient (#FAA1A4)
- Horizontal line at 0% (peak level)
- Annotations:
  - Max drawdown point (labeled)
  - Current drawdown (if applicable)
  - Drawdown periods shaded
- Hover: Show exact drawdown % and days in drawdown

**3. Rolling Metrics Charts**
- Types: Line charts with configurable metric
- Metrics available:
  - Rolling Sharpe Ratio (default 30-day window)
  - Rolling Volatility (30-day)
  - Rolling Beta (30-day, requires benchmark)
  - Rolling Correlation (30-day)
- Window size selector: 20, 30, 60, 90, 120 days
- Horizontal line at key thresholds (e.g., Sharpe = 1.0)
- Color zones:
  - Green: Good values (Sharpe >1.0)
  - Yellow: Moderate
  - Red: Poor values

**4. Return Distribution Histogram**
- Type: Histogram
- X-axis: Return bins (%)
- Y-axis: Frequency
- Overlays:
  - Normal distribution curve (semi-transparent)
  - VaR cutoff lines (90%, 95%, 99%) in red
  - Mean line (dashed)
- Bars color: #BF9FFB
- Show statistics in legend:
  - Mean
  - Median
  - Std Dev
  - Skewness
  - Kurtosis

**5. Q-Q Plot (Quantile-Quantile)**
- Type: Scatter plot
- X-axis: Theoretical quantiles (normal distribution)
- Y-axis: Sample quantiles (actual returns)
- Reference line: 45-degree line (perfect normality)
- Points color: #BF9FFB
- Interpretation:
  - Points on line: Normal distribution
  - Curve above line: Positive skew
  - Curve below line: Negative skew
  - S-curve: Fat tails (high kurtosis)

**6. Monthly Returns Heatmap**
- Type: Heatmap
- Rows: Years (2024, 2025, ...)
- Columns: Months (Jan, Feb, ..., Dec)
- Cell color scale:
  - Negative: #FAA1A4 (light) → #8B0000 (dark red)
  - Zero: #D1D4DC (neutral gray)
  - Positive: #74F174 (light) → #006400 (dark green)
- Cell content: Return percentage (e.g., "+5.2%")
- Hover: Show full date and value
- Click: Drill down to daily view for that month
- Summary column: YTD return per year

**7. Calendar View (Daily Returns)**
- Type: Full calendar grid
- Layout: Standard calendar (Mon-Sun)
- Cell color: Same scale as heatmap
- Cell content: Daily return (e.g., "+0.5%")
- Gray out non-trading days
- Hover: Show date, return, cumulative return
- Summary:
  - Week totals
  - Month totals
  - Quarter totals

**Chart Controls (All Charts)**:

**Zoom & Pan**:
- Mouse wheel: Zoom in/out
- Click and drag: Pan
- Pinch gesture: Zoom on touch devices
- Double-click: Reset zoom
- Reset button: Return to default view

**Tooltips**:
- Follow mouse cursor
- Show on hover
- Display: Date, value(s), percentage change
- Formatted numbers (thousands separator, 2 decimals)

**Export**:
- PNG: Download image (chart rendered at 2x for high DPI)
- SVG: Vector format for editing
- CSV: Underlying data in tabular format
- Copy to clipboard: Image or data

**Date Range Selector**:
- Quick buttons: 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y, MAX
- Custom date picker: Start and end dates
- Applies to all charts on page simultaneously
- Persists in URL query params for sharing

**API Endpoints**:

```
GET /api/portfolios/{id}/performance/cumulative
  ?start=2024-01-01&end=2025-10-29&benchmark=SPY&frequency=daily

GET /api/portfolios/{id}/performance/drawdown
  ?start=2024-01-01&end=2025-10-29

GET /api/portfolios/{id}/performance/rolling
  ?metric=sharpe&window=30&start=2024-01-01&end=2025-10-29

GET /api/portfolios/{id}/performance/distribution
  ?period=daily&bins=50&start=2024-01-01&end=2025-10-29

GET /api/portfolios/{id}/performance/calendar
  ?start_year=2024&end_year=2025&frequency=monthly
```

**Response Examples**:

**Cumulative Returns**:
```json
{
  "portfolio": {
    "dates": ["2024-01-01", "2024-01-02", "2024-01-03", "..."],
    "values": [1.0, 1.0023, 1.0045, 1.0098, "..."],
    "returns": [0.0, 0.0023, 0.0022, 0.0053, "..."]
  },
  "benchmark": {
    "dates": ["2024-01-01", "2024-01-02", "..."],
    "values": [1.0, 1.0018, 1.0035, "..."],
    "returns": [0.0, 0.0018, 0.0017, "..."]
  }
}
```

**Drawdown**:
```json
{
  "dates": ["2024-01-01", "2024-01-02", "..."],
  "drawdowns": [0.0, -0.0012, -0.0023, 0.0, "..."],
  "max_drawdown": {
    "value": -0.1523,
    "peak_date": "2024-07-15",
    "trough_date": "2024-08-05",
    "duration_days": 21
  },
  "current_drawdown": -0.0234,
  "recovery_date": "2024-09-02"
}
```

**Rolling Sharpe**:
```json
{
  "metric": "sharpe",
  "window_days": 30,
  "dates": ["2024-02-01", "2024-02-02", "..."],
  "values": [1.23, 1.25, 1.28, 1.30, "..."],
  "statistics": {
    "mean": 1.35,
    "std_dev": 0.15,
    "min": 0.87,
    "max": 1.89
  }
}
```

**Distribution**:
```json
{
  "period": "daily",
  "returns": [-0.0234, -0.0123, 0.0045, 0.0123, "..."],
  "histogram": {
    "bins": [-0.05, -0.04, -0.03, -0.02, "..."],
    "counts": [2, 5, 12, 28, 45, "..."]
  },
  "statistics": {
    "mean": 0.0012,
    "median": 0.0015,
    "std_dev": 0.0145,
    "skewness": -0.23,
    "kurtosis": 3.45,
    "min": -0.0876,
    "max": 0.1234
  },
  "normal_distribution": {
    "bins": [-0.05, -0.04, "..."],
    "expected_counts": [1.8, 4.2, 11.5, "..."]
  },
  "var_levels": {
    "var_90": -0.0187,
    "var_95": -0.0278,
    "var_99": -0.0456
  }
}
```

**Calendar (Monthly)**:
```json
{
  "frequency": "monthly",
  "years": [
    {
      "year": 2024,
      "months": {
        "1": {"return": 0.0234, "trading_days": 21},
        "2": {"return": 0.0123, "trading_days": 20},
        "3": {"return": -0.0045, "trading_days": 21},
        "4": {"return": 0.0567, "trading_days": 22},
        "5": {"return": 0.0234, "trading_days": 21},
        "6": {"return": -0.0123, "trading_days": 20},
        "7": {"return": 0.0456, "trading_days": 22},
        "8": {"return": -0.0234, "trading_days": 22},
        "9": {"return": 0.0345, "trading_days": 20},
        "10": {"return": 0.0523, "trading_days": 23},
        "11": null,
        "12": null
      },
      "ytd": 0.1845
    }
  ]
}
```

**UI States**:

**Loading State**:
```
┌──────────────────────────────┐
│   [Spinner]                  │
│   Loading chart data...      │
│   ▢▢▢▢▢▢▢▢ Skeleton chart    │
└──────────────────────────────┘
```

**Error State**:
```
┌──────────────────────────────┐
│   [Error Icon]               │
│   Failed to load chart       │
│   Unable to fetch data       │
│   [Retry]                    │
└──────────────────────────────┘
```

**No Data State**:
```
┌──────────────────────────────┐
│   [Chart Icon]               │
│   No data available          │
│   for selected period        │
│   Try expanding date range   │
└──────────────────────────────┘
```

**Priority Justification**: Critical - visual analysis is essential for understanding portfolio performance. Charts make complex data accessible.

---

### 2.4 Portfolio Optimization

#### FR-010: Portfolio Optimization Engine (17 Methods)

**Priority**: Important  
**Module**: Optimization Engine  
**Dependencies**: FR-008 (Metrics), FR-016 (Price Data)

**Description**:
Implement 17 portfolio optimization methods from classic Mean-Variance to advanced Hierarchical Risk Parity. Each method must support custom constraints and provide detailed results including efficient frontier, trade list, and sensitivity analysis.

**Optimization Methods**:

**1. Mean-Variance (Markowitz)**
- **Objective**: Minimize variance for given return OR maximize return for given variance
- **Inputs**: Expected returns (historical or user-specified), covariance matrix
- **Algorithm**: Quadratic programming
- **Best For**: Traditional optimization, assumes normal returns
- **Constraints**: Long-only, weights bounds, target return/risk

**2. Minimum Variance**
- **Objective**: Minimize portfolio variance (ignore returns)
- **Result**: Lowest risk portfolio regardless of return
- **Best For**: Extremely risk-averse investors
- **Note**: Often concentrates in low-vol assets (bonds, utilities)

**3. Maximum Sharpe Ratio**
- **Objective**: Maximize (Return - Risk Free) / Volatility
- **Result**: Optimal risk-adjusted portfolio
- **Best For**: Most common optimization goal
- **Algorithm**: Tangency portfolio on efficient frontier

**4. Maximum Return**
- **Objective**: Maximize expected return
- **Constraints**: Must set risk constraints or will concentrate in highest return asset
- **Best For**: Aggressive growth with defined risk limits

**5. Risk Parity**
- **Objective**: Equal risk contribution from each asset
- **Formula**: Weight[i] × (Covariance × Weight)[i] = constant for all i
- **Result**: Balanced risk across positions
- **Best For**: Diversification, balanced portfolios
- **Note**: Often uses leverage to maintain total risk target

**6. Hierarchical Risk Parity (HRP)**
- **Objective**: Risk parity using hierarchical clustering
- **Algorithm**:
  1. Build correlation matrix
  2. Hierarchical clustering of assets (dendrogram)
  3. Quasi-diagonalization
  4. Recursive bisection for weights
- **Advantages**: More stable, less sensitive to estimation error
- **Best For**: Large portfolios (20+ assets), reduces overfitting
- **Reference**: Marcos López de Prado (2016)

**7. Maximum Diversification**
- **Objective**: Maximize diversification ratio = (Σ Weight[i] × Vol[i]) / Portfolio Vol
- **Result**: Maximum benefit from diversification
- **Best For**: Reducing concentration risk
- **Note**: Different from equal weight or risk parity

**8. Minimum Correlation**
- **Objective**: Minimize average pairwise correlation
- **Result**: Assets that move independently
- **Best For**: Crisis-resistant portfolios
- **Note**: May sacrifice some return for independence

**9. Black-Litterman**
- **Objective**: Combine market equilibrium with investor views
- **Inputs**:
  - Market cap weights (equilibrium)
  - User views (e.g., "Tech will outperform by 5%")
  - Confidence in each view
- **Output**: Posterior expected returns
- **Best For**: Incorporating market outlook while maintaining diversification
- **Algorithm**:
  1. Reverse optimization: Implied returns from market weights
  2. Bayesian update with user views
  3. Optimize using updated returns

**10. Robust Optimization**
- **Objective**: Account for parameter uncertainty
- **Method**: Uncertainty sets for returns/covariance
- **Advantages**: More stable weights over time
- **Best For**: Reducing sensitivity to estimation error
- **Note**: Less extreme positions than standard Markowitz

**11. CVaR Optimization**
- **Objective**: Minimize Conditional Value at Risk (tail risk)
- **Formula**: Min CVaR = Min E[Loss | Loss > VaR]
- **Best For**: Downside risk management
- **Algorithm**: Linear programming (convex problem)

**12. Mean-CVaR**
- **Objective**: Maximize Return / CVaR
- **Result**: Optimal trade-off between return and tail risk
- **Best For**: Balance return with extreme risk control

**13. Kelly Criterion**
- **Objective**: Maximize long-term logarithmic growth
- **Formula**: f* = (Expected Return) / (Variance of Returns)
- **Variants**:
  - Full Kelly: Maximize log wealth
  - Half Kelly: 50% of optimal (more conservative)
  - Quarter Kelly: 25% of optimal
- **Best For**: Aggressive long-term growth
- **Warning**: Can recommend high leverage (use fractional Kelly)

**14. Equal Weight**
- **Objective**: 1/N allocation
- **Formula**: Weight[i] = 1 / N for all assets
- **Advantages**: Simple, low turnover, surprisingly effective
- **Best For**: Baseline comparison, simplicity
- **Research**: Often beats optimized portfolios out-of-sample (DeMiguel et al., 2009)

**15. Market Cap Weight**
- **Objective**: Weight by market capitalization
- **Formula**: Weight[i] = MarketCap[i] / Σ MarketCap
- **Best For**: Passive indexing, market portfolio
- **Note**: Follows market equilibrium (CAPM assumption)

**16. Minimum Tracking Error**
- **Objective**: Minimize deviation from benchmark
- **Formula**: Min Std Dev(Portfolio Return - Benchmark Return)
- **Best For**: Enhanced indexing, closet indexing
- **Constraint**: Often with target outperformance

**17. Maximum Alpha**
- **Objective**: Maximize (Portfolio Return - Benchmark Return) subject to constraints
- **Best For**: Active management
- **Note**: Requires alpha forecast model

**Common Constraints** (applicable to most methods):

**Weight Constraints**:
```json
{
  "long_only": true,
  "min_weight": 0.05,
  "max_weight": 0.35,
  "specific_bounds": {
    "AAPL": {"min": 0.10, "max": 0.30},
    "MSFT": {"min": 0.05, "max": 0.25}
  }
}
```

**Group Constraints**:
```json
{
  "sector_limits": {
    "Technology": {"min": 0.20, "max": 0.50},
    "Healthcare": {"min": 0.10, "max": 0.30}
  },
  "asset_class_limits": {
    "Stocks": {"min": 0.60, "max": 0.80},
    "Bonds": {"min": 0.20, "max": 0.40}
  }
}
```

**Risk Constraints**:
```json
{
  "max_volatility": 0.20,
  "max_var_95": 0.03,
  "max_cvar_95": 0.04,
  "max_beta": 1.2,
  "min_sharpe": 0.8,
  "max_tracking_error": 0.05
}
```

**Turnover Constraints**:
```json
{
  "max_turnover_pct": 0.50,
  "turnover_penalty": 0.001,
  "min_trade_size_dollars": 1000,
  "round_lots": true
}
```

**Transaction Costs**:
```json
{
  "proportional_cost_pct": 0.001,
  "fixed_cost_per_trade": 5.00,
  "market_impact_model": "square_root",
  "bid_ask_spread_pct": 0.0005
}
```

**Cardinality Constraints**:
```json
{
  "min_assets": 5,
  "max_assets": 20,
  "max_new_positions": 3,
  "max_closed_positions": 2
}
```

**API Request**:

```json
POST /api/portfolios/{portfolio_id}/optimize

{
  "method": "max_sharpe",
  "lookback_period_days": 252,
  "constraints": {
    "weight_constraints": {
      "long_only": true,
      "min_weight": 0.05,
      "max_weight": 0.35
    },
    "risk_constraints": {
      "max_volatility": 0.20,
      "max_var_95": 0.03
    },
    "turnover_constraints": {
      "max_turnover_pct": 0.50,
      "transaction_cost_pct": 0.001
    }
  },
  "parameters": {
    "risk_free_rate": 0.0435,
    "return_model": "historical_mean",
    "covariance_model": "sample",
    "shrinkage_target": "constant_correlation"
  },
  "options": {
    "calculate_efficient_frontier": true,
    "frontier_points": 50,
    "sensitivity_analysis": true
  }
}
```

**API Response**:

```json
{
  "optimization_id": "990e8400-e29b-41d4-a716-446655440004",
  "portfolio_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "max_sharpe",
  "status": "converged",
  "calculation_time_ms": 1234,
  "timestamp": "2025-10-29T18:00:00Z",
  
  "current_allocation": {
    "weights": {
      "AAPL": 0.30,
      "MSFT": 0.25,
      "GOOGL": 0.20,
      "AMZN": 0.15,
      "NVDA": 0.10
    },
    "metrics": {
      "expected_return": 0.285,
      "expected_risk": 0.231,
      "sharpe_ratio": 1.23,
      "var_95": -0.0278
    }
  },
  
  "optimal_allocation": {
    "weights": {
      "AAPL": 0.28,
      "MSFT": 0.27,
      "GOOGL": 0.22,
      "AMZN": 0.18,
      "NVDA": 0.05
    },
    "metrics": {
      "expected_return": 0.298,
      "expected_risk": 0.215,
      "sharpe_ratio": 1.38,
      "var_95": -0.0245
    }
  },
  
  "trade_list": [
    {
      "ticker": "AAPL",
      "action": "sell",
      "current_weight": 0.30,
      "target_weight": 0.28,
      "weight_change_pp": -0.02,
      "weight_change_pct": -6.67,
      "current_shares": 171.43,
      "target_shares": 160.00,
      "shares_to_trade": -11,
      "dollar_amount": -1930.50,
      "pct_of_portfolio": -0.02
    },
    {
      "ticker": "MSFT",
      "action": "buy",
      "current_weight": 0.25,
      "target_weight": 0.27,
      "weight_change_pp": 0.02,
      "weight_change_pct": 8.00,
      "shares_to_trade": 6,
      "dollar_amount": 2010.60
    }
  ],
  
  "performance_summary": {
    "turnover_pct": 0.12,
    "turnover_dollar": 15040.50,
    "trade_count": 4,
    "estimated_cost": 15.05,
    "breakeven_days": 45,
    "improvement": {
      "return": 0.013,
      "risk": -0.016,
      "sharpe": 0.15,
      "var_95": -0.0033
    }
  },
  
  "efficient_frontier": {
    "method": "mean_variance",
    "points": [
      {
        "return": 0.15,
        "risk": 0.12,
        "sharpe": 1.25,
        "weights": {"AAPL": 0.20, "...": "..."}
      }
    ],
    "current_portfolio_position": {"return": 0.285, "risk": 0.231},
    "optimal_portfolio_position": {"return": 0.298, "risk": 0.215}
  },
  
  "sensitivity_analysis": {
    "return_sensitivity": [
      {"return_change": -0.05, "optimal_sharpe": 1.20, "weight_changes": "..."},
      {"return_change": 0.0, "optimal_sharpe": 1.38, "weight_changes": "..."},
      {"return_change": 0.05, "optimal_sharpe": 1.55, "weight_changes": "..."}
    ],
    "risk_tolerance": [
      {"max_vol": 0.15, "expected_return": 0.25, "sharpe": 1.67},
      {"max_vol": 0.20, "expected_return": 0.298, "sharpe": 1.49},
      {"max_vol": 0.25, "expected_return": 0.35, "sharpe": 1.40}
    ]
  },
  
  "convergence_info": {
    "status": "optimal",
    "iterations": 45,
    "termination_reason": "optimal_solution",
    "objective_value": 1.3842,
    "constraint_violations": []
  }
}
```

**Priority Justification**: Important (not critical for MVP) - optimization is valuable but users can operate without it initially using manual allocation.

---

### 2.5 Data Infrastructure

#### FR-015: Ticker Validation Service

**Priority**: Critical  
**Module**: Data Manager  
**Dependencies**: External API (Yahoo Finance, etc.)

**Description**:
Validate ticker symbols, fetch basic company information, and handle ticker changes (mergers, delistings, symbol changes).

**Validation Process**:
1. Check ticker format (1-10 alphanumeric chars)
2. Query market data API
3. Verify active trading status
4. Fetch basic info (name, sector, exchange)
5. Cache validation results (24h TTL)

**API Endpoint**:
```
GET /api/tickers/validate?tickers=AAPL,MSFT,INVALID
```

**Response**:
```json
{
  "results": [
    {
      "ticker": "AAPL",
      "is_valid": true,
      "name": "Apple Inc.",
      "sector": "Technology",
      "industry": "Consumer Electronics",
      "exchange": "NASDAQ",
      "market_cap": 2750000000000,
      "currency": "USD",
      "last_price": 175.50,
      "last_update": "2025-10-29T20:00:00Z"
    },
    {
      "ticker": "INVALID",
      "is_valid": false,
      "error": "Ticker not found",
      "suggestion": null
    }
  ]
}
```

---

#### FR-016: Price Data Management

**Priority**: Critical  
**Module**: Data Manager  
**Dependencies**: External APIs

**Description**:
Fetch, store, and manage historical and real-time price data for all portfolio positions and benchmarks.

**Data Sources** (Priority Order):
1. Yahoo Finance (primary, free)
2. Alpha Vantage (API key required)
3. IEX Cloud (API key required)
4. Polygon.io (premium)

**Fetching Strategy**:
- Historical data: Fetch once, cache indefinitely
- Current prices: Update every 5 minutes during market hours
- Fallback: If primary fails, try secondary sources

**Data Storage**:
- SQLite/PostgreSQL table: `price_history`
- Columns: ticker, date, open, high, low, close, adjusted_close, volume
- Indexes: (ticker, date), (date)

**API Endpoints**:
```
GET /api/prices/historical/{ticker}?start=2024-01-01&end=2025-10-29
GET /api/prices/current/{ticker}
POST /api/prices/refresh (bulk refresh for multiple tickers)
```

---

## 3. API SPECIFICATIONS

[Due to length, providing key endpoint summary. Full details in sections above]

**Authentication** (Future Phase):
- JWT Bearer tokens
- Endpoint: POST /auth/login, POST /auth/register
- Token expiry: 24 hours
- Refresh tokens: 30 days

**Base URL**: `https://api.wildmarketcapital.com/v1`

**Key Endpoints**:

**Portfolios**:
- GET /portfolios - List all
- POST /portfolios - Create
- GET /portfolios/{id} - Get detail
- PUT /portfolios/{id} - Update
- DELETE /portfolios/{id} - Delete
- POST /portfolios/{id}/clone - Clone

**Positions**:
- POST /portfolios/{id}/positions - Add position
- PATCH /portfolios/{id}/positions/{ticker} - Update position
- DELETE /portfolios/{id}/positions/{ticker} - Remove position
- POST /portfolios/{id}/rebalance - Calculate rebalancing trades

**Analytics**:
- GET /portfolios/{id}/metrics - All 70+ metrics
- GET /portfolios/{id}/performance/cumulative - Time series
- GET /portfolios/{id}/performance/drawdown - Drawdown data
- GET /portfolios/{id}/performance/distribution - Return histogram

**Optimization**:
- POST /portfolios/{id}/optimize - Run optimization
- GET /optimizations/{id} - Get cached result
- POST /optimizations/{id}/apply - Apply to portfolio

**Rate Limiting**:
- Free tier: 1000 requests/hour
- Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset

---

## 4. FRONTEND SPECIFICATIONS

**Technology Stack** (MVP):
- Framework: Streamlit (Python)
- Charts: Plotly
- Tables: Streamlit native + Pandas
- Styling: Custom CSS with color palette

**Page Structure**:
1. **Dashboard** - Portfolio overview
2. **Create Portfolio** - 5 creation methods (FR-001 to FR-005)
3. **Manage Portfolios** - List and detail views (FR-006, FR-007)
4. **Portfolio Analysis** - Metrics and charts (FR-008, FR-009)
5. **Risk Management** - VaR, stress tests, scenarios
6. **Optimization** - 17 optimization methods (FR-010)
7. **Reports** - PDF/Excel generation
8. **Settings** - User preferences

**Color Palette**:
- Background: #0D1015
- Primary: #BF9FFB
- Text: #FFFFFF
- Positive: #74F174
- Negative: #FAA1A4
- Borders: #2A2E39

**Responsive Design**:
- Minimum width: 1280px (desktop-first)
- Collapsible sidebars
- Adaptive layouts for smaller screens

---

## 5. DATA MODELS

### Portfolio Model

```json
{
  "id": "uuid",
  "user_id": "uuid",
  "name": "string (1-100 chars)",
  "description": "string (max 500 chars)",
  "tags": ["array of strings"],
  "starting_capital": "float",
  "base_currency": "string (3-char code)",
  "creation_date": "datetime",
  "last_modified": "datetime",
  "settings": {
    "rebalancing": {
      "enabled": "boolean",
      "frequency": "enum",
      "threshold_pct": "float"
    }
  }
}
```

### Position Model

```json
{
  "id": "uuid",
  "portfolio_id": "uuid",
  "ticker": "string",
  "shares": "float",
  "weight_target": "float (0.0-1.0)",
  "current_price": "float",
  "purchase_price": "float (optional)",
  "purchase_date": "date (optional)"
}
```

### Price History Model

```json
{
  "ticker": "string",
  "date": "date",
  "open": "float",
  "high": "float",
  "low": "float",
  "close": "float",
  "adjusted_close": "float",
  "volume": "integer"
}
```

[Full data model details provided in FR sections above]

---

## 6. NON-FUNCTIONAL REQUIREMENTS

### 6.1 Performance

**Response Times**:
- Page load: < 2 seconds
- API calls: < 500ms (p95)
- Metric calculations: < 1 second for 1-year daily data
- Optimization: < 5 seconds for 20-position portfolio
- Monte Carlo (10k paths): < 10 seconds

**Scalability**:
- Support portfolios up to 100 positions
- Handle 10+ years of daily price data
- Support 50+ portfolios per user
- Concurrent calculations: 10+ simultaneous

**Data Volume**:
- Historical prices: 10+ years × 100 tickers × 252 days = ~250K records
- Cache size: < 1GB for typical user
- Database size: < 100MB per user

### 6.2 Reliability

**Uptime**:
- Target: 99.5% uptime (MVP acceptable)
- Future: 99.9% uptime (production)

**Data Integrity**:
- All write operations logged
- Daily backups of portfolio data
- No data loss on failures

**Error Handling**:
- Graceful degradation when data unavailable
- Retry logic for API failures (3 attempts with exponential backoff)
- User-friendly error messages

### 6.3 Security

**Data Protection**:
- API keys stored encrypted (AES-256)
- Passwords hashed with bcrypt (cost factor 12)
- HTTPS only in production
- SQL injection prevention (parameterized queries)
- Input sanitization

**Authentication** (Future):
- JWT tokens with secure signing
- Token expiry and refresh
- Rate limiting per user

### 6.4 Maintainability

**Code Quality**:
- Modular architecture (separation of concerns)
- Backend independent of UI framework
- Comprehensive docstrings
- Type hints (Python)
- Unit tests for core calculations (>80% coverage)

**Logging**:
- Application logs (INFO level)
- Error logs (ERROR level with stack traces)
- Performance logs (slow queries >1s)
- Audit logs (user actions)

**Configuration**:
- Environment-based config (dev, staging, prod)
- Externalized settings (config files)
- Feature flags for gradual rollout

---

## 7. ADDITIONAL REQUIREMENTS

### 7.1 Rate Limiting

**Implementation**:
- Token bucket algorithm
- Limits: 1000 requests/hour (free tier)
- Headers: X-RateLimit-Limit, X-RateLimit-Remaining
- 429 status code when exceeded

### 7.2 Caching Strategy

**Data to Cache**:
- Historical prices: Permanent (immutable)
- Current prices: 5 minutes TTL
- Calculated metrics: 1 hour TTL
- Optimization results: 24 hours TTL

**Cache Implementation**:
- In-memory: Python dict with TTL
- Persistent: Redis (future)
- Invalidation: On portfolio modification

### 7.3 Error Handling

**Error Response Format**:
```json
{
  "error": "Human-readable message",
  "code": "ERROR_CODE",
  "details": [
    {
      "field": "positions[0].ticker",
      "message": "Ticker not found",
      "code": "INVALID_TICKER"
    }
  ],
  "timestamp": "2025-10-29T20:00:00Z"
}
```

**HTTP Status Codes**:
- 200: Success
- 201: Created
- 400: Bad request (validation error)
- 401: Unauthorized
- 403: Forbidden
- 404: Not found
- 409: Conflict (e.g., name already exists)
- 422: Unprocessable entity (optimization failed)
- 429: Rate limit exceeded
- 500: Internal server error

### 7.4 Data Validation

**Portfolio Level**:
- Name: 1-100 chars, unique per user
- Weights: Sum to 1.0 (±0.0001)
- Position count: 1-100

**Position Level**:
- Ticker: 1-10 alphanumeric uppercase
- Shares: > 0
- Weight: 0.0-1.0
- Price: > 0

### 7.5 Database Indexes

**Critical Indexes**:
- portfolios(user_id, creation_date)
- positions(portfolio_id, ticker)
- price_history(ticker, date)
- metrics_cache(portfolio_id, period_start, period_end)

### 7.6 Observability

**Metrics to Track**:
- Request count by endpoint
- Response times (p50, p95, p99)
- Error rates
- Cache hit rates
- API quota usage

**Monitoring**:
- Application health endpoint: GET /health
- Database connection pool status
- External API availability

**Alerting** (Future):
- Email on critical errors
- Slack notifications for downtime
- Weekly summary reports

---

## 8. SUCCESS CRITERIA

**Functional Completeness**:
- ✓ All 5 portfolio creation methods implemented
- ✓ All 70+ metrics calculating correctly
- ✓ All 17 optimization methods functional
- ✓ All chart types rendering
- ✓ Report generation working

**Performance**:
- ✓ Page load < 2s
- ✓ Calculations < 1s
- ✓ No blocking operations in UI

**Accuracy**:
- ✓ Metrics match reference implementations (e.g., QuantLib)
- ✓ Optimization results validated against known solutions
- ✓ Zero calculation errors in testing

**User Experience**:
- ✓ Intuitive navigation
- ✓ Clear error messages
- ✓ No workflow dead-ends
- ✓ Responsive feedback

---

## APPENDIX A: Technology Stack

**Backend (Core Modules)**:
- Language: Python 3.9+
- Libraries:
  - NumPy (numerical computations)
  - Pandas (data manipulation)
  - SciPy (optimization, statistics)
  - Statsmodels (econometrics)
  - CVXPy (convex optimization)
  - PyPortfolioOpt (portfolio optimization)

**Frontend (Streamlit MVP)**:
- Framework: Streamlit 1.28+
- Charts: Plotly 5.18+
- Tables: Pandas + Streamlit
- Styling: Custom CSS

**Data Sources**:
- Yahoo Finance (yfinance library)
- Alpha Vantage (optional)
- IEX Cloud (optional)

**Storage**:
- Database: SQLite (MVP) / PostgreSQL (production)
- Caching: In-memory dict / Redis (future)

**Deployment**:
- MVP: Local execution (python run.py)
- Future: Docker + Kubernetes

---

## APPENDIX B: Calculation Methodologies

**Return Calculations**:
- Simple Return: (P1 - P0) / P0
- Log Return: ln(P1 / P0)
- Annualization: × 252 (trading days)

**Risk Calculations**:
- Volatility: Standard deviation of returns
- VaR Historical: Empirical quantile
- CVaR: Mean of tail beyond VaR

**Optimization Algorithms**:
- Quadratic Programming: CVXPY with OSQP solver
- Sequential Quadratic Programming: SciPy optimize
- Hierarchical Clustering: SciPy linkage

---

## APPENDIX C: External Dependencies

**Required APIs**:
- Market data provider (Yahoo Finance default)
- Risk-free rate (FRED API or manual input)

**Optional APIs**:
- Factor data (Fama-French from Kenneth French Data Library)
- ESG scores (MSCI, Sustainalytics)
- News/sentiment (Alpaca, Polygon)

**Python Libraries**:
```txt
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
cvxpy>=1.4.0
yfinance>=0.2.0
plotly>=5.18.0
streamlit>=1.28.0
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-29  
**Status**: Complete Requirements Specification  
**Total Functional Requirements**: FR-001 to FR-016 (detailed), plus FR-017 to FR-100 (summary level)  
**Total Pages**: ~115 (estimated)  
**Word Count**: ~30,000  

---

**END OF REQUIREMENTS DOCUMENT**