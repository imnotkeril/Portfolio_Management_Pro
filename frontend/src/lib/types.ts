export type User = {
  id: string;
  email: string;
  created_at: string;
  is_active: boolean;
};

export type TokenResponse = {
  access_token: string;
  token_type: string;
};

export type Position = {
  ticker: string;
  shares: number;
  weight_target: number | null;
  purchase_price: number | null;
  purchase_date: string | null;
};

export type Portfolio = {
  id: string;
  name: string;
  description: string | null;
  starting_capital: number;
  base_currency: string;
  cost_basis_method?: string;
  rebalance_interval_months?: number | null;
  /** buy_hold | transactions — set at creation, not inferred from tx count */
  ledger_mode?: "buy_hold" | "transactions";
  positions: Position[];
};

export type Transaction = {
  id: string;
  transaction_date: string;
  transaction_type: string;
  ticker: string;
  shares: number;
  price: number;
  amount: number;
  fees: number;
  notes: string | null;
  reinvest?: boolean | null;
  split_ratio?: number | null;
  currency?: string;
};

export type Holding = {
  ticker: string;
  quantity: number;
  avg_cost: number;
  market_price?: number | null;
  market_value?: number | null;
  cost_basis?: number;
  unrealized_pnl?: number | null;
};

export type PnlSummary = {
  realized_pnl: number;
  unrealized_pnl: number;
  total_return_twr?: number;
  total_return_mwr?: number;
  dividend_income?: number;
};
