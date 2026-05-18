"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { defaultFeesForTransaction } from "@/lib/ib-fees";
import { buildPositionsWithCash } from "@/lib/portfolio-allocation";
import { fetchTickerPrice } from "@/lib/ticker-price-api";

/* ───────── types ───────── */

type InputMethod = "text" | "file" | "manual" | "template";

type PositionDraft = { ticker: string; weight: number };

type TickerValidation = {
  ticker: string;
  valid: boolean;
  price: number | null;
};

type TickerPriceResponse = {
  ticker: string;
  valid: boolean;
  price: number | null;
};

type CreatedPortfolio = {
  id: string;
  name: string;
  description: string;
  starting_capital: number;
  base_currency: string;
  positions: {
    ticker: string;
    shares: number;
    weight_target: number;
    purchase_price: number | null;
    purchase_date?: string | null;
  }[];
};

type PositionPayload = {
  ticker: string;
  shares: number;
  weight_target: number;
  purchase_price?: number;
  purchase_date?: string;
};

/* ───────── templates (mirror Streamlit) ───────── */

const TEMPLATES: Record<
  string,
  { description: string; assets: string; tags: string[] }
> = {
  "Value Factor": {
    description: "Undervalued companies with low P/E and P/B ratios",
    assets: "VTV 30%, IWD 25%, BRK-B 15%, JPM 10%, WMT 8%, CVX 7%, XOM 5%",
    tags: ["Value", "Undervalued", "P/E < 15"],
  },
  "Quality Factor": {
    description: "High ROE companies with low debt and stable profits",
    assets: "QUAL 35%, MSFT 20%, AAPL 15%, JNJ 10%, PG 8%, V 7%, MA 5%",
    tags: ["Quality", "ROE > 15%", "Low Debt"],
  },
  "Growth Factor": {
    description: "Fast-growing companies with high revenue and EPS growth",
    assets: "VUG 30%, IWF 25%, NVDA 15%, GOOGL 10%, AMZN 8%, TSLA 7%, META 5%",
    tags: ["Growth", "Revenue > 10%", "EPS Growth"],
  },
  "Low Volatility": {
    description: "Low volatility stocks with beta < 0.8",
    assets: "USMV 40%, SPLV 30%, KO 8%, PG 7%, JNJ 6%, VZ 5%, WMT 4%",
    tags: ["Low Vol", "Beta < 0.8", "Defensive"],
  },
  "Dividend Factor": {
    description: "High dividend yield stocks with 3%+ yield",
    assets: "VYM 25%, SCHD 25%, HDV 20%, T 8%, VZ 7%, XOM 6%, KO 5%, PFE 4%",
    tags: ["Dividends", "Yield > 3%", "Income"],
  },
  "60/40 Portfolio": {
    description: "Classic 60% stocks, 40% bonds allocation",
    assets: "VTI 60%, BND 40%",
    tags: ["Balanced", "Classic", "Moderate Risk"],
  },
  "All Weather Portfolio": {
    description: "Multi-asset diversification across all economic conditions",
    assets: "VTI 30%, BND 25%, GLD 15%, VNQ 15%, BTC-USD 10%, TIP 5%",
    tags: ["Multi-Asset", "All Weather", "Diversified"],
  },
  "Tech Focus": {
    description: "Technology sector concentration with growth leaders",
    assets:
      "AAPL 25%, MSFT 20%, GOOGL 15%, NVDA 12%, META 10%, AMZN 8%, TSLA 5%, AMD 3%, CRM 2%",
    tags: ["Technology", "Growth", "High Risk"],
  },
};

/* ───────── helpers ───────── */

function parseTextPositions(text: string): PositionDraft[] {
  const assets: PositionDraft[] = [];
  const trimmed = text.trim();
  if (!trimmed) return assets;

  const lines = trimmed.includes(",")
    ? trimmed.split(",").map((s) => s.trim()).filter(Boolean)
    : trimmed.split("\n").map((s) => s.trim()).filter(Boolean);

  for (const line of lines) {
    const colonMatch = line.match(/^([A-Za-z0-9.-]+):\s*([0-9.%]+)$/);
    const spaceMatch = line.match(/^([A-Za-z0-9.-]+)\s+([0-9.]+)%?$/);
    const tickerOnly = line.match(/^([A-Za-z0-9.-]+)$/);

    if (colonMatch) {
      const ticker = colonMatch[1].toUpperCase();
      let w = parseFloat(colonMatch[2].replace("%", ""));
      if (w > 1) w /= 100;
      assets.push({ ticker, weight: w });
    } else if (spaceMatch) {
      const ticker = spaceMatch[1].toUpperCase();
      let w = parseFloat(spaceMatch[2]);
      if (w > 1 || line.endsWith("%")) w /= 100;
      assets.push({ ticker, weight: w });
    } else if (tickerOnly) {
      assets.push({ ticker: tickerOnly[1].toUpperCase(), weight: 0 });
    }
  }

  if (assets.length > 0 && assets.every((a) => a.weight === 0)) {
    const eq = 1 / assets.length;
    assets.forEach((a) => (a.weight = eq));
  }
  return assets;
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

/* ───────── sub-components ───────── */

function Expander({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="expander" open={defaultOpen || undefined}>
      <summary>{title}</summary>
      <div className="expander-body">{children}</div>
    </details>
  );
}

function Alert({
  type,
  children,
}: {
  type: "success" | "error" | "warning" | "info";
  children: React.ReactNode;
}) {
  return <div className={`alert alert-${type}`}>{children}</div>;
}

function Tip({ text }: { text: string }) {
  return (
    <span className="tooltip-wrap">
      <span className="tooltip-icon">?</span>
      <span className="tooltip-bubble">{text}</span>
    </span>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card text-center">
      <div className="text-xs text-white/40 mb-1">{label}</div>
      <div className="text-lg font-semibold text-white">{value}</div>
    </div>
  );
}

function ProgressBar({ step, total }: { step: number; total: number }) {
  const progress = ((step - 1) / (total - 1)) * 100;
  return (
    <div className="space-y-1">
      <div className="text-xs text-white/50">
        Step {step} of {total}
      </div>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}

/* =============================================================== */
/*                          MAIN COMPONENT                         */
/* =============================================================== */

export default function CreatePortfolioPage() {
  const router = useRouter();

  /* --- wizard state --- */
  const [step, setStep] = useState(1);
  const [creating, setCreating] = useState(false);
  const [createdPortfolio, setCreatedPortfolio] =
    useState<CreatedPortfolio | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  /* --- step 1 --- */
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [baseCurrency, setBaseCurrency] = useState("USD");
  const [capital, setCapital] = useState(100000);
  const [nameStatus, setNameStatus] = useState<
    "idle" | "checking" | "available" | "taken" | "invalid"
  >("idle");

  /* --- step 2 --- */
  const [method, setMethod] = useState<InputMethod>("text");

  /* --- step 3 --- */
  const [textInput, setTextInput] = useState("");
  const [positions, setPositions] = useState<PositionDraft[]>([]);
  const [validationMap, setValidationMap] = useState<
    Record<string, TickerValidation>
  >({});
  const [validating, setValidating] = useState(false);

  // manual entry
  const [manualTicker, setManualTicker] = useState("");
  const [manualWeight, setManualWeight] = useState(10);
  const [manualValidation, setManualValidation] =
    useState<TickerValidation | null>(null);
  const [manualValidating, setManualValidating] = useState(false);

  // template
  const [selectedTemplate, setSelectedTemplate] = useState(
    Object.keys(TEMPLATES)[0],
  );
  const [customizeTemplate, setCustomizeTemplate] = useState(false);
  const [customTemplateText, setCustomTemplateText] = useState("");

  // file upload
  const [filePositions, setFilePositions] = useState<PositionDraft[]>([]);
  const [fileName, setFileName] = useState<string | null>(null);

  /* --- step 4 --- */
  const [fetchInfo, setFetchInfo] = useState(true);
  const [autoNormalize, setAutoNormalize] = useState(true);
  const [updatePrices, setUpdatePrices] = useState(true);
  const [calculateShares, setCalculateShares] = useState(true);
  const [cashAllocation, setCashAllocation] = useState(0);
  const [portfolioMode, setPortfolioMode] = useState<
    "buy_hold" | "transactions"
  >("buy_hold");
  const [createInitialTxns, setCreateInitialTxns] = useState(true);
  const [inceptionDate, setInceptionDate] = useState(() =>
    new Date().toISOString().slice(0, 10),
  );

  /* --- derived --- */
  const totalWeight = useMemo(
    () => positions.reduce((s, p) => s + p.weight, 0),
    [positions],
  );

  const validPositions = useMemo(() => {
    if (Object.keys(validationMap).length === 0) return positions;
    return positions.filter(
      (p) => validationMap[p.ticker]?.valid !== false,
    );
  }, [positions, validationMap]);

  const canProceedStep1 = name.trim().length > 0 && nameStatus === "available";

  const canProceedStep3 = (() => {
    if (method === "text") return validPositions.length > 0;
    if (method === "manual") return positions.length > 0;
    if (method === "template") return true;
    if (method === "file") return filePositions.length > 0;
    return false;
  })();

  /* --- name validation --- */
  useEffect(() => {
    if (!name.trim()) {
      setNameStatus("idle");
      return;
    }
    if (name.trim().length > 100) {
      setNameStatus("invalid");
      return;
    }
    setNameStatus("checking");
    const timeout = setTimeout(async () => {
      try {
        const list = await api.get<{ name: string }[]>("/portfolios");
        const exists = list.some(
          (p) => p.name.toLowerCase() === name.trim().toLowerCase(),
        );
        setNameStatus(exists ? "taken" : "available");
      } catch {
        setNameStatus("available");
      }
    }, 400);
    return () => clearTimeout(timeout);
  }, [name]);

  /* --- text parse & validate --- */
  const parseAndValidate = useCallback(async () => {
    const parsed = parseTextPositions(textInput);
    setPositions(parsed);
    if (parsed.length === 0) return;
    setValidating(true);
    try {
      const tickers = parsed.map((p) => p.ticker);
      const results = await api.post<Record<string, boolean>>(
        "/validate-tickers",
        tickers,
      );
      const map: Record<string, TickerValidation> = {};
      for (const t of tickers) {
        map[t] = { ticker: t, valid: results[t] ?? false, price: null };
      }
      setValidationMap(map);
    } catch {
      setValidationMap({});
    } finally {
      setValidating(false);
    }
  }, [textInput]);

  /* --- manual: validate ticker --- */
  const validateManualTicker = useCallback(async () => {
    const t = manualTicker.trim().toUpperCase();
    if (!t) return;
    setManualValidating(true);
    try {
      const result = await api.get<TickerValidation>(`/ticker-price/${t}`);
      setManualValidation(result);
    } catch {
      setManualValidation({ ticker: t, valid: false, price: null });
    } finally {
      setManualValidating(false);
    }
  }, [manualTicker]);

  /* --- manual: add position --- */
  const addManualPosition = useCallback(async () => {
    const t = manualTicker.trim().toUpperCase();
    if (!t || manualWeight <= 0) return;
    if (positions.some((p) => p.ticker === t)) {
      setManualValidation({
        ticker: t,
        valid: false,
        price: null,
      });
      return;
    }
    // validate first if not already done
    let validation = manualValidation;
    if (!validation || validation.ticker !== t) {
      setManualValidating(true);
      try {
        validation = await api.get<TickerValidation>(`/ticker-price/${t}`);
        setManualValidation(validation);
      } catch {
        validation = { ticker: t, valid: false, price: null };
        setManualValidation(validation);
      } finally {
        setManualValidating(false);
      }
    }
    if (!validation?.valid) return;
    setPositions((prev) => [...prev, { ticker: t, weight: manualWeight / 100 }]);
    setManualTicker("");
    setManualWeight(10);
    setManualValidation(null);
  }, [manualTicker, manualWeight, positions, manualValidation]);

  /* --- file upload handler --- */
  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (ev) => {
        const text = ev.target?.result as string;
        if (!text) return;
        const lines = text
          .split("\n")
          .map((l) => l.trim())
          .filter(Boolean);
        if (lines.length < 2) return;
        const header = lines[0].toLowerCase().split(",");
        const tickerIdx = header.findIndex((h) =>
          ["ticker", "symbol"].includes(h.trim()),
        );
        const weightIdx = header.findIndex((h) =>
          ["weight", "allocation", "pct"].includes(h.trim()),
        );
        if (tickerIdx === -1) return;
        const parsed: PositionDraft[] = [];
        for (let i = 1; i < lines.length; i++) {
          const cols = lines[i].split(",");
          const ticker = cols[tickerIdx]?.trim().toUpperCase();
          if (!ticker) continue;
          let w = 0;
          if (weightIdx !== -1 && cols[weightIdx]) {
            w = parseFloat(cols[weightIdx].replace("%", ""));
            if (w > 1) w /= 100;
          }
          parsed.push({ ticker, weight: w });
        }
        if (parsed.length > 0 && parsed.every((p) => p.weight === 0)) {
          const eq = 1 / parsed.length;
          parsed.forEach((p) => (p.weight = eq));
        }
        setFilePositions(parsed);
      };
      reader.readAsText(file);
    },
    [],
  );

  /* --- go to step 3 from template: parse template --- */
  useEffect(() => {
    if (step === 3 && method === "template") {
      const tpl = TEMPLATES[selectedTemplate];
      if (tpl) {
        const text = customizeTemplate ? customTemplateText : tpl.assets;
        setPositions(parseTextPositions(text));
      }
    }
  }, [step, method, selectedTemplate, customizeTemplate, customTemplateText]);

  /* --- go to step 3 from file --- */
  useEffect(() => {
    if (step === 3 && method === "file" && filePositions.length > 0) {
      setPositions(filePositions);
    }
  }, [step, method, filePositions]);

  /* --- create portfolio --- */
  const handleCreate = useCallback(
    async (e?: FormEvent) => {
      e?.preventDefault();
      setCreating(true);
      setErrorMsg(null);
      try {
        const raw =
          method === "file" ? filePositions : positions;
        let rows = raw.filter((p) => {
          if (Object.keys(validationMap).length > 0) {
            return validationMap[p.ticker]?.valid !== false;
          }
          return true;
        });
        if (rows.length === 0) {
          setErrorMsg("No valid positions to create.");
          return;
        }

        let normWeights = rows.map((p) => ({ ...p }));
        if (autoNormalize) {
          const sum = normWeights.reduce((s, p) => s + p.weight, 0);
          if (sum > 0 && Math.abs(sum - 1) > 1e-6) {
            normWeights = normWeights.map((p) => ({
              ...p,
              weight: p.weight / sum,
            }));
          }
        }

        let posPayload: PositionPayload[] = [];
        const priceByTicker: Record<string, number> = {};
        const asOfDate =
          portfolioMode === "transactions"
            ? inceptionDate
            : new Date().toISOString().slice(0, 10);

        if (calculateShares) {
          const prices: Record<string, number> = {};
          for (const p of normWeights) {
            const t = p.ticker.toUpperCase();
            if (t === "CASH") continue;
            const res = await fetchTickerPrice(t, asOfDate);
            const px =
              res?.valid && res.price != null && res.price > 0
                ? Number(res.price)
                : null;
            if (px == null) {
              setErrorMsg(
                `Could not get a valid price for ${t} on ${asOfDate}. Check the ticker or date.`,
              );
              return;
            }
            prices[t] = px;
            priceByTicker[t] = px;
          }

          const built = buildPositionsWithCash({
            rows: normWeights.map((p) => ({
              ticker: p.ticker,
              weight: p.weight,
            })),
            prices,
            totalCapital: capital,
            cashAllocationPct: cashAllocation,
            floorShares: true,
          });

          if (built.positions.filter((p) => p.ticker !== "CASH").length === 0) {
            setErrorMsg(
              "No whole-share positions could be built. Lower prices or increase capital.",
            );
            return;
          }

          posPayload = built.positions.map((p) => ({
            ticker: p.ticker,
            shares: p.shares,
            weight_target: p.weight_target,
            purchase_price: p.ticker === "CASH" ? 1 : p.purchase_price,
          }));
        } else {
          posPayload = normWeights.map((p) => ({
            ticker: p.ticker,
            shares: 1,
            weight_target: p.weight,
            purchase_date: asOfDate,
          }));
        }

        const result = await api.post<CreatedPortfolio>("/portfolios", {
          name: name.trim(),
          description: description.trim(),
          starting_capital: capital,
          base_currency: baseCurrency,
          positions: posPayload,
        });

        if (portfolioMode === "transactions" && createInitialTxns) {
          for (const pos of result.positions) {
            if (pos.ticker === "CASH") {
              if (pos.shares > 0) {
                await api.post(`/portfolios/${result.id}/transactions`, {
                  transaction_date: inceptionDate,
                  transaction_type: "DEPOSIT",
                  ticker: "CASH",
                  shares: pos.shares,
                  price: 1.0,
                  fees: 0,
                  notes: "Initial cash allocation",
                });
              }
              continue;
            }
            let price = pos.purchase_price ?? priceByTicker[pos.ticker] ?? null;
            if (!price || price <= 0) {
              const res = await api.get<TickerPriceResponse>(
                `/ticker-price/${encodeURIComponent(pos.ticker)}`,
              );
              price =
                res?.valid && res.price != null && res.price > 0
                  ? res.price
                  : 1;
            }
            const wholeShares = Math.floor(pos.shares);
            if (wholeShares > 0) {
              await api.post(`/portfolios/${result.id}/transactions`, {
                transaction_date: inceptionDate,
                transaction_type: "BUY",
                ticker: pos.ticker,
                shares: wholeShares,
                price,
                fees: defaultFeesForTransaction("BUY", wholeShares, price),
                notes: "Initial position",
              });
            }
          }
        }

        setCreatedPortfolio(result);
        setStep(6);
      } catch (err) {
        setErrorMsg(String(err));
      } finally {
        setCreating(false);
      }
    },
    [
      name,
      description,
      capital,
      baseCurrency,
      positions,
      filePositions,
      method,
      validationMap,
      calculateShares,
      autoNormalize,
      cashAllocation,
      portfolioMode,
      createInitialTxns,
      inceptionDate,
    ],
  );

  /* --- navigation helpers --- */
  const goPrev = () => setStep((s) => Math.max(1, s - 1));

  const goNext = () => {
    if (step === 3 && method === "file") {
      setPositions(filePositions);
    }
    setStep((s) => Math.min(5, s + 1));
  };

  const resetAll = () => {
    setStep(1);
    setName("");
    setDescription("");
    setBaseCurrency("USD");
    setCapital(100000);
    setMethod("text");
    setTextInput("");
    setPositions([]);
    setValidationMap({});
    setCreatedPortfolio(null);
    setErrorMsg(null);
    setManualTicker("");
    setManualWeight(10);
    setManualValidation(null);
    setFilePositions([]);
    setFileName(null);
    setCashAllocation(0);
    setPortfolioMode("buy_hold");
  };

  /* ───────── RENDER ───────── */

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-white">Create New Portfolio</h1>

      {/* --- Top help --- */}
      <Expander title="How to Create a Portfolio">
        <div className="space-y-3">
          <p className="font-medium text-white/80">
            Step-by-Step Portfolio Creation
          </p>
          <ol className="list-decimal list-inside space-y-1.5">
            <li>
              <strong>Portfolio Information</strong> — Name, description,
              currency, and initial investment
            </li>
            <li>
              <strong>Choose Input Method</strong> — Select how to add assets:
              <ul className="list-disc list-inside ml-5 mt-1 space-y-0.5">
                <li>
                  <strong>Text Input</strong> — Fast entry using natural
                  language formats
                </li>
                <li>
                  <strong>Upload File</strong> — Import from CSV or Excel
                </li>
                <li>
                  <strong>Manual Entry</strong> — Add each asset individually
                </li>
                <li>
                  <strong>Use Template</strong> — Start with pre-built
                  strategies
                </li>
              </ul>
            </li>
            <li>
              <strong>Add Assets</strong> — Enter your portfolio assets
            </li>
            <li>
              <strong>Settings & Review</strong> — Configure options and review
            </li>
            <li>
              <strong>Create</strong> — Finalize and create your portfolio
            </li>
          </ol>
          <div className="mt-3">
            <p className="font-medium text-white/80 mb-1">
              Supported Text Input Formats
            </p>
            <div className="code-block space-y-0.5">
              <div>{`AAPL:40%, MSFT:30%, GOOGL:30%          # Colon with percentages`}</div>
              <div>{`AAPL 0.4, MSFT 0.3, GOOGL 0.3         # Space with decimals`}</div>
              <div>{`AAPL 40, MSFT 30, GOOGL 30             # Numbers > 1 (auto %)`}</div>
              <div>{`AAPL, MSFT, GOOGL                      # Equal weights`}</div>
            </div>
          </div>
          <div className="mt-3 space-y-1">
            <p className="font-medium text-white/80">Important Notes</p>
            <ul className="list-disc list-inside space-y-0.5">
              <li>Weights are automatically normalized to sum to 100%</li>
              <li>
                Use standard ticker symbols (AAPL, MSFT, GOOGL, etc.)
              </li>
              <li>Invalid tickers will be highlighted and excluded</li>
              <li>All portfolios are validated before creation</li>
            </ul>
          </div>
        </div>
      </Expander>

      {/* --- Progress --- */}
      {step <= 5 && (
        <div className="panel p-5">
          <h2 className="text-lg font-semibold text-white mb-3">
            Portfolio Creation
          </h2>
          <ProgressBar step={Math.min(step, 5)} total={5} />
        </div>
      )}

      {/* ═══════════ STEP 1 ═══════════ */}
      {step === 1 && (
        <div className="panel p-6 space-y-5">
          <h3 className="text-xl font-semibold text-white">
            Step 1: Portfolio Information
          </h3>

          <Expander title="What information do I need?">
            <div className="space-y-3">
              <p>
                <strong className="text-white/90">Portfolio Name</strong> — A
                unique name to identify your portfolio (required). Use
                descriptive names like &quot;Tech Growth Portfolio&quot; or
                &quot;Dividend Income&quot;.
              </p>
              <p>
                <strong className="text-white/90">Description</strong> —
                Optional notes about your investment strategy. Helps you
                remember the purpose of this portfolio.
              </p>
              <p>
                <strong className="text-white/90">Base Currency</strong> —
                Currently supports: USD, EUR, GBP, JPY, CAD, AUD. All
                calculations will be in this currency.
              </p>
              <p>
                <strong className="text-white/90">Initial Investment</strong> —
                Starting capital amount. Used to calculate number of shares.
                Minimum: $1.00.
              </p>
            </div>
          </Expander>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="label">
                Portfolio Name *
                <Tip text="Enter a unique name for your portfolio. Must be unique — cannot match existing portfolio names." />
              </label>
              <input
                className="input"
                placeholder="e.g., Tech Growth Portfolio"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div>
              <label className="label">
                Base Currency
                <Tip text="The currency for your portfolio. All values and calculations will be in this currency." />
              </label>
              <select
                className="input"
                value={baseCurrency}
                onChange={(e) => setBaseCurrency(e.target.value)}
              >
                {["USD", "EUR", "GBP", "JPY", "CAD", "AUD"].map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">
                Description
                <Tip text="Optional notes about your investment strategy. Helps you remember the purpose of this portfolio." />
              </label>
              <textarea
                className="input"
                placeholder="Optional description of your investment strategy"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
              />
            </div>
            <div>
              <label className="label">
                Initial Investment ($)
                <Tip text="Starting capital amount. Used to calculate number of shares for each position. Minimum: $1.00." />
              </label>
              <input
                className="input"
                type="number"
                min={1}
                step={1000}
                value={capital}
                onChange={(e) => setCapital(Number(e.target.value))}
              />
            </div>
          </div>

          {/* name validation feedback */}
          {name.trim() && nameStatus === "checking" && (
            <Alert type="info">Checking name availability...</Alert>
          )}
          {name.trim() && nameStatus === "available" && (
            <Alert type="success">Portfolio name is available</Alert>
          )}
          {name.trim() && nameStatus === "taken" && (
            <Alert type="error">
              A portfolio with this name already exists!
            </Alert>
          )}
          {name.trim() && nameStatus === "invalid" && (
            <Alert type="error">
              Portfolio name must be at most 100 characters
            </Alert>
          )}

          {/* navigation */}
          <div className="flex justify-between items-center pt-2">
            <button
              type="button"
              className="btn btn-danger"
              onClick={resetAll}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={!canProceedStep1}
              onClick={goNext}
            >
              Next Step →
            </button>
          </div>
        </div>
      )}

      {/* ═══════════ STEP 2 ═══════════ */}
      {step === 2 && (
        <div className="panel p-6 space-y-5">
          <h3 className="text-xl font-semibold text-white">
            Step 2: Choose Input Method
          </h3>

          <Expander title="Which method should I choose?">
            <div className="space-y-3">
              <p>
                <strong className="text-white/90">Text Input</strong> — Best
                for quick entry. Fastest way, supports multiple formats, good
                for 5-20 assets.
              </p>
              <p>
                <strong className="text-white/90">Upload File</strong> — Best
                for existing data. Import from CSV, perfect if you have a
                portfolio list.
              </p>
              <p>
                <strong className="text-white/90">Manual Entry</strong> — Best
                for precision. Add each asset one by one. Real-time ticker
                validation.
              </p>
              <p>
                <strong className="text-white/90">Use Template</strong> — Best
                for beginners. Pre-built strategies (Value, Growth, Quality,
                60/40, All Weather). Customizable.
              </p>
            </div>
          </Expander>

          <div className="label">
            How would you like to add assets?
            <Tip text="Choose the most convenient method for your data. You can always go back and change." />
          </div>

          <div className="radio-group">
            {(
              [
                ["text", "Text Input"],
                ["file", "Upload File"],
                ["manual", "Manual Entry"],
                ["template", "Use Template"],
              ] as [InputMethod, string][]
            ).map(([value, label]) => (
              <div
                key={value}
                className={`radio-option ${method === value ? "active" : ""}`}
                onClick={() => setMethod(value)}
              >
                <div className="radio-dot">
                  <div className="radio-dot-inner" />
                </div>
                <span className="text-sm text-white/80">{label}</span>
              </div>
            ))}
          </div>

          {/* method preview */}
          <div className="mt-2">
            {method === "text" && (
              <div className="space-y-2">
                <Alert type="info">
                  <strong>Fastest method</strong>: Enter ticker symbols with
                  weights
                </Alert>
                <div className="code-block">
                  AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, TSLA 10%
                </div>
                <p className="text-xs text-white/50">
                  Supported formats:{" "}
                  <code className="text-white/70">AAPL:30%</code>,{" "}
                  <code className="text-white/70">AAPL 0.3</code>,{" "}
                  <code className="text-white/70">AAPL 30</code>,{" "}
                  <code className="text-white/70">AAPL, MSFT</code> (equal)
                </p>
              </div>
            )}
            {method === "file" && (
              <div className="space-y-2">
                <Alert type="info">
                  <strong>From spreadsheet</strong>: Upload CSV files
                </Alert>
                <p className="text-xs text-white/50">
                  Required columns: <code className="text-white/70">ticker</code>
                  . Optional: <code className="text-white/70">weight</code>,{" "}
                  <code className="text-white/70">name</code>,{" "}
                  <code className="text-white/70">sector</code>
                </p>
              </div>
            )}
            {method === "manual" && (
              <div className="space-y-2">
                <Alert type="info">
                  <strong>Full control</strong>: Add each asset individually
                </Alert>
                <p className="text-xs text-white/50">
                  Best for: Detailed portfolio construction with custom settings
                </p>
              </div>
            )}
            {method === "template" && (
              <div className="space-y-2">
                <Alert type="info">
                  <strong>Quick start</strong>: Begin with proven strategies
                </Alert>
                <p className="text-xs text-white/50">
                  Available: Value Factor, Growth Factor, Quality Factor, 60/40,
                  All Weather, Tech Focus, and more
                </p>
              </div>
            )}
          </div>

          <div className="flex justify-between items-center pt-2">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={goPrev}
            >
              ← Previous
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={goNext}
            >
              Next Step →
            </button>
          </div>
        </div>
      )}

      {/* ═══════════ STEP 3 ═══════════ */}
      {step === 3 && (
        <div className="panel p-6 space-y-5">
          <h3 className="text-xl font-semibold text-white">
            Step 3: Add Your Assets
          </h3>

          {/* ---- TEXT INPUT ---- */}
          {method === "text" && (
            <div className="space-y-4">
              <p className="text-sm text-white/70">
                <strong className="text-white/90">
                  Enter your portfolio assets
                </strong>{" "}
                (one of these formats):
              </p>

              <Expander title="How does text input work?">
                <div className="space-y-2">
                  <p className="font-medium text-white/80">
                    Supported Formats:
                  </p>
                  <ul className="list-disc list-inside space-y-1">
                    <li>
                      <code className="text-white/70">
                        AAPL:40%, MSFT:30%, GOOGL:30%
                      </code>{" "}
                      — Colon with percentages
                    </li>
                    <li>
                      <code className="text-white/70">
                        AAPL 0.4, MSFT 0.3, GOOGL 0.3
                      </code>{" "}
                      — Space with decimals (0.0-1.0)
                    </li>
                    <li>
                      <code className="text-white/70">
                        AAPL 40, MSFT 30, GOOGL 30
                      </code>{" "}
                      — Numbers &gt; 1 (auto treated as %)
                    </li>
                    <li>
                      <code className="text-white/70">AAPL, MSFT, GOOGL</code>{" "}
                      — Equal weights
                    </li>
                  </ul>
                  <p className="mt-2">
                    <strong className="text-white/80">Tips:</strong> Weights
                    are auto-normalized. Invalid tickers are highlighted and
                    excluded. You can mix formats.
                  </p>
                </div>
              </Expander>

              {/* example tabs */}
              <div className="tab-bar">
                {[
                  ["Percentages", "AAPL:40%, MSFT:30%, GOOGL:30%"],
                  ["Decimals", "AAPL 0.4, MSFT 0.3, GOOGL 0.3"],
                  ["Numbers", "AAPL 40, MSFT 30, GOOGL 30"],
                  ["Equal Weight", "AAPL, MSFT, GOOGL"],
                ].map(([label, example]) => (
                  <button
                    key={label}
                    type="button"
                    className="tab-btn"
                    onClick={() => setTextInput(example)}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div>
                <label className="label">Portfolio Assets:</label>
                <textarea
                  className="input"
                  rows={5}
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder={`Enter tickers and weights here...\nExample: AAPL 30%, MSFT 25%, GOOGL 20%, AMZN 15%, TSLA 10%`}
                />
              </div>

              <button
                type="button"
                className="btn btn-primary w-full"
                onClick={parseAndValidate}
                disabled={!textInput.trim() || validating}
              >
                {validating ? "Validating tickers..." : "Parse & Validate"}
              </button>

              {/* validation results */}
              {positions.length > 0 && (
                <div className="space-y-3">
                  {Object.keys(validationMap).length > 0 && (
                    <>
                      {positions.filter(
                        (p) => validationMap[p.ticker]?.valid,
                      ).length > 0 && (
                        <Alert type="success">
                          Parsed {validPositions.length} valid asset
                          {validPositions.length !== 1 ? "s" : ""} successfully
                        </Alert>
                      )}
                      {positions.filter(
                        (p) => validationMap[p.ticker]?.valid === false,
                      ).length > 0 && (
                        <Alert type="warning">
                          Unknown tickers (will be excluded):{" "}
                          {positions
                            .filter(
                              (p) => validationMap[p.ticker]?.valid === false,
                            )
                            .map((p) => p.ticker)
                            .join(", ")}
                        </Alert>
                      )}
                    </>
                  )}

                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Weight</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {positions.map((p) => {
                        const v = validationMap[p.ticker];
                        return (
                          <tr key={p.ticker}>
                            <td className="font-mono">{p.ticker}</td>
                            <td>{pct(p.weight)}</td>
                            <td>
                              {v === undefined ? (
                                <span className="text-white/40">—</span>
                              ) : v.valid ? (
                                <span className="text-[var(--ok)]">
                                  ✓ Valid
                                </span>
                              ) : (
                                <span className="text-[var(--danger)]">
                                  ✗ Invalid
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>

                  {/* metrics */}
                  <div className="grid grid-cols-3 gap-3">
                    <MetricCard
                      label="Assets"
                      value={String(validPositions.length)}
                    />
                    <MetricCard
                      label="Total Weight"
                      value={pct(
                        validPositions.reduce((s, p) => s + p.weight, 0),
                      )}
                    />
                    <MetricCard
                      label="Status"
                      value={
                        Math.abs(
                          validPositions.reduce((s, p) => s + p.weight, 0) -
                            1.0,
                        ) < 0.001
                          ? "Perfect"
                          : "Will normalize"
                      }
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ---- FILE UPLOAD ---- */}
          {method === "file" && (
            <div className="space-y-4">
              <p className="text-sm text-white/70">
                <strong className="text-white/90">
                  Upload a CSV file
                </strong>{" "}
                with your portfolio data
              </p>

              <Expander title="What file format do I need?">
                <div className="space-y-2">
                  <p>
                    <strong className="text-white/80">
                      File Requirements:
                    </strong>
                  </p>
                  <ul className="list-disc list-inside space-y-1">
                    <li>Supported formats: CSV</li>
                    <li>
                      Required column:{" "}
                      <code className="text-white/70">ticker</code> (or{" "}
                      <code className="text-white/70">symbol</code>)
                    </li>
                    <li>
                      Optional columns:{" "}
                      <code className="text-white/70">weight</code>,{" "}
                      <code className="text-white/70">name</code>
                    </li>
                  </ul>
                  <div className="code-block mt-2">
                    <div>ticker,weight</div>
                    <div>AAPL,30</div>
                    <div>MSFT,25</div>
                    <div>GOOGL,20</div>
                  </div>
                </div>
              </Expander>

              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="input"
              />

              {fileName && (
                <Alert type="success">
                  File loaded: <strong>{fileName}</strong> —{" "}
                  {filePositions.length} rows parsed
                </Alert>
              )}

              {filePositions.length > 0 && (
                <div className="space-y-3">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filePositions.map((p) => (
                        <tr key={p.ticker}>
                          <td className="font-mono">{p.ticker}</td>
                          <td>{pct(p.weight)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard
                      label="Assets"
                      value={String(filePositions.length)}
                    />
                    <MetricCard
                      label="Total Weight"
                      value={pct(
                        filePositions.reduce((s, p) => s + p.weight, 0),
                      )}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ---- MANUAL ENTRY ---- */}
          {method === "manual" && (
            <div className="space-y-4">
              <p className="text-sm text-white/70">
                <strong className="text-white/90">Add assets manually</strong>{" "}
                for full control over your portfolio
              </p>

              <Expander title="How does manual entry work?">
                <div className="space-y-2">
                  <p className="font-medium text-white/80">
                    Manual Entry Process:
                  </p>
                  <ol className="list-decimal list-inside space-y-1">
                    <li>Enter ticker symbol (e.g., AAPL)</li>
                    <li>Enter weight as percentage (e.g., 30 for 30%)</li>
                    <li>
                      Click &quot;Validate Ticker&quot; to check validity
                    </li>
                    <li>Click &quot;Add Asset&quot; to add to portfolio</li>
                    <li>Repeat for each asset</li>
                  </ol>
                  <p className="mt-2">
                    <strong className="text-white/80">Features:</strong>{" "}
                    Real-time ticker validation, see current price, view total
                    weight. Weights are auto-normalized to 100%.
                  </p>
                </div>
              </Expander>

              <div className="panel p-4 space-y-4">
                <h4 className="text-base font-semibold text-white">
                  Add New Asset
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="label">
                      Ticker *
                      <Tip text="Stock symbol, e.g. AAPL for Apple, MSFT for Microsoft. Use standard ticker symbols." />
                    </label>
                    <input
                      className="input"
                      placeholder="e.g., AAPL"
                      value={manualTicker}
                      onChange={(e) => {
                        setManualTicker(e.target.value);
                        setManualValidation(null);
                      }}
                    />
                  </div>
                  <div>
                    <label className="label">
                      Weight (%)
                      <Tip text="Percentage allocation (will be normalized to sum to 100%). E.g., 30 for 30% of portfolio." />
                    </label>
                    <input
                      className="input"
                      type="number"
                      min={0.1}
                      max={100}
                      step={0.1}
                      value={manualWeight}
                      onChange={(e) =>
                        setManualWeight(Number(e.target.value))
                      }
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={addManualPosition}
                    disabled={
                      !manualTicker.trim() ||
                      manualWeight <= 0 ||
                      manualValidating
                    }
                  >
                    {manualValidating ? "Validating..." : "Add Asset"}
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={validateManualTicker}
                    disabled={!manualTicker.trim() || manualValidating}
                  >
                    {manualValidating ? "Checking..." : "Validate Ticker"}
                  </button>
                </div>

                {/* validation feedback */}
                {manualValidation && (
                  <>
                    {manualValidation.valid ? (
                      <Alert type="success">
                        {manualValidation.ticker} is valid
                        {manualValidation.price
                          ? ` — Current price: $${manualValidation.price.toFixed(2)}`
                          : " — Price unavailable"}
                      </Alert>
                    ) : (
                      <Alert type="error">
                        {manualValidation.ticker} is not a valid ticker
                        {positions.some(
                          (p) => p.ticker === manualValidation!.ticker,
                        )
                          ? " (already in portfolio)"
                          : ""}
                      </Alert>
                    )}
                  </>
                )}
              </div>

              {/* current assets list */}
              {positions.length > 0 && (
                <div className="space-y-3">
                  <h4 className="text-base font-semibold text-white">
                    Current Assets
                  </h4>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Weight</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {positions.map((p, i) => (
                        <tr key={p.ticker}>
                          <td className="font-mono">{p.ticker}</td>
                          <td>{pct(p.weight)}</td>
                          <td>
                            <button
                              type="button"
                              className="text-[var(--danger)] text-xs hover:underline"
                              onClick={() =>
                                setPositions((prev) =>
                                  prev.filter((_, idx) => idx !== i),
                                )
                              }
                            >
                              Remove
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  <div className="grid grid-cols-3 gap-3">
                    <MetricCard
                      label="Total Assets"
                      value={String(positions.length)}
                    />
                    <MetricCard
                      label="Total Weight"
                      value={pct(totalWeight)}
                    />
                    <MetricCard
                      label="Status"
                      value={
                        Math.abs(totalWeight - 1.0) < 0.001
                          ? "Perfect"
                          : "Will normalize"
                      }
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ---- TEMPLATE SELECTION ---- */}
          {method === "template" && (
            <div className="space-y-4">
              <p className="text-sm text-white/70">
                <strong className="text-white/90">
                  Start with a proven strategy
                </strong>{" "}
                and customize as needed
              </p>

              <Expander title="What are portfolio templates?">
                <div className="space-y-3">
                  <p className="font-medium text-white/80">
                    Factor-Based Strategies:
                  </p>
                  <ul className="list-disc list-inside space-y-1">
                    <li>
                      <strong>Value Factor</strong> — Undervalued companies
                      with low P/E
                    </li>
                    <li>
                      <strong>Quality Factor</strong> — High ROE, low debt
                    </li>
                    <li>
                      <strong>Growth Factor</strong> — Fast-growing companies
                    </li>
                    <li>
                      <strong>Low Volatility</strong> — Beta &lt; 0.8
                    </li>
                    <li>
                      <strong>Dividend Factor</strong> — Yield &gt; 3%
                    </li>
                  </ul>
                  <p className="font-medium text-white/80 mt-2">
                    Classic Strategies:
                  </p>
                  <ul className="list-disc list-inside space-y-1">
                    <li>
                      <strong>60/40 Portfolio</strong> — Balanced allocation
                    </li>
                    <li>
                      <strong>All Weather</strong> — Multi-asset diversification
                    </li>
                    <li>
                      <strong>Tech Focus</strong> — Technology concentration
                    </li>
                  </ul>
                  <p className="mt-2">
                    All templates can be customized after selection.
                  </p>
                </div>
              </Expander>

              <div>
                <label className="label">
                  Choose a template:
                  <Tip text="Select a base strategy to start with. All templates can be customized after selection." />
                </label>
                <select
                  className="input"
                  value={selectedTemplate}
                  onChange={(e) => {
                    setSelectedTemplate(e.target.value);
                    setCustomizeTemplate(false);
                  }}
                >
                  {Object.keys(TEMPLATES).map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>

              {/* template details */}
              {selectedTemplate && TEMPLATES[selectedTemplate] && (
                <div className="space-y-3">
                  <Alert type="info">
                    <strong>{selectedTemplate}</strong>:{" "}
                    {TEMPLATES[selectedTemplate].description}
                  </Alert>
                  <div className="code-block">
                    {TEMPLATES[selectedTemplate].assets}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {TEMPLATES[selectedTemplate].tags.map((tag) => (
                      <span key={tag} className="tag">
                        {tag}
                      </span>
                    ))}
                  </div>

                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={customizeTemplate}
                      onChange={(e) => {
                        setCustomizeTemplate(e.target.checked);
                        if (e.target.checked) {
                          setCustomTemplateText(
                            TEMPLATES[selectedTemplate].assets,
                          );
                        }
                      }}
                    />
                    Customize template
                    <Tip text="Modify the template allocation to suit your needs. Change weights, add/remove assets." />
                  </label>

                  {customizeTemplate && (
                    <div>
                      <label className="label">Modify the template:</label>
                      <textarea
                        className="input"
                        rows={3}
                        value={customTemplateText}
                        onChange={(e) => setCustomTemplateText(e.target.value)}
                      />
                    </div>
                  )}

                  {/* preview */}
                  {positions.length > 0 && (
                    <div className="space-y-2">
                      <Alert type="success">
                        Template contains {positions.length} assets
                      </Alert>
                      <Expander title="Template Preview">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Ticker</th>
                              <th>Weight</th>
                            </tr>
                          </thead>
                          <tbody>
                            {positions.map((p) => (
                              <tr key={p.ticker}>
                                <td className="font-mono">{p.ticker}</td>
                                <td>{pct(p.weight)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </Expander>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* step 3 navigation */}
          <div className="flex justify-between items-center pt-2">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={goPrev}
            >
              ← Previous
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={!canProceedStep3}
              onClick={goNext}
            >
              Next Step →
            </button>
          </div>
        </div>
      )}

      {/* ═══════════ STEP 4 ═══════════ */}
      {step === 4 && (
        <div className="panel p-6 space-y-5">
          <h3 className="text-xl font-semibold text-white">
            Step 4: Portfolio Settings & Review
          </h3>

          <Expander title="What settings should I configure?">
            <div className="space-y-3">
              <p>
                <strong className="text-white/90">Portfolio Options:</strong>
              </p>
              <ul className="list-disc list-inside space-y-1">
                <li>
                  <strong>Fetch company information</strong> — Get company
                  names, sectors
                </li>
                <li>
                  <strong>Auto-normalize weights</strong> — Adjust to sum to
                  100%
                </li>
                <li>
                  <strong>Update current prices</strong> — Fetch latest prices
                </li>
                <li>
                  <strong>Calculate share quantities</strong> — Based on initial
                  investment
                </li>
              </ul>
              <p className="mt-2">
                <strong className="text-white/90">Cash Management:</strong>{" "}
                Percentage to keep in cash for liquidity or entry points.
              </p>
            </div>
          </Expander>

          {/* Portfolio Options */}
          <div>
            <h4 className="text-base font-semibold text-white mb-3">
              Portfolio Options
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={fetchInfo}
                  onChange={(e) => setFetchInfo(e.target.checked)}
                />
                Fetch company information
                <Tip text="Automatically get company names, sectors, and market data for all assets." />
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={updatePrices}
                  onChange={(e) => setUpdatePrices(e.target.checked)}
                />
                Update current prices
                <Tip text="Fetch latest market prices for all assets in the portfolio." />
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={autoNormalize}
                  onChange={(e) => setAutoNormalize(e.target.checked)}
                />
                Auto-normalize weights
                <Tip text="Automatically adjust weights to sum to 100%. If weights add up to 80%, each will be scaled proportionally." />
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={calculateShares}
                  onChange={(e) => setCalculateShares(e.target.checked)}
                />
                Calculate share quantities
                <Tip text="Calculate number of shares based on initial investment and current prices." />
              </label>
            </div>
          </div>

          {/* Cash Management */}
          <div className="border-t border-white/10 pt-4">
            <h4 className="text-base font-semibold text-white mb-3">
              Cash Management
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
              <div>
                <label className="label">
                  Planned Cash Allocation: {cashAllocation}%
                  <Tip text="Percentage to intentionally keep in cash. Useful for maintaining liquidity or waiting for better entry points." />
                </label>
                <input
                  type="range"
                  min={0}
                  max={50}
                  step={1}
                  value={cashAllocation}
                  onChange={(e) => setCashAllocation(Number(e.target.value))}
                />
              </div>
              {cashAllocation > 0 && (
                <div className="space-y-2">
                  <MetricCard
                    label="Cash Amount"
                    value={`$${(capital * (cashAllocation / 100)).toLocaleString()}`}
                  />
                  <Alert type="info">
                    Remaining for investments: {100 - cashAllocation}%
                  </Alert>
                </div>
              )}
            </div>
          </div>

          {/* Portfolio Mode */}
          <div className="border-t border-white/10 pt-4">
            <h4 className="text-base font-semibold text-white mb-3">
              Portfolio Mode
            </h4>
            <div className="radio-group">
              <div
                className={`radio-option ${portfolioMode === "buy_hold" ? "active" : ""}`}
                onClick={() => setPortfolioMode("buy_hold")}
              >
                <div className="radio-dot">
                  <div className="radio-dot-inner" />
                </div>
                <div>
                  <span className="text-sm text-white/90 font-medium">
                    Buy-and-Hold
                  </span>
                  <p className="text-xs text-white/40 mt-0.5">
                    Simple mode — positions remain fixed. Fast analysis, best
                    for quick backtests.
                  </p>
                </div>
              </div>
              <div
                className={`radio-option ${portfolioMode === "transactions" ? "active" : ""}`}
                onClick={() => setPortfolioMode("transactions")}
              >
                <div className="radio-dot">
                  <div className="radio-dot-inner" />
                </div>
                <div>
                  <span className="text-sm text-white/90 font-medium">
                    With Transactions
                  </span>
                  <p className="text-xs text-white/40 mt-0.5">
                    Advanced mode — track all real trades over time. More
                    accurate performance tracking.
                  </p>
                </div>
              </div>
            </div>

            {portfolioMode === "transactions" && (
              <div className="mt-3 space-y-3">
                <div>
                  <label className="label">
                    Portfolio inception date
                    <Tip text="Initial BUY/DEPOSIT transactions use this date." />
                  </label>
                  <input
                    type="date"
                    className="input"
                    value={inceptionDate}
                    max={new Date().toISOString().slice(0, 10)}
                    onChange={(e) => setInceptionDate(e.target.value)}
                  />
                </div>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={createInitialTxns}
                    onChange={(e) => setCreateInitialTxns(e.target.checked)}
                  />
                  Create initial transactions from positions
                  <Tip text="If checked, initial BUY transactions will be created for each position when portfolio is created. You can add more transactions later." />
                </label>
                <Alert type="info">
                  Transactions dated <strong>{inceptionDate}</strong>. Overview
                  holdings use the ledger after creation.
                </Alert>
              </div>
            )}

            {portfolioMode === "buy_hold" && (
              <Alert type="info">
                Buy-and-Hold: positions only (today). For a dated ledger, choose
                &quot;With Transactions&quot;.
              </Alert>
            )}
          </div>

          {/* Portfolio Summary */}
          <div className="border-t border-white/10 pt-4">
            <h4 className="text-base font-semibold text-white mb-3">
              Portfolio Summary
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <MetricCard label="Name" value={name || "N/A"} />
              <MetricCard label="Currency" value={baseCurrency} />
              <MetricCard
                label="Initial Value"
                value={`$${capital.toLocaleString()}`}
              />
              <MetricCard label="Method" value={method} />
              <MetricCard
                label="Assets"
                value={String(
                  method === "file" ? filePositions.length : positions.length,
                )}
              />
              <MetricCard
                label="Total Weight"
                value={pct(
                  method === "file"
                    ? filePositions.reduce((s, p) => s + p.weight, 0)
                    : totalWeight,
                )}
              />
            </div>
          </div>

          <div className="flex justify-between items-center pt-2">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={goPrev}
            >
              ← Previous
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={goNext}
            >
              Create Portfolio →
            </button>
          </div>
        </div>
      )}

      {/* ═══════════ STEP 5 ═══════════ */}
      {step === 5 && (
        <div className="panel p-6 space-y-5">
          <h3 className="text-xl font-semibold text-white">
            Step 5: Confirm & Create
          </h3>

          <div className="panel p-4 space-y-3">
            <p className="text-white/80">
              Ready to create portfolio:{" "}
              <strong className="text-white">{name}</strong>
            </p>
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div>
                <span className="text-white/40">Currency:</span>{" "}
                <span className="text-white">{baseCurrency}</span>
              </div>
              <div>
                <span className="text-white/40">Capital:</span>{" "}
                <span className="text-white">${capital.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-white/40">Assets:</span>{" "}
                <span className="text-white">
                  {method === "file" ? filePositions.length : positions.length}
                </span>
              </div>
            </div>
          </div>

          {errorMsg && <Alert type="error">{errorMsg}</Alert>}

          <div className="flex justify-between items-center pt-2">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={goPrev}
            >
              ← Back to Review
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={creating}
              onClick={handleCreate}
            >
              {creating ? "Creating..." : "Create Portfolio"}
            </button>
          </div>
        </div>
      )}

      {/* ═══════════ STEP 6 (success) ═══════════ */}
      {step === 6 && createdPortfolio && (
        <div className="panel p-6 space-y-5">
          <Alert type="success">
            Portfolio created successfully!
          </Alert>

          <h3 className="text-xl font-semibold text-white">
            Portfolio Created
          </h3>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard
              label="Total Assets"
              value={String(createdPortfolio.positions.length)}
            />
            <MetricCard
              label="Total Value"
              value={`$${createdPortfolio.starting_capital.toLocaleString()}`}
            />
            <MetricCard
              label="Created"
              value={new Date().toLocaleTimeString()}
            />
            <MetricCard
              label="Currency"
              value={createdPortfolio.base_currency}
            />
          </div>

          {/* Asset Allocation */}
          <div>
            <h4 className="text-base font-semibold text-white mb-3">
              Asset Allocation
            </h4>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Weight</th>
                  <th>Shares</th>
                </tr>
              </thead>
              <tbody>
                {createdPortfolio.positions.map((p) => (
                  <tr key={p.ticker}>
                    <td className="font-mono">{p.ticker}</td>
                    <td>{pct(p.weight_target)}</td>
                    <td>{p.shares.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* What's Next */}
          <div>
            <h4 className="text-base font-semibold text-white mb-3">
              What&apos;s Next?
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <button
                type="button"
                className="btn btn-primary w-full"
                onClick={() => router.push("/analysis")}
              >
                Analyze Portfolio
              </button>
              <button
                type="button"
                className="btn btn-secondary w-full"
                onClick={resetAll}
              >
                Create Another
              </button>
              <button
                type="button"
                className="btn btn-ghost w-full"
                onClick={() => router.push("/")}
              >
                Go to Dashboard
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════ DRAFT ASSETS (persistent sidebar-like view) ═══════════ */}
      {step >= 3 && step <= 5 && (
        <div className="panel p-5">
          <h2 className="text-base font-semibold text-white mb-3">
            Draft Assets
            <span className="text-white/40 text-sm font-normal ml-2">
              ({method === "file" ? filePositions.length : positions.length}{" "}
              assets, {pct(method === "file" ? filePositions.reduce((s, p) => s + p.weight, 0) : totalWeight)} total)
            </span>
          </h2>
          <div className="space-y-1.5 text-sm max-h-64 overflow-y-auto">
            {(method === "file" ? filePositions : positions).length === 0 ? (
              <div className="text-white/40">No assets yet.</div>
            ) : (
              (method === "file" ? filePositions : positions).map(
                (pos, index) => (
                  <div
                    key={`${pos.ticker}-${index}`}
                    className="flex items-center justify-between py-2 px-3 rounded-lg bg-[var(--surface)] border border-white/5"
                  >
                    <span className="font-mono text-white/80">
                      {pos.ticker}
                    </span>
                    <span className="text-white/50">{pct(pos.weight)}</span>
                  </div>
                ),
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}
