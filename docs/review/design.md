## Design

Design gate, round 1, 2026-11-05, branch `chore/audit`. App: Next.js frontend at http://127.0.0.1:33499 (WILD MARKET), production stack already running (not restarted).
Tool: Playwright (Python 1.x library, installed Google Chrome channel, headless). Playwright MCP was not available; the same checks were scripted against the running app. Browser locale = system (ru-RU), which affects number formatting (see D-10).
Signed in as `audit-design@example.com`. A portfolio "Design Audit Portfolio" (AAPL 50% / MSFT 50%, With Transactions) was created through `/create` text input, so data pages are shown filled, not empty. A GOOGL x3 BUY was added during the F-2 check.

**Verdict: FIX** - P0 0 · P1 3 · P2 5 · P3 7. Ban hits: 0. App console errors when signed in: 0; anonymous pages: 2 per load (401 on `/api/auth/me`, D-12).

### F-2 / F-3 re-check (from docs/QA.md)
| Id | Result | Evidence |
|---|---|---|
| F-2 (auto-filled price fails `step=0.01`, Add does nothing) | **PASS** | `/portfolio/<id>/transactions` @1440: date 2025-06-02, ticker GOOGL, shares 3, price **not typed**. Auto-filled price input `168.24` (`step=0.01`, `checkValidity()=true`), hint "Close on 2025-06-02: $168.24", form valid. Click Add Transaction -> `POST /api/proxy/portfolios/<id>/transactions` 200, table rows 3 -> 4, 1 GOOGL row. Screenshots `docs/design/flow/f2-before-submit-1440.png`, `docs/design/flow/f2-after-submit-1440.png`. |
| F-3 (portfolio overview horizontal overflow at 375) | **PASS** | `/portfolio/<id>` @375: `scrollWidth 375 == clientWidth 375`; holdings table scrolls inside its `overflow-x-auto` box. Also no overflow at 768/1440. `docs/design/portfolio-overview-375.png`. The same defect still exists on two other pages: D-01, D-02. |

### Findings
| Id | Sev | Page @ width | Evidence | Problem. Fix. |
|---|---|---|---|---|
| D-01 | P1 | portfolio-transactions @375, @768 | `scrollWidth 775` vs 375/768; `docs/design/portfolio-transactions-375.png`, `-768.png`; `frontend/src/app/portfolio/[id]/transactions/page.tsx:102` | Transactions table is not wrapped, so the whole page scrolls sideways (header/background stop at 375, table runs to 775). Wrap the table in `<div className="overflow-x-auto">` as done on the overview (`app/portfolio/[id]/page.tsx:200`). |
| D-02 | P1 | portfolios @375, @768 | `scrollWidth 795` @375, `821` @768; `docs/design/portfolios-375.png`, `-768.png`; `frontend/src/app/portfolios/page.tsx:615` | Desktop table renders on mobile together with the mobile cards (`:675 md:hidden`), so the list is shown twice and the page scrolls sideways; at 768 a hidden `.tooltip-bubble` (Export All) also sticks out to 821. Hide the table below `md` (`hidden md:block`) and wrap it in `overflow-x-auto`; see D-07 for the tooltip. |
| D-03 | P1 | create (step 2 method, step 4 portfolio mode) @all | 30 Tab presses on step 2 cycle nav -> expanders -> Previous/Next and never reach a `.radio-option`; all 4 have `tabIndex=-1`, `role=null`. `docs/design/flow/create-step2-1440.png`; `frontend/src/app/create/page.tsx:979, 1808, 1825` | Choice cards are `<div onClick>`: keyboard users cannot pick File/Manual/Template or Buy-and-Hold vs With Transactions (WCAG 2.1.1). Use native `<input type="radio">` inside a `<label>` (or `role="radio"` + `tabIndex=0` + Space/Arrow handling) in a `role="radiogroup"`. |
| D-04 | P2 | all signed-in pages @375 | `docs/design/dashboard-375.png` (any `*-375.png`) | 10 nav items wrap into 5 rows (~255 px of an 812 px viewport) and the logo breaks into "WILD / MARKET"; content starts below the fold line on short phones. Collapse nav into a menu button below `md` (`frontend/src/components/main-nav.tsx:38`). |
| D-05 | P2 | many @375 | `docs/design/metrics.json` key `small`: nav links 38 px high; transactions Delete 55x26; portfolios View/Edit/Analyze/Delete 30 px; "← Dashboard" 75x16; portfolio sub-tabs 35 px; analysis comparison chips 30 px; forecasting method chips 34 px; "Reset To Method Default" 26 px; checkboxes 13x13 | Touch targets under 44 px on the mobile layout. Give interactive elements `min-height: 44px` (or padded hit area) below `md`; larger checkbox hit area via the label. |
| D-06 | P2 | create, analysis, optimization, risk, forecasting, portfolio-transactions, portfolios @all | `metrics.json` key `unl` (e.g. transactions: date, type select, ticker, shares, price, fees, notes; risk: 6; optimization: 5 incl. 3 range sliders) | Inputs have a visible label but no programmatic association (no `label[for]`, no wrapping label, no `aria-label`); screen readers announce "edit text". Add `id` + `htmlFor` (e.g. `transaction-form.tsx` labels), `aria-label` on sliders. |
| D-07 | P2 | analysis @375, @768; portfolios @768 | `scrollWidth 377` @375, `784` @768 on `/analysis`, caused by `SPAN.tooltip-bubble`; `frontend/src/app/globals.css:335` | Hidden tooltips still take layout and widen the page (small sideways scroll). Hide with `display:none`/`visibility:hidden` until hover, or clamp/flip position near the right edge. |
| D-08 | P2 | billing @all | `docs/design/billing-1440.png`; `frontend/src/app/billing/page.tsx:142, 179` | End users see operator instructions: "Set STRIPE_SECRET_KEY and STRIPE_PRICE_ID_PRO…" and `stripe listen --forward-to localhost:8000/stripe/webhook`. Show a neutral "Upgrades are not available right now"; keep dev hints behind `NODE_ENV==="development"`. |
| D-09 | P3 | risk, forecasting, optimization, analysis | `docs/design/risk-375.png`, `forecasting-375.png`, `optimization-375.png`; `app/risk/page.tsx:385`, `app/forecasting/page.tsx:283, 436`, `app/opti/page.tsx:1119`, `app/analysis/page.tsx:2579` | Internal copy leaks into UI: "aligned with the Streamlit risk page", "default in Streamlit", `meta.training_start`, raw param `target_return_as_floor`. Rewrite as user-facing text. |
| D-10 | P3 | portfolio-overview @1440 | `docs/design/portfolio-overview-1440.png` | "$-2,00" (sign after the currency symbol), "-0.0%" (negative zero), and mixed separators: money uses the browser locale ("$99 998,00") while percents use "." ("50.0%"). Put the sign first (`-$2.00`), clamp -0 to 0, pick one explicit locale for all number formatting. |
| D-11 | P3 | home, optimization @all | `docs/design/home-1440.png`, `optimization-375.png` vs `dashboard-1440.png` | H1 style differs: regular weight inside a panel on Home/Optimization, bold without a panel elsewhere. Use one page-header pattern. |
| D-12 | P3 | home, login, register @all | `metrics.json` key `console`: `HTTP 401 /api/auth/me` + "Failed to load resource … 401" on every anonymous load | Console error noise for signed-out visitors. Skip the `/me` call without a session cookie, or return 200 `{user:null}`. |
| D-13 | P3 | create, analysis, transactions, portfolios @1440 | `metrics.json` key `contrast`: "?" tooltip glyphs `rgba(255,255,255,.4)` 10 px = 3.71-3.77:1 | Below 4.5:1 for text (passes 3:1 as an icon). Raise to `white/60`. Other flagged items were hidden tooltips (opacity 0) or disabled buttons and are not findings. |
| D-14 | P3 | settings @375; forecasting @all | `docs/design/settings-375.png`, `forecasting-375.png` | "Billing & plan" and "Sign out" buttons touch with no gap; forecasting method checkboxes render native white while the rest of the app uses accent-styled checkboxes. Add `gap-2`; apply the shared checkbox style (`accent-color`). |
| D-15 | P3 | home @all | `docs/design/home-1440.png`, `home-375.png` | No primary CTA: Sign in / Register / Dashboard are three equal ghost buttons, and "Dashboard" is offered to anonymous visitors. Make Register the accent button; drop Dashboard when signed out. |

### Checks run
| Check | Result |
|---|---|
| Screenshots 375x812, 768x1024, 1440x900, full page, 14 pages (12 requested + portfolio overview + transactions) | done, 42 PNGs in `docs/design/` |
| Wizard flow evidence (create steps 2-5 + result) | `docs/design/flow/create-*.png` |
| Slop bans (`design/system.md`) | 0 hits: no AI gradients/blobs (background is a faint navy radial), not Inter-only (Geist + Geist Mono), no emoji icons, no lorem, no pure #000/#fff slabs (`--background #090b10`, `--foreground #e7e7ef`) |
| Fidelity to own visual language (no `docs/design.md` exists; judged vs `globals.css` tokens) | mostly consistent dark panels, lavender accent, 10-12 px radius; deviations D-11, D-14. `grep` finds 96 hardcoded `#rrggbb` in `.tsx` (mostly chart colors) - not a finding on its own |
| Horizontal overflow @375 (`scrollWidth > clientWidth`) | fails: portfolios, portfolio-transactions, analysis (D-01, D-02, D-07); all other pages pass |
| Touch targets >= 44 px @375 | fails (D-04, D-05) |
| Focus visible (keyboard Tab on /login, /dashboard) | pass: nav links and buttons show a ring (`docs/design/flow/focus-login-nav-crop.png`, `focus-dashboard-btn-1440.png`), inputs get accent border + glow (`focus-login-email-1440.png`) |
| Keyboard reachability | fail on create wizard choice cards (D-03) |
| Labels / alt text | labels fail (D-06); 0 `<img>` without `alt` |
| Contrast (automated walk of text nodes @1440, initial state) | only D-13; contrast inside opened expanders/results (after Calculate) not measured |
| States | empty/idle states present (analysis "Click Calculate Metrics…", billing not-configured notice); loading/error states of analytics runs not exercised |
| Reduced motion | no `prefers-reduced-motion` rule, but motion is only 0.15-0.3 s hover/width transitions, no heavy motion; not a finding |
| Console errors | signed-in pages: 0; anonymous pages: 2 per load (D-12). Note: the host has Kaspersky injecting scripts into Chrome (`gc.kis.v2.scr.kaspersky-labs.com`); no errors came from it |

### Checks skipped
- impeccable: not linked (`.claude/skills/impeccable` absent).
- Playwright MCP: not available; used the Playwright Python library with installed Chrome instead (same browser engine, real screenshots).
- Running analyses (Calculate Metrics, VaR, Optimization, Forecasting) and screenshotting their result charts: not run in this round, so result-state layout and chart colors are not reviewed.
- `prefers-reduced-motion` emulation: skipped (no heavy motion found by grep and on screen).
- Round 2 re-review: not run (no fixes yet; reviewers do not fix).

## Round 2 (2026-11-05, fix commit `81bd623`)
Same method (Playwright Python + installed Chrome, same account, portfolio `7fcf5c18-…`). Re-shot (overwritten): `docs/design/portfolio-transactions-{375,768,1440}.png`, `portfolios-*.png`, `billing-*.png`, `create-*.png`; keyboard evidence `docs/design/flow/r2-create-step2-focus-1440.png`, `docs/design/flow/r2-create-step4-focus-1440.png`; measurements `docs/design/metrics-round2.json`. The transactions table now has 10 rows (GOOGL dividends were added by the ledger after the F-2 BUY).

| Id | Result | Evidence |
|---|---|---|
| D-01 | **PASS** | `/portfolio/<id>/transactions`: page `scrollWidth` = 375 / 768 / 1440 (= `clientWidth`). Table parent is `overflow-x-auto`: @375 box 327 px wide, table scrollWidth 757; @768 box 720, table 757; @1440 box 1152, no inner scroll. |
| D-02 | **FAIL (partial)** | Fixed: table wrapper `hidden md:block overflow-x-auto` -> @375 table hidden (width 0), mobile cards `display:block`, no duplicate list; @768 and @1440 table visible (1440: 982 px, cards `display:none`). Still failing: page `scrollWidth` **467 @375** and **821 @768**. Causes: the `Refresh / Export All` button row does not wrap (`DIV.flex gap-2`, `BUTTON.btn-ghost` right edge 391 > 375, `frontend/src/app/portfolios/page.tsx:539`) and the hidden Export All `.tooltip-bubble` (right edge 467 @375, 821 @768; same cause as D-07, `globals.css:335`). Fix: `flex flex-wrap gap-2` on the row; hide `.tooltip-bubble` with `display:none`/`visibility:hidden` until hover, or anchor it right near the edge. New P3: on mobile the table panel remains as an empty box showing only "1 portfolio" (`docs/design/portfolios-375.png`); move the count outside the hidden wrapper or hide the panel. |
| D-03 | **PASS** | Step 2: all 4 cards are `role="radio"` tab stops with `aria-checked`. Tab presses counted from a click on the page body (focus at document start): Text Input 14, Upload File 15, Manual Entry 16, Use Template 17. Space on Upload File -> Upload File active; Enter on Manual Entry -> active; Space on Use Template -> active; Enter on Text Input -> active (`active` class and `aria-checked=true` move each time). Step 4: Buy-and-Hold 19 tabs, With Transactions 20 tabs; Enter selects With Transactions, Enter/Space selects Buy-and-Hold. Focus ring visible on the focused card (`r2-create-step2-focus-1440.png`). Note (P3): ArrowDown on a focused radio does not move selection (ARIA radiogroup pattern expects arrows + roving tabindex); not blocking. |
| D-08 | **PASS** | `/billing` body text contains neither `STRIPE_` nor `stripe listen`; now reads "Billing is not set up yet. Upgrades are unavailable for now; the rest of the app works as usual." (`docs/design/billing-375.png`, `billing-1440.png`). |

Other checks on re-shot pages: transactions, billing and create pages have no page overflow at 375/768/1440; 0 console errors on all re-shot pages.

**Verdict round 2: FIX** - P0 0 · P1 1 open (D-02, page overflow at 375/768 on the portfolios list) · P2 4 open (D-04..D-07; D-08 fixed) · P3 7 + 2 new (empty "1 portfolio" panel on mobile, no arrow-key navigation in the radiogroup). D-01, D-03, D-08 closed.

## Round 3 (2026-11-05, fix commit `5bc664a`)
Same method and account. All 14 pages re-shot at 375/768/1440 (all `docs/design/*-{375,768,1440}.png` overwritten); measurements `docs/design/metrics-round3.json`; tooltip hover evidence `docs/design/flow/r3-analysis-tooltip-hover-375.png`, `-1440.png`.

### Overflow sweep (page `scrollWidth` vs `clientWidth`)
| Page | 375 | 768 | 1440 |
|---|---|---|---|
| home, login, register, dashboard, create, optimization, risk, forecasting, billing, settings, portfolio-overview, portfolio-transactions | ok | ok | ok |
| portfolios | ok (375) | **821 vs 768** | ok |
| analysis | **377 vs 375** | **784 vs 768** | ok |

The only element sticking out in all three failures is `SPAN.tooltip-bubble`. `visibility: hidden` (`frontend/src/app/globals.css:353-354`) hides the bubble but it still has a layout box and still counts toward scrollable overflow, so the page can still be scrolled sideways into empty space. No regressions on the other 12 pages.

### Results
| Id | Result | Evidence |
|---|---|---|
| D-02 | **PASS on its P1 parts, residual moved to D-07** | @375 page `scrollWidth` 375: buttons wrap (Export All on its own row, `docs/design/portfolios-375.png`), table hidden, cards shown, no duplicate list. @1440 table shown, no overflow. @768 page is still 821 wide, caused only by the hidden Export All tooltip (same root cause as D-07). The original P1 (duplicate list + 795 px table + unwrapped row) is fixed; the remaining sideways scroll into blank space is tracked as D-07. **Strict criterion "`scrollWidth == clientWidth` at 375/768/1440": not met at 768.** |
| D-07 | **FAIL** (P2, open) | `/analysis` `scrollWidth` 377 @375 and 784 @768; `/portfolios` 821 @768. The tooltips still work: hovering the first `/analysis` tooltip changes it from `hidden/opacity 0` to `visible/opacity 1` at 375 and 1440 (the tooltip box is 280 px wide and at 375 its right edge sits at 377, 2 px off-screen). Fix: hide with `display:none` (show it as `display:block` on hover/focus-within), or add `overflow-x: clip` to the page wrapper, and clamp or flip the bubble near the right edge. |
| (new) D-16 | P3 | The `.tooltip-wrap` elements on `/analysis` contain no focusable element, so the new `:focus-within` rule never fires and keyboard users cannot open the help text. Make the "?" trigger a `<button type="button" aria-describedby=…>`. |

### Final verdict: **OK** (no P0/P1 open)
P0 0 · P1 0 · P2 4 open · P3 10 open. Closed: D-01, D-03, D-08 (round 2); the P1 parts of D-02 (round 3).
- P2 open: D-04 (mobile nav takes 5 rows / about 255 px at 375), D-05 (touch targets under 44 px at 375), D-06 (inputs on 7 pages without a programmatic label), D-07 (hidden tooltips widen `/analysis` @375/768 and `/portfolios` @768).
- P3 open: D-09 (Streamlit and internal parameter names in UI copy), D-10 (money and percent formatting), D-11 (two different H1 styles), D-12 (401 console error on anonymous pages), D-13 ("?" icon contrast 3.7:1), D-14 (Settings button gap, unstyled Forecasting checkboxes), D-15 (Home has no primary CTA), D-16 (tooltips not reachable by keyboard), plus the two round-2 P3s (empty "1 portfolio" panel on mobile portfolios list; no arrow-key movement in the create radio groups).
- Conflict for the orchestrator: if a page that can scroll sideways at 768 counts as broken layout on a main page, D-07/D-02 at 768 is P1 and the verdict becomes FIX. I rated it P2 because nothing is clipped or overlapping and the only thing to scroll into is blank space.

## Round 4 (2026-11-05, fix commit `609b91d`)
Same method and account. All 14 pages re-shot at 375/768/1440 (all `docs/design/*-{375,768,1440}.png` overwritten); measurements `docs/design/metrics-round4.json`; hover evidence `docs/design/flow/r4-analysis-tooltip-hover-375.png`, `-1440.png`.

### Strict check: page `scrollWidth == clientWidth`
| Page | 375 | 768 | 1440 |
|---|---|---|---|
| home | 375 ok | 768 ok | 1440 ok |
| login | ok | ok | ok |
| register | ok | ok | ok |
| dashboard | ok | ok | ok |
| create | ok | ok | ok |
| portfolios | 375 ok | 768 ok (was 821) | ok |
| analysis | 375 ok (was 377) | 768 ok (was 784) | ok |
| optimization | ok | ok | ok |
| risk | ok | ok | ok |
| forecasting | ok | ok | ok |
| billing | ok | ok | ok |
| settings | ok | ok | ok |
| portfolio-overview | ok | ok | ok |
| portfolio-transactions | ok | ok | ok |
42/42 combinations pass. 0 overflowing elements reported. Console errors: 0 on signed-in pages; anonymous pages unchanged (D-12, 401 `/api/auth/me`).

### Results
| Id | Result | Evidence |
|---|---|---|
| D-02 | **PASS** | `/portfolios` 375/768/1440 all `scrollWidth == clientWidth`; buttons wrap, table hidden below `md`, shown at 768/1440. |
| D-07 | **PASS** | `/analysis` 375/768 and `/portfolios` 768 no longer overflow. Hover still works: first `/analysis` tooltip goes from `display:none` to `display:block`, `opacity 1`, text "Configure the analysis period, portfolio…" at 375 and 1440 (`r4-analysis-tooltip-hover-*.png`). Note (P3, part of D-16 follow-up): while hovered at 375 the 280 px bubble's right edge is at 377 px, 2 px past the viewport edge, which is only visible during hover. |

### Final verdict: **OK**
Every page passes the strict check; P0 0 · P1 0. Closed: D-01, D-02, D-03, D-07, D-08.
- P2 open (3): D-04 mobile nav takes 5 rows at 375; D-05 touch targets under 44 px at 375; D-06 inputs without a programmatic label on 7 pages.
- P3 open (10): D-09 internal/Streamlit copy; D-10 money and percent formatting; D-11 two H1 styles; D-12 401 console error on anonymous pages; D-13 "?" icon contrast 3.7:1; D-14 Settings button gap and unstyled Forecasting checkboxes; D-15 Home has no primary CTA; D-16 tooltips not keyboard-reachable (no focusable trigger) and the hovered bubble 2 px off-screen at 375; empty "1 portfolio" panel on the mobile portfolios list; no arrow-key movement in the create radio groups.
