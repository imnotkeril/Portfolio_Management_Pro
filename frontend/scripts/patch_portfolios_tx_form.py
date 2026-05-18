from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src/app/portfolios/page.tsx"
text = path.read_text(encoding="utf-8")
marker_start = "              {/* Add transaction form */}"
marker_end = "              {/* Transaction history */}"
if marker_start not in text:
    raise SystemExit("start marker not found")
start = text.index(marker_start)
end = text.index(marker_end, start)
replacement = """              <Expander
                title="Add Transaction"
                defaultOpen={transactions.length === 0}
              >
                <TransactionForm onSubmit={handleAddTransaction} />
              </Expander>

"""
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
print("patched portfolios tx form")
