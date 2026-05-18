from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src/app/portfolios/page.tsx"
text = path.read_text(encoding="utf-8")
start_marker = (
    '          {tab === "strategies" && (\n            <div className="space-y-3">'
)
replacement = """          {tab === "strategies" && selected && (
            <RebalanceStrategyPanel
              portfolio={selected}
              intervalValue={strategyInterval}
              onIntervalChange={setStrategyInterval}
              onSave={() => void saveRebalanceStrategy()}
              saving={savingStrategy}
              showSaveButton
            />
          )"""

count = 0
while start_marker in text:
    start = text.index(start_marker)
    end = text.index("\n          )}", start) + len("\n          )}")
    text = text[:start] + replacement + text[end:]
    count += 1

path.write_text(text, encoding="utf-8")
print(f"replaced {count} strategies blocks")
