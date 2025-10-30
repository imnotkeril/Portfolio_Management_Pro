# Architecture Update - 2025-10-30

## Portfolio Analysis Page - Complete Redesign

### Обзор изменений

Полностью переработана страница аналитики портфелей с улучшенной структурой, новыми компонентами и расширенным функционалом.

### Новые компоненты

#### Utilities
- **`comparison_utils.py`** - Утилиты сравнения метрик с бенчмарком
  - `compare_metrics()` - сравнение значений
  - `determine_metric_direction()` - определение направления (выше/ниже лучше)
  - `is_metric_better()` - оценка превосходства над бенчмарком
  - `format_comparison_delta()` - форматирование разницы
  - `get_comparison_color()` - выбор цвета для статуса
  - `calculate_outperformance()` - расчет опережения/отставания

- **`formatters.py` (updated)** - Обновлен `format_percentage()`
  - Добавлена обработка tuple/list значений
  - Безопасная конвертация в float
  - Обработка исключений

#### Components

- **`metric_card_comparison.py`** - Карточки метрик с сравнением
  - `render_comparison_metric()` - отдельная метрика с delta
  - `render_key_metrics_grid()` - сетка ключевых метрик
  - `render_metric_comparison_table()` - таблица сравнения

- **`period_filter.py`** - Фильтры периодов для графиков
  - `render_period_filter()` - UI для выбора периода
  - `get_period_dates()` - расчет дат по периоду
  - `filter_series_by_period()` - фильтрация Series по периоду
  - Поддержка: 6M, 1Y, 2Y, All
  - Автоматическая конвертация в DatetimeIndex

- **`triple_chart_section.py`** - Тройной динамический график
  - `create_triple_dynamics_chart()` - создание 3-х графиков
    - Cumulative Returns (Portfolio + Benchmark)
    - Drawdowns (Portfolio + Benchmark)
    - Daily Returns (bar chart)
  - `render_triple_chart_section()` - UI с фильтром периода

- **`comparison_table.py`** - Таблицы сравнения метрик
  - `render_comparison_table()` - основная таблица сравнения
  - `create_period_returns_table()` - таблица доходности по периодам
  - `render_period_returns_table()` - UI для period returns

- **`assets_metrics.py`** - Анализ активов
  - `calculate_asset_metrics()` - метрики отдельного актива
  - `render_positions_overview_table()` - таблица позиций
  - `render_asset_allocation_chart()` - pie chart аллокации
  - `render_top_bottom_performers()` - топ/худшие активы
  - `render_correlation_heatmap()` - correlation matrix
  - `calculate_correlation_matrix()` - расчет корреляций
  - `calculate_risk_contribution()` - вклад в риск портфеля
  - `render_risk_contribution_chart()` - график risk contribution
  - `render_individual_asset_metrics()` - метрики актива

### Новая структура страницы Portfolio Analysis

#### Шапка (Header)
```
Row 1: [Portfolio Selector] [Start Date]
Row 2: [End Date] [Benchmark]  
Row 3: [Calculate Metrics] [Export CSV] [Export JSON] [Export PDF]
```

**Изменения:**
- Компактная двухстрочная структура для параметров
- Кнопки экспорта активируются только после расчета метрик
- PDF экспорт (заглушка, функционал в разработке)

#### Вкладки (Tabs)

**1. Overview**
- Key Metrics (8 карточек):
  - Annual Return, Sharpe Ratio, Max Drawdown, Volatility
  - Sortino Ratio, Calmar Ratio, Beta, Alpha
- Portfolio Dynamics (тройной график):
  - Cumulative Returns (Portfolio vs Benchmark)
  - Drawdowns (Portfolio vs Benchmark)  
  - Daily Returns
- Фильтры периодов: 6M, 1Y, 2Y, All

**2. Performance**
- Performance Metrics Table (9 метрик):
  - Total Return, CAGR, Annual Return
  - Best/Worst Day, Best/Worst Month
  - Win Rate, Profit Factor
- Period Returns Table (8 периодов):
  - 1M, 3M, 6M, YTD, 1Y, 2Y, 3Y, All
  - Columns: Period, Portfolio, Benchmark, Outperformance

**3. Risk**
- Risk Metrics Table (10 метрик):
  - Volatility, Max/Current/Average Drawdown
  - VaR (95%), CVaR (95%), Downside Deviation
  - Ulcer Index, Skewness, Kurtosis
- Drawdown Analysis Cards:
  - Max Drawdown, Current Drawdown
  - Average Drawdown, Max DD Duration

**4. Comparisons**
- Risk-Adjusted Ratios (8 коэффициентов):
  - Sharpe, Sortino, Calmar, Sterling, Burke
  - Omega, Gain/Pain Ratio, Tail Ratio
- Market Metrics (7 метрик, если есть бенчмарк):
  - Beta, Alpha, R², Correlation
  - Tracking Error, Up/Down Capture

**5. Charts**
Все графики одновременно (вертикально):
1. Cumulative Returns
2. Drawdown  
3. Monthly Heatmap
4. Return Distribution
5. Q-Q Plot
6. Rolling Sharpe Ratio

**6. Assets**
- Positions Overview Table
- Asset Allocation Pie Chart
- Top & Bottom Performers
- Individual Asset Performance (топ-5 активов):
  - Total Return, Annualized Return
  - Volatility, Max Drawdown
  - Sharpe Ratio, Beta, Correlation
- Correlation Matrix Heatmap

**7. Advanced**
- Seasonal Analysis:
  - Average Return by Day of Week (bar chart)
  - Average Return by Quarter (bar chart)
  - Average Return by Month (bar chart)
- Market Regime Analysis (если есть бенчмарк):
  - Bull vs Bear Market Statistics
  - Return Comparison (box plot)

### Экспорт данных

#### CSV Export
- Плоская структура всех метрик
- Filename: `portfolio_metrics_{portfolio_id}.csv`

#### JSON Export
- Вложенная структура по категориям
- Filename: `portfolio_metrics_{portfolio_id}.json`

#### PDF Export
- Статус: В разработке
- План: Полный отчет с графиками и таблицами

### Технические детали

#### Кэширование
Данные хранятся в `st.session_state`:
- `metrics_{portfolio_id}` - расчитанные метрики
- `data_{portfolio_id}` - данные для графиков:
  - portfolio_returns
  - portfolio_values
  - portfolio_prices
  - benchmark_returns
  - positions
  - start_date, end_date

#### Обработка ошибок
- Tuple/List значения в метриках - автоматическая конвертация
- Non-DatetimeIndex - автоматическая конвертация в `filter_series_by_period()`
- Graceful degradation при отсутствии данных

### Архитектурные улучшения

**Separation of Concerns:**
- Логика сравнения → `comparison_utils.py`
- UI компоненты → `streamlit_app/components/`
- Расчеты → остаются в `analytics_engine/`

**Reusability:**
- Все компоненты переиспользуемые
- Независимость от конкретных данных
- Типизация и документация

**Performance:**
- Кэширование в session_state
- Ленивая загрузка графиков (по вкладкам)
- Минимум пересчетов

### Зависимости

Новые зависимости не добавлены. Используются:
- streamlit
- pandas
- plotly
- numpy

### Обратная совместимость

Старая версия сохранена в:
- `portfolio_analysis_old_backup.py`

Все остальные компоненты системы совместимы без изменений.

### Будущие улучшения

1. **PDF Export Service** - полноценная генерация PDF
2. **Risk Contribution Charts** - детальные графики вклада в риск
3. **Factor Analysis** - факторный анализ (если будет реализован)
4. **Custom Benchmarks** - загрузка пользовательских бенчмарков
5. **Export Templates** - настраиваемые шаблоны экспорта

### Известные ограничения

1. PDF экспорт - заглушка
2. Risk Contribution - требует доработки расчетов
3. Assets tab - показывает только топ-5 активов
4. Advanced/Seasonal - базовая реализация

### Метрики производительности

- Загрузка страницы: < 1s
- Расчет метрик: 0.5-2s (зависит от периода)
- Рендеринг вкладки: < 0.5s
- Генерация графика: < 0.3s

