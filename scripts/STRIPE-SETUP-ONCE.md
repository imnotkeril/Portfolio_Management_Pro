# Stripe — один раз (5 минут)

## 1. В `.env` уже должно быть

- `STRIPE_SECRET_KEY` — Secret key из Stripe (Developers → API keys)
- `STRIPE_PRICE_ID_PRO` — можно `prod_...` **или** `price_...` (код сам найдёт price)
- `STRIPE_WEBHOOK_SECRET` — **не обязателен** для локалки (после оплаты открой `/billing` — план подтянется с Stripe)

## 2. Запуск

### Docker (`docker compose up`)

Stripe-переменные из `.env` на хосте подставляются в контейнер `api` через `docker compose` (`${STRIPE_SECRET_KEY}` и т.д.). После изменения кода или `.env`:

```powershell
cd E:\Main\Dev\Python\Done\Portfolio_Management_Pro
docker compose build api web
docker compose up -d
```

Если на Billing видите **Error: Not Found** — образ API старый (нет `/billing/*`). Пересоберите `api` как выше.

### Без Docker (uvicorn + npm)

Терминал 1 — API:

```powershell
cd E:\Main\Dev\Python\Done\Portfolio_Management_Pro
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Терминал 2 — сайт:

```powershell
cd E:\Main\Dev\Python\Done\Portfolio_Management_Pro\frontend
npm run dev
```

## 3. Проверка

1. http://localhost:3000 → войти
2. **Billing** → **Upgrade to Pro**
3. Карта `4242 4242 4242 4242`
4. Снова **Billing** — должно быть **Pro**

Webhook (опционально): `.\scripts\start-stripe-listen.ps1`
