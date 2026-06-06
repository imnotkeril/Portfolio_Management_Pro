# Forward Stripe webhooks to local API (optional if billing sync-on-refresh is enough).
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
Set-Location $PSScriptRoot\..
Write-Host "Login once if asked, then keep this window open."
stripe listen --forward-to localhost:8000/stripe/webhook
