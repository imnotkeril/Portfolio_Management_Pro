import { api } from "@/lib/api";

export type TickerPriceResponse = {
  ticker: string;
  valid: boolean;
  price: number | null;
  price_date?: string | null;
  requested_date?: string | null;
};

export async function fetchTickerPrice(
  ticker: string,
  onDate?: string,
): Promise<TickerPriceResponse> {
  const t = ticker.trim().toUpperCase();
  const qs = onDate ? `?on_date=${encodeURIComponent(onDate)}` : "";
  return api.get<TickerPriceResponse>(
    `/ticker-price/${encodeURIComponent(t)}${qs}`,
  );
}
