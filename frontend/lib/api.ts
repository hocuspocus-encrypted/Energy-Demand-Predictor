export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export interface PredictionFeatures {
  temp_c: number;
  humidity: number;
  wind_speed: number;
  hour: number;
  dayofweek: number;
  month: number;
  is_weekend: number;
  hour_sin: number;
  hour_cos: number;
  doy_sin: number;
  doy_cos: number;
  energy_lag_1h: number;
  energy_lag_2h: number;
  energy_lag_3h: number;
  energy_lag_24h: number;
  energy_lag_168h: number;
  energy_roll_mean_24h: number;
  energy_roll_std_24h: number;
  temp_roll_mean_24h: number;
  energy_roll_mean_168h: number;
  energy_roll_std_168h: number;
  temp_roll_mean_168h: number;
}

export interface SampleResponse {
  datetime: string;
  actual_energy_mw: number;
  features: PredictionFeatures;
}

export interface PredictionResponse {
  predicted_energy_mw: number;
  model_version: string;
  run_id: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${path} failed (${res.status}): ${body}`);
  }
  return res.json() as Promise<T>;
}

export const fetchSample = () => request<SampleResponse>("/sample");

export const predict = (features: PredictionFeatures) =>
  request<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(features),
  });

export const checkHealth = () => request<{ status: string; model_version: string }>("/health");

/** Calendar features (hour/dayofweek/.../doy_cos) derived from a JS Date,
 * matching src/features.py's add_calendar_features exactly. */
export function deriveCalendarFeatures(date: Date) {
  const hour = date.getHours();
  const dayofweek = (date.getDay() + 6) % 7; // JS: Sun=0..Sat=6 -> Python: Mon=0..Sun=6
  const month = date.getMonth() + 1;
  const is_weekend = dayofweek >= 5 ? 1 : 0;

  const startOfYear = new Date(date.getFullYear(), 0, 1);
  const dayofyear = Math.floor((date.getTime() - startOfYear.getTime()) / 86400000) + 1;

  return {
    hour,
    dayofweek,
    month,
    is_weekend,
    hour_sin: Math.sin((2 * Math.PI * hour) / 24),
    hour_cos: Math.cos((2 * Math.PI * hour) / 24),
    doy_sin: Math.sin((2 * Math.PI * dayofyear) / 365.25),
    doy_cos: Math.cos((2 * Math.PI * dayofyear) / 365.25),
  };
}
