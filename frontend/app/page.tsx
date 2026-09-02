"use client";

import { useEffect, useMemo, useState } from "react";
import {
  API_BASE,
  PredictionFeatures,
  PredictionResponse,
  checkHealth,
  deriveCalendarFeatures,
  fetchSample,
  predict,
} from "@/lib/api";

type Health = "checking" | "online" | "offline";

const LAG_FIELDS: { key: keyof PredictionFeatures; label: string }[] = [
  { key: "energy_lag_1h", label: "1h ago" },
  { key: "energy_lag_2h", label: "2h ago" },
  { key: "energy_lag_3h", label: "3h ago" },
  { key: "energy_lag_24h", label: "Same hour, yesterday" },
  { key: "energy_lag_168h", label: "Same hour, last week" },
];

const ROLLING_FIELDS: { key: keyof PredictionFeatures; label: string; unit: string }[] = [
  { key: "energy_roll_mean_24h", label: "24h avg demand", unit: "MW" },
  { key: "energy_roll_std_24h", label: "24h demand volatility", unit: "MW" },
  { key: "energy_roll_mean_168h", label: "7d avg demand", unit: "MW" },
  { key: "energy_roll_std_168h", label: "7d demand volatility", unit: "MW" },
  { key: "temp_roll_mean_24h", label: "24h avg temp", unit: "°C" },
  { key: "temp_roll_mean_168h", label: "7d avg temp", unit: "°C" },
];

function NumberField({
  label,
  value,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  unit?: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-slate-400">{label}</span>
      <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 focus-within:border-accent/60 transition-colors">
        <input
          type="number"
          value={Number.isFinite(value) ? value : ""}
          step="any"
          onChange={(e) => onChange(e.target.valueAsNumber)}
          className="w-full bg-transparent text-sm text-slate-100 outline-none [appearance:textfield]"
        />
        {unit && <span className="text-xs text-slate-500">{unit}</span>}
      </div>
    </label>
  );
}

export default function Home() {
  const [health, setHealth] = useState<Health>("checking");
  const [modelVersion, setModelVersion] = useState<string | null>(null);

  const [datetimeLocal, setDatetimeLocal] = useState("");
  const [features, setFeatures] = useState<PredictionFeatures | null>(null);
  const [originalFeatures, setOriginalFeatures] = useState<PredictionFeatures | null>(null);
  const [actual, setActual] = useState<{ datetime: string; value: number } | null>(null);

  const [loadingSample, setLoadingSample] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkHealth()
      .then((h) => {
        setHealth("online");
        setModelVersion(h.model_version);
      })
      .catch(() => setHealth("offline"));
  }, []);

  const loadSample = async () => {
    setLoadingSample(true);
    setError(null);
    setResult(null);
    try {
      const sample = await fetchSample();
      setFeatures(sample.features);
      setOriginalFeatures(sample.features);
      setActual({ datetime: sample.datetime, value: sample.actual_energy_mw });
      setDatetimeLocal(sample.datetime.slice(0, 16));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load a sample.");
    } finally {
      setLoadingSample(false);
    }
  };

  useEffect(() => {
    loadSample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const updateField = (key: keyof PredictionFeatures) => (value: number) => {
    setFeatures((f) => (f ? { ...f, [key]: value } : f));
  };

  const handleDatetimeChange = (value: string) => {
    setDatetimeLocal(value);
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return;
    const calendar = deriveCalendarFeatures(date);
    setFeatures((f) => (f ? { ...f, ...calendar } : f));
  };

  const runPredict = async () => {
    if (!features) return;
    setPredicting(true);
    setError(null);
    try {
      const res = await predict(features);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed.");
    } finally {
      setPredicting(false);
    }
  };

  // "Actual demand" is only a real observation if the inputs still match
  // exactly what /sample returned -- edit anything (the date included) and
  // it becomes a hypothetical scenario with no ground truth to compare to.
  const matchesSample = useMemo(() => {
    if (!features || !originalFeatures) return false;
    return (Object.keys(originalFeatures) as (keyof PredictionFeatures)[]).every(
      (key) => features[key] === originalFeatures[key]
    );
  }, [features, originalFeatures]);

  const delta = useMemo(() => {
    if (!result || !actual || !matchesSample) return null;
    const diff = result.predicted_energy_mw - actual.value;
    const pct = (diff / actual.value) * 100;
    return { diff, pct };
  }, [result, actual, matchesSample]);

  return (
    <main className="mx-auto max-w-3xl px-6 py-14">
      <header className="mb-10 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Energy Demand Predictor
          </h1>
          <p className="mt-1.5 text-sm text-slate-400">
            RandomForest model trained on real PJM East hourly grid demand, tracked with
            MLflow and served over FastAPI.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              health === "online"
                ? "bg-emerald-400"
                : health === "offline"
                ? "bg-rose-400"
                : "bg-amber-400"
            }`}
          />
          {health === "online" ? `API online · v${modelVersion}` : health === "offline" ? "API offline" : "Checking API…"}
        </div>
      </header>

      {health === "offline" && (
        <div className="mb-6 rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          Can&apos;t reach the API at <code className="text-rose-100">{API_BASE}</code>. Set{" "}
          <code className="text-rose-100">NEXT_PUBLIC_API_URL</code> to your backend&apos;s URL.
        </div>
      )}

      <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 backdrop-blur">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-sm font-medium text-slate-300">Inputs</h2>
          <button
            onClick={loadSample}
            disabled={loadingSample}
            className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:bg-white/10 disabled:opacity-50"
          >
            {loadingSample ? "Loading…" : "Load real example"}
          </button>
        </div>

        {!features ? (
          <p className="text-sm text-slate-500">Loading a real historical snapshot…</p>
        ) : (
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <label className="flex flex-col gap-1.5">
                <span className="text-xs font-medium text-slate-400">Date &amp; hour</span>
                <input
                  type="datetime-local"
                  value={datetimeLocal}
                  onChange={(e) => handleDatetimeChange(e.target.value)}
                  className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 outline-none focus:border-accent/60"
                />
              </label>
              <NumberField
                label="Temperature"
                value={features.temp_c}
                unit="°C"
                onChange={updateField("temp_c")}
              />
              <NumberField
                label="Humidity"
                value={features.humidity}
                unit="%"
                onChange={updateField("humidity")}
              />
              <NumberField
                label="Wind speed"
                value={features.wind_speed}
                unit="km/h"
                onChange={updateField("wind_speed")}
              />
            </div>

            <details className="group rounded-lg border border-white/10 bg-black/20 open:pb-4">
              <summary className="cursor-pointer list-none px-4 py-3 text-xs font-medium text-slate-400 hover:text-slate-200">
                Recent demand context (from history) – click to expand
              </summary>
              <div className="grid grid-cols-1 gap-4 px-4 sm:grid-cols-2 md:grid-cols-3">
                {LAG_FIELDS.map(({ key, label }) => (
                  <NumberField
                    key={key}
                    label={label}
                    unit="MW"
                    value={features[key]}
                    onChange={updateField(key)}
                  />
                ))}
                {ROLLING_FIELDS.map(({ key, label, unit }) => (
                  <NumberField
                    key={key}
                    label={label}
                    unit={unit}
                    value={features[key]}
                    onChange={updateField(key)}
                  />
                ))}
              </div>
            </details>

            <button
              onClick={runPredict}
              disabled={predicting}
              className="w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-slate-950 transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {predicting ? "Predicting…" : "Predict demand"}
            </button>
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        )}
      </section>

      {result && (
        <section className="mt-6 rounded-2xl border border-accent/20 bg-accent/5 p-6">
          <p className="text-xs font-medium text-slate-400">Predicted demand</p>
          <p className="mt-1 text-4xl font-semibold tracking-tight text-white">
            {result.predicted_energy_mw.toLocaleString(undefined, { maximumFractionDigits: 0 })}{" "}
            <span className="text-lg font-normal text-slate-400">MW</span>
          </p>

          {actual && delta ? (
            <p className="mt-3 text-sm text-slate-300">
              Actual demand at this timestamp was{" "}
              <span className="font-medium text-slate-100">
                {actual.value.toLocaleString(undefined, { maximumFractionDigits: 0 })} MW
              </span>{" "}
              &mdash; off by{" "}
              <span className={delta.pct >= 0 ? "text-amber-300" : "text-sky-300"}>
                {delta.pct >= 0 ? "+" : ""}
                {delta.pct.toFixed(1)}%
              </span>
              .
            </p>
          ) : (
            <p className="mt-3 text-sm text-slate-500">
              Hypothetical scenario &mdash; inputs have been edited, so there&apos;s no real
              observation to compare against.
            </p>
          )}

          <p className="mt-4 text-xs text-slate-500">
            Model v{result.model_version} &middot; run {result.run_id.slice(0, 8)}
          </p>
        </section>
      )}

      <footer className="mt-14 text-center text-xs text-slate-600">
        Trained on PJM East hourly grid demand (2002–2018) &middot; tracked with MLflow &middot;
        served with FastAPI
      </footer>
    </main>
  );
}
