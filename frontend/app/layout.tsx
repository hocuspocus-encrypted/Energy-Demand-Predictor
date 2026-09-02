import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Energy Demand Predictor",
  description: "Hourly grid demand forecasts from a RandomForest model trained on real PJM data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen text-slate-100 antialiased">{children}</body>
    </html>
  );
}
