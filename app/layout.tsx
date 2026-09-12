import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Backtest Overfitting Detector",
  description:
    "Deflated Sharpe Ratio, Probability of Backtest Overfitting and Monte Carlo validation — decide whether an impressive Sharpe ratio is real or statistical luck.",
  keywords: [
    "deflated sharpe ratio",
    "probability of backtest overfitting",
    "CSCV",
    "quantitative finance",
    "Monte Carlo",
  ],
};

export const viewport: Viewport = {
  themeColor: "#07080a",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
