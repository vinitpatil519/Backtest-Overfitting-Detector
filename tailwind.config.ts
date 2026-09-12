import type { Config } from "tailwindcss";

/**
 * Colours are declared as `rgb(var(--token) / <alpha-value>)` so that opacity
 * modifiers work (`text-ink/80`, `border-beam/60`). The variables themselves are
 * channel triplets defined in `app/globals.css`.
 */
const token = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: token("canvas"),
        panel: token("panel"),
        panel2: token("panel-2"),
        line: token("line"),
        ink: token("ink"),
        muted: token("muted"),
        pass: token("pass"),
        warn: token("warn"),
        fail: token("fail"),
        beam: token("beam"),
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
      },
      boxShadow: {
        panel:
          "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 12px 40px -20px rgba(0,0,0,0.9)",
      },
    },
  },
  plugins: [],
};

export default config;
