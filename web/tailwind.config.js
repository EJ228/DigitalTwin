/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Design system from Claude Design: light, high-contrast, control-room.
        page: "#F0F1F3",
        card: "#FFFFFF",
        line: "#E5E7EB",
        ink: { DEFAULT: "#1A1A1F", muted: "#6B7280", faint: "#9CA3AF" },
        // Semantic state colours. Components reference these names, never hex,
        // so a palette change is a one-file edit.
        state: {
          run: "#2E9E5B",
          blocked: "#C2703A",
          starved: "#3B82F6",
          down: "#DC2626",
          drift: "#D2551E",
          bottleneck: "#8B2FE8",
          blind: "#9CA3AF",
        },
        accent: "#8B2FE8",
        alert: "#D2551E",
      },
      fontFamily: {
        sans: ["'Inter Variable'", "Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["'JetBrains Mono Variable'", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      letterSpacing: { label: "0.12em" },
      // Two elevations only. Cards sit flat; raised is for the andon, which is
      // the one thing on screen allowed to interrupt.
      boxShadow: {
        card: "0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.05)",
        raised: "0 1px 2px rgba(16,24,40,0.05), 0 8px 24px -6px rgba(16,24,40,0.12)",
      },
    },
  },
  plugins: [],
};
