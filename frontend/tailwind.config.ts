import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  darkMode: "media",
  theme: {
    extend: {
      colors: {
        ink: "#0b1220",
        surface: "#0f1729",
        accent: "#22d3ee",
      },
    },
  },
  plugins: [],
};

export default config;
