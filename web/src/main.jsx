import React from "react";
import ReactDOM from "react-dom/client";
// Self-hosted variable fonts. Inter was named in the Tailwind config but never
// loaded, so everything silently fell back to the system UI face.
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import App from "./App.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
