# BSES Transformer & Distribution AI Agent Demo

A dependency-free, static web dashboard that simulates transformer and electricity-distribution telemetry for a BSES-style network. The browser generates synthetic telemetry, detects anomalies, estimates maintenance urgency, and provides a rule-based GenBI query interface—all without a server or build step.

## Features

- Interactive network view from grid infeed to LT consumer circuits
- Live, synthetic telemetry with autoplay and manual advancement
- Selectable transformers, feeders, and operating scenarios
- Adjustable sensor noise, asset degradation, telemetry dropout, and alert thresholds
- AI-style anomaly, maintenance, efficiency, and remaining-useful-life scoring
- Root-cause findings, field recommendations, and work-order preview
- Canvas-based telemetry and risk charts with no third-party JavaScript dependencies
- Plain-English GenBI queries and an export-ready KPI workspace
- Responsive layout for desktop, tablet, and mobile

## Run locally

No Python environment or package installation is required. Serve the repository with any static file server:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

> Opening `index.html` directly also works in most browsers, but a local web server better matches GitHub Pages behavior.

## Deploy with GitHub Pages

1. Merge this branch into the branch you want to publish (normally `main`).
2. Open the repository on GitHub and go to **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Select the publishing branch (normally `main`) and the **`/ (root)`** folder.
5. Click **Save**. GitHub will publish `index.html` as the site entry point.

The `.nojekyll` marker tells GitHub Pages to serve the repository as a plain static site. No Streamlit secrets, Python runtime, actions workflow, or external API are required.

## Project structure

- `index.html` — semantic dashboard markup and controls
- `styles.css` — responsive visual system and dashboard layout
- `app.js` — telemetry simulation, AI scoring, charts, interactivity, and GenBI logic
- `.nojekyll` — disables Jekyll processing on GitHub Pages

## Data note

All telemetry and diagnostics are synthetic and generated locally in the browser. This stakeholder demo can later be connected to real SCADA, AMI, GIS, OMS, ERP/EAM, CMMS, or data-lake services through APIs.
