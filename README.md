# BSES Transformer & Distribution AI Agent Demo

A Streamlit demo that simulates transformer and electricity distribution telemetry for a BSES-style distribution network. It shows how an AI agent can use telemetry to detect anomalies, predict maintenance windows, and provide a GenBI interface for efficiency and diagnostics questions.

## What the demo includes

- Distribution network view for grid infeed → substation → transformer → feeder → LT circuits
- Synthetic telemetry for transformer loading, oil temperature, winding temperature, DGA proxy, voltage deviation, phase imbalance, THD, power factor, feeder loss, breaker trip probability and smart meter telemetry health
- AI anomaly agent with root-cause findings and recommended field actions
- Predictive maintenance scoring with remaining useful life and work-order preview
- GenBI query layer for questions such as:
  - current risk
  - why is anomaly high?
  - show last 80 ticks oil trend
  - feeder loss efficiency
  - recommended diagnostics
  - asset scope

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload all files from this folder to the repository root.
3. Go to Streamlit Community Cloud.
4. Select the GitHub repo.
5. Set the main file path to `app.py`.
6. Deploy.

## Notes

This is a synthetic offline demo. It is designed for stakeholder demonstration and can later be connected to real SCADA, AMI, GIS, OMS, ERP/EAM, CMMS or data lake sources.
