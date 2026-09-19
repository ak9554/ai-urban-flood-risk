# FloodSense AI — NCR Urban Flood Risk

FloodSense AI is an explainable flood-risk planning prototype for exploring location-level flood concern across seven NCR districts.

It helps users:

- Explore a flood-concern map
- Understand why an area is highlighted
- Identify locations requiring attention first
- Simulate rainfall what-if scenarios
- Compare districts
- Review Bihar district-level validation evidence
- Understand the model’s methodology and limitations

## Live Demo

[Open the FloodSense AI live app](https://floodsense-ai-ncr.streamlit.app/)

Replace `https://floodsense-ai-ncr.streamlit.app/` with the deployed Streamlit URL.

## Study Area

The NCR reference dataset covers:

- Faridabad
- Gautam Buddha Nagar
- Ghaziabad
- Gurugram
- New Delhi
- Panipat
- Sonipat

The reference dataset contains 8,471 usable map locations across seven districts.

## Main Application Flow

The application is organized around a practical decision flow:

1. Command Center
2. Flood Map
3. Why This Area?
4. What Should We Check First?
5. Response Planner
6. Rainfall What-If
7. Compare Districts
8. Bihar Evidence
9. Methodology & Model Insights

## How the Risk View Is Built

The NCR reference risk view combines:

- Rainfall conditions over multiple time windows
- Ground and surface susceptibility
- Historical flood context
- People and property exposure
- Location and district information
- A transparent explanation of the main contributing signal

The result is intended for relative prioritization and planning support. It is not a live warning system or an official forecast.

## Model and Data Notes

The project includes spatial and temporal flood-risk outputs generated during the research workflow.

The repository contains:

- Processed location-level risk data
- District boundaries
- Historical flood context
- Flood inventory records
- Model artifacts
- Reproducible research scripts
- The Streamlit demonstration application

The application reads the processed project outputs and presents them using user-friendly labels. Technical field mappings and model details are available only in the Methodology & Model Insights page.

## Bihar Evidence Validation

The Bihar Evidence page provides a district-level transfer-validation exercise.

It uses:

- Historical Bihar flood records from 2015–2023
- Static district flood-impact context
- All 38 Bihar districts
- Official 2025 NRSC/ISRO observed affected-area data
- Comparisons against simpler ranking baselines

The 2025 benchmark covers 16 affected districts and approximately 134,966 hectares.

This evidence supports district-level ranking transferability only. It does not prove that the NCR neighbourhood-level map is accurate for every location in Bihar. A true Bihar fine-scale model would require Bihar-wide terrain, land-cover, rainfall, and inundation datasets.

## Scientific Caveats

This project is a planning and demonstration prototype.

Important limitations:

- The base NCR view is based on processed historical project data.
- It does not use live sensor feeds.
- Rainfall What-If results are simulated scenarios, not real forecasts.
- Model scores indicate relative concern, not guaranteed flooding.
- Local conditions and official government warnings remain authoritative.
- The Bihar validation is district-level, not neighbourhood-level.
- Field verification is required before operational decisions.

## Repository Structure

```text
ai-urban-flood-risk/
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   │   ├── ncr_hyperlocal_risk.csv
│   │   ├── ncr_7_districts.geojson
│   │   └── ncr_historical_context.csv
│   └── raw/
│       ├── District_FloodedArea.csv
│       ├── District_FloodImpact.csv
│       └── India_Flood_Inventory_v3.csv
├── models/
│   ├── ncr_flood_risk_random_forest.joblib
│   └── ncr_hgb_v4_temporal_model.joblib
├── src/
├── requirements.txt
├── README.md
└── .gitignore
```

## Local Setup

Clone the repository:

```powershell
git clone https://github.com/ak9554/ai-urban-flood-risk.git
cd ai-urban-flood-risk
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check the application:

```powershell
python -m py_compile app\app.py
```

Run the application:

```powershell
python -m streamlit run app\app.py --server.port 8511
```

Open:

```text
http://localhost:8511
```

Run the application from the repository root so that data paths resolve consistently.

## Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment settings:

```text
Repository: ak9554/ai-urban-flood-risk
Branch: master
Main file path: app/app.py
```

Streamlit installs dependencies from the root `requirements.txt`.

## Official Bihar Evidence Sources

- [NRSC/ISRO Bihar flood and inundation report — 29 August 2025](https://ndem.nrsc.gov.in/documents/Disaster_Document/2025/BR/brflood50dsc29082025_0600hrs/brflood50dsc29082025_0600hrs_report.pdf)
- [Press Information Bureau Bihar flood assessment — 16 September 2026](https://www.pib.gov.in/PressReleseDetailm.aspx?PRID=2311105&lang=2&reg=48)

## Project Status

Version: `v1.0.0`

Status: Demo-ready research prototype

The project is suitable for:

- Hackathon demonstrations
- Research presentations
- Flood-risk exploration
- Model explainability demonstrations
- Scenario-based planning discussions