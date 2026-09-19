\# NCR Urban Flood Risk Prediction



An explainable machine-learning prototype for district-level urban flood-risk prediction across seven NCR districts.



\## Study Area



\- Faridabad

\- Gautam Buddha Nagar

\- Ghaziabad

\- Gurugram

\- New Delhi

\- Panipat

\- Sonipat



\## Model



The project uses a Random Forest classifier with a temporal evaluation split:



\- Training period: 2015–2021

\- Test period: 2022–2023

\- Total observations: 23,009

\- Training observations: 17,899

\- Test observations: 5,110



\## Model Features



\- 24-hour rainfall

\- 3-day rainfall

\- 7-day rainfall

\- Historical flooded-area percentage

\- Population

\- Historical mean flood duration

\- District identity

\- Seasonal date features

\- Rainfall ratio features



`historical\_flood\_days` is excluded because it was calculated from the same flood labels used as the prediction target and would cause target leakage.



\## Test Results



\- ROC-AUC: 0.6465

\- PR-AUC: 0.0128

\- Flood events in test period: 15

\- Detected flood events: 2



This is an early prototype. The model is more useful for ranking relative risk than for issuing confirmed flood warnings.



\## Generated Outputs



\- `data/processed/ncr\_flood\_risk\_map.html`

\- `data/processed/ncr\_flood\_risk\_report.html`

\- `data/processed/ncr\_high\_risk\_alerts.csv`

\- `data/processed/ncr\_risk\_predictions\_2022\_2023.csv`

\- `models/ncr\_flood\_risk\_random\_forest.joblib`



\## How to Run



Activate the virtual environment:



```powershell

.\\.venv\\Scripts\\Activate.ps1

