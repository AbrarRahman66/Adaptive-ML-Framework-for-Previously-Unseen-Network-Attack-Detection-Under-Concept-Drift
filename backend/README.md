# IDS backend (connected to the `nids` MariaDB database)

## Folder layout
backend/
  main.py  nids_migration.sql  requirements.txt  test_sample.py
  IDS_Project/            <- saved models from Google Drive (without data/)

## 1. Database (XAMPP)
1. Start Apache + MySQL in the XAMPP control panel.
2. phpMyAdmin -> import `nids.sql` (Rayan's file) if the `nids` database is not already there.
3. XAMPP default login is user `root` with an EMPTY password.

## 2. Run
Python 3.13, fresh venv, `python -m pip install -r requirements.txt`, then (PowerShell):

    $env:DATABASE_URL="mysql+pymysql://root:@127.0.0.1:3306/nids"
    python -m uvicorn main:app --reload

If root has a password: mysql+pymysql://root:PASSWORD@127.0.0.1:3306/nids

On startup the backend (a) adds extra NULLABLE columns to `predictions` (see nids_migration.sql; nothing is
removed or changed) and (b) registers this model in `experiments` (model = "Hybrid RF+XGB+AE") and uses its exp_id.

## 3. Test
    python test_sample.py        -> True / True / True   (does not write to the database)

## How it uses Rayan's tables
- flows        : new flows posted to /predict are inserted (features JSON with underscore names, label/split NULL)
- predictions  : predicted_label = attack | benign | unseen_anomaly ; confidence ; plus the new columns
                 rf/xgb probabilities, recon_error, anomaly_flag, final_verdict
- alerts       : created for every attack and unseen_anomaly; severity high>=0.9, medium>=0.7, else low (his rule)
- drift_events : written when ADWIN detects drift in the autoencoder error (score = the error)
- experiments  : one row for this model version

## Endpoints
GET /health, /features, /predictions/recent, /alerts/recent, /drift/status
POST /predict, /predict/batch   (add ?store=false to skip the database)
POST /predict/stored?limit=200&split=test   (scores flows already in the database)
PATCH /alerts/{id}?status=ack|closed
