# Workflow-CI (Kriteria 3 - MLflow Project + GitHub Actions)

Repo ini dibuat untuk memenuhi **Kriteria 3**: membuat **workflow CI** menggunakan **MLflow Project** agar proses training model berjalan otomatis saat trigger terpenuhi.

## Struktur Repository

## Trigger Workflow (CI)

Workflow akan berjalan ketika:

1. **Manual trigger** melalui GitHub Actions:
   - Tab **Actions** → pilih workflow **CI - Retrain MLflow Project** → klik **Run workflow**

2. **Otomatis saat push** jika ada perubahan di:
   - `MLProject/**`
   - `.github/workflows/**`

## Cara Kerja Singkat

Workflow akan:
1. Checkout repository
2. Setup Python
3. Install dependency (mlflow, pandas, numpy, scikit-learn)
4. Menjalankan MLflow Project:
   - `cd MLProject`
   - `mlflow run . --env-manager local`

## Bukti Kriteria

Bukti eksekusi workflow dan log job disiapkan pada folder pengumpulan submission (file `Workflow-CI.txt` beserta screenshot).
