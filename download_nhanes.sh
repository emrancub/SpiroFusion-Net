#!/usr/bin/env bash
# ==============================================================================
# SpiroFusion-Net: Automated US CDC NHANES (2007-2012) External Data Downloader
# ==============================================================================
set -e
mkdir -p ./nhanes_external_data
cd ./nhanes_external_data

echo "[*] Downloading 18 official US CDC NHANES (2007-2012) .XPT files..."
for comp in SPX SPXRAW DEMO BMX SMQ MCQ; do
    wget -c "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/${comp}_E.XPT"
    wget -c "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/${comp}_F.XPT"
    wget -c "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/${comp}_G.XPT"
done
cd ..
echo "[✓] All 18 NHANES .XPT files downloaded into ./nhanes_external_data/"
python -u run_nhanes_external_validation.py --nhanes_dir ./nhanes_external_data --output_csv ./nhanes_external_data/nhanes_mapped_ukb_schema.csv
