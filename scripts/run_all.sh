#!/bin/bash
# Full study, resumable: every step skips what is already computed. Run from the medaudit folder.
set -e
cd "$(dirname "$0")/.."
LOG=run_all.log
echo "=== $(date) start" | tee -a $LOG
python scripts/00_inspect.py 2>&1 | tee -a $LOG
# 1. datasets (small and signal datasets first, images last because embedding takes longest)
python scripts/01_prepare.py uci_voice darwin pamap2 uci_har mitbih chbmit hw_pahaw hw_newhandpdsig 2>&1 | tee -a $LOG
python scripts/01_prepare.py abide ham10000 isic2018_test ptbxl 2>&1 | tee -a $LOG
python scripts/01_prepare.py nih_cxr chexpert 2>&1 | tee -a $LOG
# 2. audits
python scripts/02_audit.py 2>&1 | tee -a $LOG
# 3. simulation (synthetic background, then real background from the voice dataset)
python scripts/03_simulation.py --reps 50 2>&1 | tee -a $LOG
python scripts/03_simulation.py --reps 20 --base uci_voice 2>&1 | tee -a $LOG
# 4. predictive validity
python scripts/06_benchmark_model.py 2>&1 | tee -a $LOG
python scripts/04_predictive_validity.py 2>&1 | tee -a $LOG
# 5. tables and figures
python scripts/07_new_participant_validation.py 2>&1 | tee -a $LOG
python scripts/05_tables_figures.py 2>&1 | tee -a $LOG
echo "=== $(date) done" | tee -a $LOG
