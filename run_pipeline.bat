@echo off
echo ========================================
echo Membangun Dataset (Ambang Batas Baru: 83)
echo ========================================
python src\build_dataset.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Ekstraksi Fitur (Multi-Cutoff + EWS)
echo ========================================
python src\features.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Baseline S1-S5 (Retrospektif)
echo ========================================
python scripts\run_baseline_experiments.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo EWS Multi-Cutoff (W1,W2,W3,C_Full) + Dummy Baseline
echo ========================================
python scripts\run_optimized_experiment.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo SHAP untuk S3 Dasar
echo ========================================
python scripts\generate_shap_s3_dasar.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo SHAP untuk Model Final (C_Full)
echo ========================================
python scripts\generate_shap.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Generate HASIL_ANALISIS_V3.md
echo ========================================
python scripts\generate_analysis_v3.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Pipeline selesai! HASIL_ANALISIS_V3.md telah diperbarui.
echo ========================================
pause
