@echo off
echo ========================================
echo Membangun Dataset (Ambang Batas Baru: 83)
echo ========================================
python src\build_dataset.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Ekstraksi Fitur
echo ========================================
python src\features.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Menjalankan Eksperimen Teroptimasi
echo ========================================
python scripts\run_optimized_experiment.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Pipeline selesai! Hasil telah diperbarui.
echo ========================================
pause
