@echo off
@REM script to run entire pipeline from gathering data subset, to running analysis

echo Running get_sample_dataset.py...
python scripts/get_sample_dataset.py

echo Running move_sample_images.py...
python scripts/move_sample_images.py

echo Running preprocess_images.py...
python scripts/preprocess_images.py

echo Running analyze_data.py...
python scripts/analyze_data.py

pause
