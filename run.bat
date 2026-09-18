@echo off
pip install -r requirements.txt --upgrade-strategy=only-if-needed
pause
python pipeline.py "./Videos"