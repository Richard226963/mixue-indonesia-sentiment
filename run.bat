@echo off
cd /d "%~dp0"
title mixue sentiment pipeline

where python >nul 2>nul
if errorlevel 1 (
    echo python not found. install it from python.org and tick "add to PATH".
    pause
    exit /b 1
)

python -m pip show instaloader >nul 2>nul
if errorlevel 1 (
    echo first run: installing dependencies. torch is a big download, this can take a while.
    python -m pip install -r requirements.txt
)

echo.
echo == 1/8 scraping instagram, login with your own account when asked ==
python scrape.py
if errorlevel 1 goto fail

echo == 2/8 scraping suara.com ==
python news.py

echo == 3/8 preprocessing + bot filter ==
python clean.py
if errorlevel 1 goto fail

echo == 4/8 indoBERT labeling + evaluation ==
python sentiment.py
if errorlevel 1 goto fail

echo == 5/8 absa ==
python absa.py

echo == 6/8 topics ==
python topics.py

echo == 7/8 analytics ==
python analytics.py

echo == 8/8 opening the dashboard in your browser, close the window to stop ==
python -m streamlit run dashboard.py
goto end

:fail
echo pipeline stopped, read the message above and run this file again.

:end
pause
