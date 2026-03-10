#!/usr/bin/env python3
"""Render model_architecture.html → model_architecture.png via headless Chrome."""
import pathlib, time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

ROOT = pathlib.Path(__file__).resolve().parent
HTML = ROOT / "checkpoints_blind" / "model_architecture.html"
OUT  = ROOT / "checkpoints_blind" / "model_architecture.png"

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-gpu")
opts.add_argument("--force-device-scale-factor=2")   # 2× for retina-quality
opts.add_argument("--window-size=1200,1000")

driver = webdriver.Chrome(options=opts)
driver.get(HTML.as_uri())
time.sleep(1.5)  # let fonts load

# resize to actual content
body = driver.find_element("tag name", "body")
w = driver.execute_script("return document.body.scrollWidth")
h = driver.execute_script("return document.body.scrollHeight")
driver.set_window_size(w + 40, h + 40)
time.sleep(0.3)

driver.save_screenshot(str(OUT))
driver.quit()

print(f"✓ Saved {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
