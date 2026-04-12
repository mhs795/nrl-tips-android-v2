[app]
title = NRL Tips
package.name = nrltips
package.domain = org.nrltips
source.dir = .
source.include_exts = py,csv,json,xlsx,png,npz,txt
icon.filename = %(source.dir)s/icon.png
# (str) Full version
version = 1.0.1
# (int) Alpha, beta or release version
version.numeric = 101


requirements = python3,kivy==2.3.0,pandas,numpy,requests,charset-normalizer,beautifulsoup4,openpyxl,et_xmlfile,certifi

orientation = portrait
fullscreen = 0
android.permissions = INTERNET,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
android.api = 34
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a
android.enable_androidx = True

# Exclude large files not needed in the APK
source.exclude_patterns = __pycache__,*.pyc,*.pyo,.git,.github,bin,.buildozer

[buildozer]
log_level = 2
warn_on_root = 1
