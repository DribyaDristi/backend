[app]
title = ASL Detector
package.name = asldetector
package.domain = org.asldetector
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,h5,txt
version = 1.0

# Dependencies
requirements = python3,\
    kivy==2.2.1,\
    opencv-python,\
    numpy,\
    mediapipe==0.10.7,\
    tensorflow==2.13.0,\
    pillow

# Android specific settings
android.permissions = CAMERA, WRITE_EXTERNAL_STORAGE
android.api = 33
android.minapi = 21
android.sdk = 33
android.ndk = 25b
android.arch = arm64-v8a
android.accept_sdk_license = True

# Include your model and assets
android.add_src = assets

# Build settings
android.gradle_dependencies = org.tensorflow:tensorflow-lite:2.13.0
android.enable_androidx = True

# App settings
orientation = portrait
fullscreen = 0
android.presplash_color = #FFFFFF
osx.python_version = 3

[buildozer]
log_level = 2
warn_on_root = 1