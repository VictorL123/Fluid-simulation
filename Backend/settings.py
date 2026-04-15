"""
Django settings for the Fluid Simulation API backend.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SECRET_KEY = 'dev-secret-key-change-in-production'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.auth',
    'corsheaders',
    'rest_framework',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'urls'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Allow Node.js frontend to talk to Django
CORS_ALLOW_ALL_ORIGINS = True

REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ]
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Path to C++ solver binary — adjust if needed
SOLVER_DIR = str(BASE_DIR.parent / 'solver')
SOLVER_BIN_WINDOWS = str(BASE_DIR.parent / 'solver' / 'fluid_sim_omp.exe')
SOLVER_BIN_UNIX    = str(BASE_DIR.parent / 'solver' / 'fluid_sim_omp')
