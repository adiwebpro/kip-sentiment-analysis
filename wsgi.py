# wsgi.py
import sys
import os

# Add your project directory to the Python path
path = '/home/yourusername/kip_sentiment_analysis'
if path not in sys.path:
    sys.path.insert(0, path)

# Import the Flask app
from app import app as application

# Optional: Configure any production settings
application.debug = False