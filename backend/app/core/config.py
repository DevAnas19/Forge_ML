"""
config.py

Reads configuration from environment variables (spec section 29, rule 12:
keep secrets in environment variables). Uses python-dotenv to load a local
.env file during development, so you don't have to export vars manually
every time you open a terminal.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env if present; does nothing if it's not (e.g. in production)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://forgeml:forgeml123@localhost:5432/forgeml_db"  # local dev fallback only
)