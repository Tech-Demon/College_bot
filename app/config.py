import os
from dotenv import load_dotenv

load_dotenv()

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Gemini settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-001")

# College website settings
WEBSITE_URL = os.getenv("WEBSITE_URL", "https://your-college-website.edu")

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///college_data.db")

# MongoDB Vector Store settings
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "college_bot")

# PDF directory
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./pdfs")