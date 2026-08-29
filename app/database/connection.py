import os
import shutil
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_URL, VIDEOS_DIR, FRAMES_DIR

# SQLite specific argument connect_args={"check_same_thread": False} is required
# because FastAPI can handle requests across multiple threads.
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    """Initializes the database schema and ensures tables exist."""
    # Import models so they are registered with Base metadata
    import app.database.models  # noqa: F401
    Base.metadata.create_all(bind=engine)

def reset_db(clear_files: bool = True):
    """
    Clears all data in the database by dropping and recreating all tables.
    Optionally removes all stored video and frame files.
    """
    import app.database.models  # noqa: F401
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    if clear_files:
        # Clear videos directory
        if os.path.exists(VIDEOS_DIR):
            for item in os.listdir(VIDEOS_DIR):
                item_path = os.path.join(VIDEOS_DIR, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error cleaning {item_path}: {e}")
                    
        # Clear frames directory
        if os.path.exists(FRAMES_DIR):
            for item in os.listdir(FRAMES_DIR):
                item_path = os.path.join(FRAMES_DIR, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error cleaning {item_path}: {e}")

def get_db():
    """Dependency for getting database sessions in FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
