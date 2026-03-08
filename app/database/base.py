"""
Database Base Configuration
SQLAlchemy declarative base and shared utilities
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from typing import Generator

# Database URL - Can be configured via environment variable
# Default: SQLite for development, switch to PostgreSQL for production
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./icdi_x.db"  # SQLite for local development
    # For production: "postgresql://user:password@localhost/icdi_x_db"
)

# Create SQLAlchemy engine
# For SQLite, we need check_same_thread=False
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

# Create SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()


def get_db() -> Generator:
    """
    FastAPI dependency for database sessions
    Yields a database session and ensures it's closed after use
    
    Usage:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables
    Called during application startup
    """
    Base.metadata.create_all(bind=engine)
