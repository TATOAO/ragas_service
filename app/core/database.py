from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
import asyncio
from typing import AsyncGenerator
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=StaticPool,
    pool_pre_ping=True,
    echo=settings.DEBUG
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLModel base class
Base = SQLModel


def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_db():
    """Initialize database tables"""
    try:
        # Import all models to ensure they are registered
        from app.models import dataset, evaluation, sample, metric
        
        # Create all tables
        SQLModel.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db():
    """Close database connections"""
    try:
        engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


# test database connection
# python -m app.core.database
if __name__ == "__main__":

    print("environment variable")

    print(f"DATABASE_URL: {settings.DATABASE_URL}")
    print(f"POSTGRES_USER: {settings.POSTGRES_USER}")
    print(f"POSTGRES_PASSWORD: {settings.POSTGRES_PASSWORD}")
    print(f"POSTGRES_HOSTNAME: {settings.POSTGRES_HOSTNAME}")
    print(f"POSTGRES_DATABASE: {settings.POSTGRES_DATABASE}")
    print(f"POSTGRES_PORT: {settings.POSTGRES_PORT}")


    from sqlalchemy import text
    db = SessionLocal()
    print(db.execute(text("SELECT 1")))
    db.close()

    print("Database connection test passed")