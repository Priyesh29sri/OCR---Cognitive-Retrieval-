"""
Pytest Configuration
Shared fixtures and test setup
"""
import pytest
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database.base import Base
from app.models.user import User
from app.services.auth_service import auth_service


# Test database URL
TEST_DATABASE_URL = "sqlite:///./test_icdi_x.db"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def test_db():
    """
    Create a fresh test database for each test
    """
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(test_db):
    """Create a test user"""
    user = auth_service.create_user(
        db=test_db,
        email="testuser@example.com",
        password="testpassword123",
        full_name="Test User"
    )
    return user


@pytest.fixture
def test_token(test_user):
    """Generate JWT token for test user"""
    from datetime import timedelta
    token = auth_service.create_access_token(
        data={"sub": test_user.id, "email": test_user.email},
        expires_delta=timedelta(hours=1)
    )
    return token


@pytest.fixture
def auth_headers(test_token):
    """Create authorization headers"""
    return {"Authorization": f"Bearer {test_token}"}
