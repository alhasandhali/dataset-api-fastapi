import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
import logging

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Shared SECRET_KEY from auth_api
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-to-a-strong-secret-key")
ALGORITHM = "HS256"

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Decode JWT token and return the username (sub)."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError as e:
        logger.error(f"JWT Error: {e}")
        raise credentials_exception
