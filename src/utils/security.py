import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict
import os
from functools import wraps
import logging
from ratelimit import limits, sleep_and_retry
import secrets
import re

logger = logging.getLogger(__name__)

class SecurityManager:
    def __init__(self):
        self.secret_key = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
        self.token_expiry = timedelta(hours=1)
        self.max_requests_per_minute = 60
        self.blocked_ips = set()
        self.failed_attempts = {}

    def generate_token(self, user_id: str) -> str:
        """Generate a JWT token for authentication"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + self.token_expiry,
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None

    @sleep_and_retry
    @limits(calls=60, period=60)
    def rate_limit_check(self, ip_address: str):
        """Implement rate limiting"""
        if ip_address in self.blocked_ips:
            raise SecurityError("IP address blocked")

    def sanitize_input(self, input_text: str) -> str:
        """Sanitize user input"""
        # Remove potential SQL injection patterns
        sql_patterns = r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE)\b)'
        input_text = re.sub(sql_patterns, '', input_text, flags=re.IGNORECASE)
        
        # Remove script tags
        input_text = re.sub(r'<script.*?>.*?</script>', '', input_text, flags=re.IGNORECASE|re.DOTALL)
        
        # Remove potential command injection characters
        input_text = re.sub(r'[;&|`]', '', input_text)
        
        return input_text

    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        valid_key = os.getenv('API_KEY')
        if not valid_key:
            logger.error("API key not configured")
            return False
        return secrets.compare_digest(api_key, valid_key)

class SecurityError(Exception):
    pass

def secure_endpoint(f):
    """Decorator for securing API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        security = SecurityManager()
        
        # Get request context (assuming Flask-like request object)
        from flask import request
        
        # Rate limiting
        security.rate_limit_check(request.remote_addr)
        
        # API key validation
        api_key = request.headers.get('X-API-Key')
        if not security.validate_api_key(api_key):
            raise SecurityError("Invalid API key")
        
        # Token validation
        token = request.headers.get('Authorization')
        if not token or not security.verify_token(token.split(' ')[1]):
            raise SecurityError("Invalid or missing token")
        
        # Input sanitization
        if request.json:
            sanitized_data = {
                k: security.sanitize_input(v) if isinstance(v, str) else v
                for k, v in request.json.items()
            }
            request.json = sanitized_data
        
        return f(*args, **kwargs)
    
    return decorated_function

class PasswordManager:
    def __init__(self):
        self.min_length = 12
        self.require_special = True
        self.require_numbers = True
        self.require_uppercase = True
        self.require_lowercase = True

    def hash_password(self, password: str) -> bytes:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    def verify_password(self, password: str, hashed: bytes) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode(), hashed)

    def validate_password_strength(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.min_length:
            return False
        if self.require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False
        if self.require_numbers and not re.search(r'\d', password):
            return False
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            return False
        if self.require_lowercase and not re.search(r'[a-z]', password):
            return False
        return True