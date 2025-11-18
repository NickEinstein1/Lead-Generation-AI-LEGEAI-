import hashlib
import hmac
import secrets
import base64
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import re
import redis

logger = logging.getLogger(__name__)

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"

@dataclass
class EncryptionKey:
    key_id: str
    key_data: bytes
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

@dataclass
class DataMaskingRule:
    field_name: str
    pii_type: PIIType
    masking_method: str  # "hash", "mask", "encrypt", "tokenize"
    preserve_format: bool = False
    mask_character: str = "*"

class DataProtectionManager:
    """Comprehensive data protection and encryption system"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 5
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Encryption keys
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        self.current_key_id = "default"
        
        # Initialize default encryption key
        self._initialize_encryption_keys()
        
        # PII detection patterns
        self.pii_patterns = self._load_pii_patterns()
        
        # Data masking rules
        self.masking_rules = self._load_masking_rules()
        
        # Tokenization vault
        self.token_vault: Dict[str, str] = {}
        
    def _initialize_encryption_keys(self):
        """Initialize encryption keys"""
        # Generate default key if not exists
        default_key = os.getenv('DEFAULT_ENCRYPTION_KEY')
        if not default_key:
            default_key = Fernet.generate_key()
        
        self.encryption_keys["default"] = EncryptionKey(
            key_id="default",
            key_data=default_key if isinstance(default_key, bytes) else default_key.encode(),
            algorithm="Fernet",
            created_at=datetime.now(timezone.utc)
        )
        
        # Generate RSA key pair for asymmetric encryption
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        self.rsa_private_key = private_key
        self.rsa_public_key = private_key.public_key()
    
    def _load_pii_patterns(self) -> Dict[PIIType, str]:
        """Load PII detection patterns"""
        return {
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            PIIType.SSN: r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b',
            PIIType.CREDIT_CARD: r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            PIIType.DATE_OF_BIRTH: r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b',
            PIIType.DRIVER_LICENSE: r'\b[A-Z]{1,2}[0-9]{6,8}\b',
            PIIType.PASSPORT: r'\b[A-Z]{1,2}[0-9]{6,9}\b'
        }
    
    def _load_masking_rules(self) -> List[DataMaskingRule]:
        """Load data masking rules"""
        return [
            DataMaskingRule("email", PIIType.EMAIL, "mask", preserve_format=True),
            DataMaskingRule("phone", PIIType.PHONE, "mask", preserve_format=True),
            DataMaskingRule("ssn", PIIType.SSN, "encrypt"),
            DataMaskingRule("credit_card", PIIType.CREDIT_CARD, "tokenize"),
            DataMaskingRule("first_name", PIIType.NAME, "mask"),
            DataMaskingRule("last_name", PIIType.NAME, "mask"),
            DataMaskingRule("address", PIIType.ADDRESS, "hash"),
            DataMaskingRule("date_of_birth", PIIType.DATE_OF_BIRTH, "encrypt"),
        ]
    
    def encrypt_data(self, data: Union[str, bytes], key_id: str = None) -> str:
        """Encrypt data using specified key"""
        try:
            key_id = key_id or self.current_key_id
            encryption_key = self.encryption_keys.get(key_id)
            
            if not encryption_key:
                raise ValueError(f"Encryption key not found: {key_id}")
            
            # Convert to bytes if string
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Use Fernet for symmetric encryption
            fernet = Fernet(encryption_key.key_data)
            encrypted_data = fernet.encrypt(data)
            
            # Return base64 encoded string with key ID prefix
            return f"{key_id}:{base64.b64encode(encrypted_data).decode('utf-8')}"
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data"""
        try:
            # Extract key ID and encrypted data
            if ':' in encrypted_data:
                key_id, encrypted_part = encrypted_data.split(':', 1)
            else:
                key_id = self.current_key_id
                encrypted_part = encrypted_data
            
            encryption_key = self.encryption_keys.get(key_id)
            if not encryption_key:
                raise ValueError(f"Encryption key not found: {key_id}")
            
            # Decode and decrypt
            encrypted_bytes = base64.b64decode(encrypted_part.encode('utf-8'))
            fernet = Fernet(encryption_key.key_data)
            decrypted_data = fernet.decrypt(encrypted_bytes)
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_asymmetric(self, data: Union[str, bytes]) -> str:
        """Encrypt data using RSA public key"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.rsa_public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Asymmetric encryption failed: {e}")
            raise
    
    def decrypt_asymmetric(self, encrypted_data: str) -> str:
        """Decrypt data using RSA private key"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            decrypted_data = self.rsa_private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Asymmetric decryption failed: {e}")
            raise
    
    def hash_data(self, data: str, salt: str = None) -> str:
        """Hash data with optional salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for secure hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode('utf-8'),
            iterations=100000,
        )
        
        key = kdf.derive(data.encode('utf-8'))
        return f"{salt}:{base64.b64encode(key).decode('utf-8')}"
    
    def verify_hash(self, data: str, hashed_data: str) -> bool:
        """Verify data against hash"""
        try:
            salt, hash_part = hashed_data.split(':', 1)
            expected_hash = self.hash_data(data, salt)
            return hmac.compare_digest(expected_hash, hashed_data)
        except Exception:
            return False
    
    def detect_pii(self, text: str) -> Dict[PIIType, List[str]]:
        """Detect PII in text"""
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def mask_data(self, data: str, pii_type: PIIType, preserve_format: bool = False, mask_char: str = "*") -> str:
        """Mask sensitive data"""
        if not data:
            return data
        
        if pii_type == PIIType.EMAIL:
            if preserve_format:
                # Mask username part: john.doe@example.com -> j***.***@example.com
                if '@' in data:
                    username, domain = data.split('@', 1)
                    masked_username = username[0] + mask_char * (len(username) - 1)
                    return f"{masked_username}@{domain}"
            return mask_char * len(data)
        
        elif pii_type == PIIType.PHONE:
            if preserve_format:
                # Mask middle digits: (555) 123-4567 -> (555) ***-4567
                return re.sub(r'(\d{3})\d{3}(\d{4})', r'\1***\2', data)
            return mask_char * len(data)
        
        elif pii_type == PIIType.SSN:
            if preserve_format:
                # Mask first 5 digits: 123-45-6789 -> ***-**-6789
                return re.sub(r'\d{3}-\d{2}', '***-**', data)
            return mask_char * len(data)
        
        elif pii_type == PIIType.CREDIT_CARD:
            if preserve_format:
                # Mask middle digits: 1234567890123456 -> 1234********3456
                return data[:4] + mask_char * (len(data) - 8) + data[-4:]
            return mask_char * len(data)
        
        elif pii_type == PIIType.NAME:
            if preserve_format and len(data) > 2:
                # Mask middle characters: John -> J**n
                return data[0] + mask_char * (len(data) - 2) + data[-1]
            return mask_char * len(data)
        
        else:
            # Default masking
            if preserve_format and len(data) > 4:
                return data[:2] + mask_char * (len(data) - 4) + data[-2:]
            return mask_char * len(data)
    
    def tokenize_data(self, data: str) -> str:
        """Tokenize sensitive data"""
        # Generate unique token
        token = f"TOK_{secrets.token_urlsafe(16)}"
        
        # Store mapping in vault (encrypted)
        encrypted_data = self.encrypt_data(data)
        self.token_vault[token] = encrypted_data
        
        # Store in Redis for persistence
        self.redis_client.setex(f"token:{token}", 86400 * 30, encrypted_data)  # 30 days
        
        return token
    
    def detokenize_data(self, token: str) -> Optional[str]:
        """Detokenize data"""
        try:
            # Check vault first
            encrypted_data = self.token_vault.get(token)
            
            if not encrypted_data:
                # Check Redis
                encrypted_data = self.redis_client.get(f"token:{token}")
                if encrypted_data:
                    encrypted_data = encrypted_data.decode('utf-8')
            
            if encrypted_data:
                return self.decrypt_data(encrypted_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Detokenization failed: {e}")
            return None
    
    def anonymize_dataset(self, data: Dict[str, Any], classification: DataClassification = DataClassification.CONFIDENTIAL) -> Dict[str, Any]:
        """Anonymize entire dataset based on classification"""
        anonymized_data = data.copy()
        
        for rule in self.masking_rules:
            if rule.field_name in anonymized_data:
                field_value = str(anonymized_data[rule.field_name])
                
                if rule.masking_method == "mask":
                    anonymized_data[rule.field_name] = self.mask_data(
                        field_value, rule.pii_type, rule.preserve_format, rule.mask_character
                    )
                elif rule.masking_method == "encrypt":
                    anonymized_data[rule.field_name] = self.encrypt_data(field_value)
                elif rule.masking_method == "hash":
                    anonymized_data[rule.field_name] = self.hash_data(field_value)
                elif rule.masking_method == "tokenize":
                    anonymized_data[rule.field_name] = self.tokenize_data(field_value)
        
        # Add anonymization metadata
        anonymized_data['_anonymization'] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'classification': classification.value,
            'method': 'automated_pii_protection'
        }
        
        return anonymized_data
    
    def secure_delete(self, data_identifier: str) -> bool:
        """Securely delete data"""
        try:
            # Remove from token vault
            tokens_to_remove = [token for token, _ in self.token_vault.items() if data_identifier in token]
            for token in tokens_to_remove:
                del self.token_vault[token]
                self.redis_client.delete(f"token:{token}")
            
            # Log secure deletion
            logger.info(f"Securely deleted data: {data_identifier}")
            return True
            
        except Exception as e:
            logger.error(f"Secure deletion failed: {e}")
            return False
    
    def generate_privacy_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate privacy compliance report for data"""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_classification': DataClassification.CONFIDENTIAL.value,
            'pii_detected': {},
            'protection_applied': {},
            'compliance_status': 'compliant'
        }
        
        # Detect PII in all text fields
        for field_name, field_value in data.items():
            if isinstance(field_value, str):
                detected_pii = self.detect_pii(field_value)
                if detected_pii:
                    report['pii_detected'][field_name] = {
                        pii_type.value: matches for pii_type, matches in detected_pii.items()
                    }
        
        # Check protection applied
        for rule in self.masking_rules:
            if rule.field_name in data:
                report['protection_applied'][rule.field_name] = {
                    'method': rule.masking_method,
                    'pii_type': rule.pii_type.value
                }
        
        return report

# Global data protection manager
data_protection = DataProtectionManager()

# Decorator for automatic data protection
def protect_pii(classification: DataClassification = DataClassification.CONFIDENTIAL):
    """Decorator to automatically protect PII in function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                return data_protection.anonymize_dataset(result, classification)
            elif isinstance(result, list):
                return [
                    data_protection.anonymize_dataset(item, classification) 
                    if isinstance(item, dict) else item 
                    for item in result
                ]
            
            return result
        
        return wrapper
    return decorator

# Example usage
def example_usage():
    """Example of data protection usage"""
    
    # Sample sensitive data
    sensitive_data = {
        'first_name': 'John',
        'last_name': 'Doe',
        'email': 'john.doe@example.com',
        'phone': '(555) 123-4567',
        'ssn': '123-45-6789',
        'credit_card': '4532123456789012',
        'address': '123 Main St, Anytown, ST 12345',
        'date_of_birth': '01/15/1985'
    }
    
    print("Original data:")
    print(json.dumps(sensitive_data, indent=2))
    
    # Detect PII
    text = "Contact John Doe at john.doe@example.com or (555) 123-4567. SSN: 123-45-6789"
    detected_pii = data_protection.detect_pii(text)
    print(f"\nDetected PII: {detected_pii}")
    
    # Anonymize dataset
    anonymized_data = data_protection.anonymize_dataset(sensitive_data)
    print("\nAnonymized data:")
    print(json.dumps(anonymized_data, indent=2))
    
    # Generate privacy report
    privacy_report = data_protection.generate_privacy_report(sensitive_data)
    print("\nPrivacy report:")
    print(json.dumps(privacy_report, indent=2))
    
    # Test encryption/decryption
    original_text = "This is sensitive information"
    encrypted = data_protection.encrypt_data(original_text)
    decrypted = data_protection.decrypt_data(encrypted)
    print(f"\nEncryption test: {original_text} -> {encrypted[:50]}... -> {decrypted}")
    
    # Test tokenization
    credit_card = "4532123456789012"
    token = data_protection.tokenize_data(credit_card)
    detokenized = data_protection.detokenize_data(token)
    print(f"Tokenization test: {credit_card} -> {token} -> {detokenized}")

if __name__ == "__main__":
    example_usage()