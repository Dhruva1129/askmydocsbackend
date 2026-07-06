"""
Security Constants & RBAC Configuration
-----------------------------------------
Central place for all roles, access levels, and permission mappings.
"""

from enum import Enum
from typing import Dict, List, Set

# ── Roles ────────────────────────────────────────────────────────
class UserRole(str, Enum):
    ADMIN     = "admin"
    HR        = "hr"
    FINANCE   = "finance"
    MANAGER   = "manager"
    DEVELOPER = "developer"
    EMPLOYEE  = "employee"


# ── Access levels (document confidentiality) ─────────────────────
class AccessLevel(str, Enum):
    PUBLIC       = "public"
    INTERNAL     = "internal"
    CONFIDENTIAL = "confidential"
    SECRET       = "secret"


# ── Document types ───────────────────────────────────────────────
class DocumentType(str, Enum):
    POLICY      = "policy"
    FINANCIAL   = "financial"
    HR_DATA     = "hr_data"
    TECHNICAL   = "technical"
    GENERAL     = "general"
    CONTRACT    = "contract"


# ── Departments ─────────────────────────────────────────────────
class Department(str, Enum):
    HR          = "HR"
    FINANCE     = "Finance"
    ENGINEERING = "Engineering"
    SALES       = "Sales"
    LEGAL       = "Legal"
    GENERAL     = "General"


# ── RBAC Permission Matrix ───────────────────────────────────────
# Maps each role → which departments they can access docs from
ROLE_DEPARTMENT_ACCESS: Dict[str, Set[str]] = {
    UserRole.ADMIN:     {"HR", "Finance", "Engineering", "Sales", "Legal", "General"},
    UserRole.HR:        {"HR", "General"},
    UserRole.FINANCE:   {"Finance", "General"},
    UserRole.MANAGER:   {"HR", "Finance", "Engineering", "Sales", "General"},
    UserRole.DEVELOPER: {"Engineering", "General"},
    UserRole.EMPLOYEE:  {"General"},
}

# Maps each role → which access levels they can read
ROLE_ACCESS_LEVEL: Dict[str, List[str]] = {
    UserRole.ADMIN:     ["public", "internal", "confidential", "secret"],
    UserRole.HR:        ["public", "internal", "confidential"],
    UserRole.FINANCE:   ["public", "internal", "confidential"],
    UserRole.MANAGER:   ["public", "internal", "confidential"],
    UserRole.DEVELOPER: ["public", "internal"],
    UserRole.EMPLOYEE:  ["public"],
}

# Which roles can see salary / compensation data
SALARY_ALLOWED_ROLES: Set[str] = {UserRole.ADMIN, UserRole.HR, UserRole.FINANCE, UserRole.MANAGER}

# Which roles can upload documents
UPLOAD_ALLOWED_ROLES: Set[str] = {UserRole.ADMIN, UserRole.MANAGER}

# Which roles can access the admin panel
ADMIN_PANEL_ROLES: Set[str] = {UserRole.ADMIN}

# ── JWT Configuration ────────────────────────────────────────────
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# ── Prompt Injection Patterns ────────────────────────────────────
import re

INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r'ignore\s+(previous|all|above|prior)\s+instructions?', re.I),
    re.compile(r'reveal\s+(confidential|secret|private|hidden|all)', re.I),
    re.compile(r'show\s+(me\s+)?(all\s+)?(?:employee\s+)?(?:salary|salaries|compensation|password|secret|api.?key)', re.I),
    re.compile(r'bypass\s+(?:the\s+)?(?:permission|security|access|auth|restriction)', re.I),
    re.compile(r'act\s+as\s+(an?\s+)?(admin|root|superuser|system)', re.I),
    re.compile(r'you\s+are\s+now\s+(a\s+)?(different|new|another|unrestricted)', re.I),
    re.compile(r'forget\s+(everything|all|previous|your\s+instructions)', re.I),
    re.compile(r'\bjailbreak\b', re.I),
    re.compile(r'\bDAN\s+mode\b', re.I),
    re.compile(r'pretend\s+(you|that).*(no|without).*(restriction|limit|filter)', re.I),
    re.compile(r'system\s+prompt', re.I),
    re.compile(r'override\s+(system|instructions|rules)', re.I),
]

# ── Sensitive Data Patterns (for masking) ─────────────────────────
SENSITIVE_PATTERNS: Dict[str, re.Pattern] = {
    # Salary: matches 'salary: ₹X', 'salary of X is ₹Y', 'pay: X', Indian lakh notation, etc.
    "salary": re.compile(
        r'(?:'
        # Format 1: keyword followed by colon/is/= and currency+number
        r'(?:salary|compensation|wage|pay|ctc|package|remuneration)'
        r'(?:[^.\n]{0,50}?)'
        r'(?:is|are|of|:)?\s*'
        r'(?:Rs\.?|₹|INR|\$|£|€|₹)?\s*[\d,]+(?:\.\d+)?(?:\s*(?:lakh|lakhs|crore|crores|k|thousand|million))?'
        r'|'
        # Format 2: currency symbol directly before number (standalone)
        r'(?:Rs\.?|₹|INR)\s*[\d,]{4,}(?:\.\d+)?'
        r')',
        re.I
    ),
    "ssn":         re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "password":    re.compile(r'(?:password|passwd|pwd)[:\s]+\S+', re.I),
    "api_key":     re.compile(r'(?:api[_-]?key|api[_-]?token|bearer)[:\s]+[A-Za-z0-9_\-\.]{20,}', re.I),
    "credit_card": re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'),
    # Phone: require at least 10 consecutive digits (not dates like 2015 or page numbers)
    "phone":       re.compile(r'\b(?:\+91[\s-]?)?[6-9]\d{9}\b'),
    "bank_acc":    re.compile(r'\b(?:account|acc)[.:\s#]+\d{8,20}\b', re.I),
}
