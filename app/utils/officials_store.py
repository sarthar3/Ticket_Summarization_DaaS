import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock

OFFICIALS_FILE = Path(__file__).resolve().parents[2] / "data" / "company_officials.json"
_officials_lock = Lock()

DEFAULT_OFFICIALS = [
    {
        "id": "OFF-101",
        "name": "Vikram Sharma",
        "designation": "Head of SRE & Cloud Infrastructure",
        "email": "vikram.sre@company.com",
        "department": "Cloud Infrastructure",
        "is_primary": True
    },
    {
        "id": "OFF-102",
        "name": "Priya Sundaram",
        "designation": "Lead Architect - Core Banking & Payment Gateways",
        "email": "priya.banking@company.com",
        "department": "Banking & Financial Services",
        "is_primary": True
    },
    {
        "id": "OFF-103",
        "name": "Arvind Mehta",
        "designation": "Director of Cybersecurity & SOC Incident Response",
        "email": "arvind.security@company.com",
        "department": "Cybersecurity & Fraud",
        "is_primary": True
    },
    {
        "id": "OFF-104",
        "name": "Ananya Roy",
        "designation": "Principal Engineer - Mobile Client & Auth Services",
        "email": "ananya.mobile@company.com",
        "department": "Authentication & Mobile Client",
        "is_primary": True
    },
    {
        "id": "OFF-105",
        "name": "Rajesh Gopinath",
        "designation": "VP of Operations & Logistics Fulfillment",
        "email": "rajesh.logistics@company.com",
        "department": "Logistics & Fulfillment",
        "is_primary": False
    }
]

def _ensure_officials_file():
    OFFICIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not OFFICIALS_FILE.exists():
        with open(OFFICIALS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_OFFICIALS, f, indent=2)

def get_all_officials() -> List[Dict[str, Any]]:
    """Retrieves all registered company officials."""
    _ensure_officials_file()
    with _officials_lock:
        try:
            with open(OFFICIALS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return list(DEFAULT_OFFICIALS)

def add_official(name: str, designation: str, email: str, department: str = "General Engineering") -> Dict[str, Any]:
    """Adds a new company official to the directory."""
    _ensure_officials_file()
    with _officials_lock:
        try:
            with open(OFFICIALS_FILE, "r", encoding="utf-8") as f:
                officials = json.load(f)
        except Exception:
            officials = []

        new_id = f"OFF-{len(officials) + 101}"
        new_official = {
            "id": new_id,
            "name": name.strip(),
            "designation": designation.strip(),
            "email": email.strip(),
            "department": department.strip(),
            "is_primary": False
        }
        officials.append(new_official)

        with open(OFFICIALS_FILE, "w", encoding="utf-8") as f:
            json.dump(officials, f, indent=2)

        return new_official

def delete_official(official_id: str) -> bool:
    """Removes an official from the directory by ID."""
    _ensure_officials_file()
    with _officials_lock:
        try:
            with open(OFFICIALS_FILE, "r", encoding="utf-8") as f:
                officials = json.load(f)
        except Exception:
            return False

        updated = [o for o in officials if o.get("id") != official_id]
        if len(updated) == len(officials):
            return False

        with open(OFFICIALS_FILE, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)
        return True

def get_official_for_category(category_or_domain: str) -> Dict[str, Any]:
    """Finds the most suitable official based on department or domain matching."""
    officials = get_all_officials()
    if not officials:
        return DEFAULT_OFFICIALS[0]

    cat_lower = (category_or_domain or "").lower()
    
    # Priority matches
    if any(k in cat_lower for k in ["cloud", "infra", "503", "gateway", "system", "outage"]):
        for o in officials:
            if "cloud" in o.get("department", "").lower() or "sre" in o.get("designation", "").lower():
                return o
    if any(k in cat_lower for k in ["banking", "payment", "finance", "billing", "debit", "ledger"]):
        for o in officials:
            if "banking" in o.get("department", "").lower() or "payment" in o.get("designation", "").lower():
                return o
    if any(k in cat_lower for k in ["fraud", "security", "unauthorized", "soc", "stolen", "compromise"]):
        for o in officials:
            if "security" in o.get("department", "").lower() or "fraud" in o.get("department", "").lower():
                return o
    if any(k in cat_lower for k in ["auth", "login", "session", "ios", "token", "mobile"]):
        for o in officials:
            if "auth" in o.get("department", "").lower() or "mobile" in o.get("department", "").lower():
                return o
    if any(k in cat_lower for k in ["logistics", "shipping", "order", "delivery", "carrier", "warehouse"]):
        for o in officials:
            if "logistics" in o.get("department", "").lower() or "fulfillment" in o.get("department", "").lower():
                return o

    # Default to first primary official or first official
    for o in officials:
        if o.get("is_primary"):
            return o
    return officials[0]
