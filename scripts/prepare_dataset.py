import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.dataset import DatasetSplitter
from app.preprocessing.token_stats import calculate_token_statistics
from app.config.settings import get_settings

EXPANDED_TICKET_DATASET = [
    {
        "ticket_id": "TICK-2001",
        "ticket_text": "Customer reported being unable to login to their mobile banking application after the recent v4.2 update. They receive error code ERR-902 on iOS 17.4. User tried resetting password and reinstalling the application without success. Customer needs access urgently to process payroll.",
        "summary": "User unable to login to mobile banking app following v4.2 update on iOS (Error ERR-902). Password reset and reinstall failed; customer needs urgent access for payroll.",
        "sector": "Fintech", "intent": "Technical Support", "category": "Authentication", "priority": "High"
    },
    {
        "ticket_id": "TICK-2002",
        "ticket_text": "Order #88392 was placed on October 12th for express 2-day shipping, but tracking status has been stuck at 'In Transit - Carrier Origin Facility' for 5 days. Customer is requesting a full refund of shipping fees and an updated estimated delivery date.",
        "summary": "Order #88392 express shipping delayed 5 days in transit. Customer requests full shipping fee refund and updated delivery date.",
        "sector": "E-commerce", "intent": "Order Status", "category": "Shipping & Logistics", "priority": "Medium"
    },
    {
        "ticket_id": "TICK-2003",
        "ticket_text": "Enterprise client noticed a double billing charge of $1,499 on invoice #INV-2024-099. Both the annual subscription fee and monthly add-on were processed twice on their corporate credit card. Requesting immediate credit memo and refund to card.",
        "summary": "Enterprise customer charged twice ($1,499) on invoice #INV-2024-099 for annual and monthly subscription. Requesting immediate refund and credit memo.",
        "sector": "SaaS", "intent": "Billing Issue", "category": "Payment Discrepancy", "priority": "Urgent"
    },
    {
        "ticket_id": "TICK-2004",
        "ticket_text": "Subscribed user experiencing intermittent fiber broadband connection drops every 20-30 minutes in the Austin, TX area. Router power cycle temporary fixes the issue for 10 minutes before dropping again. Optical line terminal LED status indicates line noise.",
        "summary": "Intermittent fiber broadband connection drops in Austin, TX area. Line noise indicated on OLT; temporary router resets fail.",
        "sector": "Telecom", "intent": "Outage Reporting", "category": "Network Infrastructure", "priority": "High"
    },
    {
        "ticket_id": "TICK-2005",
        "ticket_text": "Patient inquiring about prescription refill authorization for medication ID #RX-99201. Pharmacist states insurance coverage denial due to missing prior authorization form from primary care physician. Patient needs PC provider follow-up.",
        "summary": "Prescription refill denied by insurance due to missing prior authorization form. Patient requests physician follow-up.",
        "sector": "Healthcare", "intent": "Service Request", "category": "Insurance Authorization", "priority": "Medium"
    },
    {
        "ticket_id": "TICK-2006",
        "ticket_text": "Cloud API customer getting HTTP 429 Too Many Requests when calling endpoint /v1/data/extract, despite having an active tier 3 SLA permit allowing up to 1,000 requests per minute. Current usage spike reached only 450 rpm.",
        "summary": "API client receiving HTTP 429 error at 450 RPM despite Tier 3 SLA permit allowing up to 1,000 RPM. Escalated for rate limiter threshold adjustment.",
        "sector": "Cloud Infrastructure", "intent": "API Error", "category": "Rate Limiting", "priority": "High"
    },
    {
        "ticket_id": "TICK-2007",
        "ticket_text": "User wants to request deletion of all personal identifiable information (PII) under GDPR Article 17 provisions. User account handle @johndoe99. Account was closed 30 days ago.",
        "summary": "Customer requesting complete GDPR PII data erasure for closed account @johndoe99.",
        "sector": "SaaS", "intent": "Compliance Request", "category": "Data Privacy", "priority": "Low"
    },
    {
        "ticket_id": "TICK-2008",
        "ticket_text": "Merchant payment gateway failing to process Visa transactions starting at 14:00 UTC. Mastercard and AMEX payments operating normally. Error response code 504 Gateway Timeout returned from acquiring bank.",
        "summary": "Visa payment processing failure with HTTP 504 timeouts via acquiring bank. AMEX and Mastercard unaffected.",
        "sector": "Fintech", "intent": "Outage Reporting", "category": "Payment Gateway", "priority": "Urgent"
    },
    {
        "ticket_id": "TICK-2009",
        "ticket_text": "User unable to export financial reports in PDF format from dashboard. Error log displays 'DOMException: Failed to execute render on Canvas'. Issue occurs across Chrome 122 and Edge browser on macOS Sonoma.",
        "summary": "PDF export failure in financial dashboard due to DOM Canvas execution error on macOS Chrome/Edge.",
        "sector": "Fintech", "intent": "Bug Report", "category": "Export Utility", "priority": "Medium"
    },
    {
        "ticket_id": "TICK-2010",
        "ticket_text": "E-commerce merchant reports inventory sync failure between Shopify store and ERP backend system. Stock quantities for SKU #SKU-992-BLK are out of sync by 150 units, causing overselling.",
        "summary": "Inventory synchronization error between Shopify and ERP for SKU #SKU-992-BLK resulting in overselling 150 units.",
        "sector": "E-commerce", "intent": "Technical Support", "category": "ERP Integration", "priority": "High"
    },
    {
        "ticket_id": "TICK-2011",
        "ticket_text": "Healthcare system administrator reports single sign-on (SSO) SAML assertion failure for hospital staff entering EMR portal. Error logs indicate expired X.509 certificate on identity provider side.",
        "summary": "SSO SAML authentication failure for EMR portal caused by expired X.509 certificate on IdP server.",
        "sector": "Healthcare", "intent": "Security Incident", "category": "SSO / Identity", "priority": "Urgent"
    },
    {
        "ticket_id": "TICK-2012",
        "ticket_text": "Logistics driver unable to access mobile routing app due to location permission denial popup loop on Android 14. Clearing app cache and re-enabling GPS permissions did not resolve the issue.",
        "summary": "Driver mobile app stuck in Android 14 GPS permission loop; cache reset failed.",
        "sector": "Logistics", "intent": "Technical Support", "category": "Mobile App", "priority": "High"
    },
    {
        "ticket_id": "TICK-2013",
        "ticket_text": "DevOps engineer reporting Kubernetes pod crash loop back-off error in cluster us-east-1a node pool. OOMKilled flag triggered by memory spike in Redis cache deployment.",
        "summary": "Kubernetes pod CrashLoopBackOff in us-east-1a triggered by Redis memory OOMKilled event.",
        "sector": "Cloud Infrastructure", "intent": "Incident Management", "category": "Kubernetes / Infrastructure", "priority": "Urgent"
    },
    {
        "ticket_id": "TICK-2014",
        "ticket_text": "Telecom subscriber requesting SIM swap due to misplaced physical SIM card. Verification completed via two-factor SMS and security passphrase. Requires eSIM QR activation code email.",
        "summary": "Verified SIM swap request for misplaced card; customer requires eSIM QR code via email.",
        "sector": "Telecom", "intent": "Account Service", "category": "SIM Activation", "priority": "Low"
    },
    {
        "ticket_id": "TICK-2015",
        "ticket_text": "Cybersecurity compliance team detected suspicious login attempts from unrecognized IP block 198.51.100.44 targeting admin dashboard. Requesting IP address ban on WAF rules.",
        "summary": "Suspicious admin login attempts from IP 198.51.100.44; requesting immediate WAF IP block.",
        "sector": "CyberSecurity", "intent": "Security Incident", "category": "WAF / Firewall", "priority": "Urgent"
    },
    {
        "ticket_id": "TICK-2016",
        "ticket_text": "SaaS user unable to add new team members under Pro Plan subscription seat limit. System erroneously reports max 5 users reached despite current active user count being 3.",
        "summary": "Seat limit error blocking team invites on Pro Plan despite having 2 available seats.",
        "sector": "SaaS", "intent": "Account Management", "category": "License Management", "priority": "Medium"
    },
    {
        "ticket_id": "TICK-2017",
        "ticket_text": "Customer received damaged product package for Order #77123. Outer box crumpled and glass bottle shattered inside. Customer submitted photos and requests replacement shipment.",
        "summary": "Damaged package received for Order #77123 with shattered item; replacement shipment requested.",
        "sector": "E-commerce", "intent": "Return / Damage Claim", "category": "Logistics / Damage", "priority": "Medium"
    },
    {
        "ticket_id": "TICK-2018",
        "ticket_text": "Fintech platform user reporting failed ACH withdrawal of $500 to linked Chase bank account. Transaction status changed to 'Rejected by Receiving Bank' with code R01 Insufficient Funds.",
        "summary": "ACH withdrawal of $500 rejected by bank due to R01 insufficient funds error.",
        "sector": "Fintech", "intent": "Transaction Query", "category": "ACH Processing", "priority": "Medium"
    },
    {
        "ticket_id": "TICK-2019",
        "ticket_text": "Cloud storage customer getting slow upload speeds (< 100 KB/s) to S3 bucket in eu-west-1 region from Frankfurt office. ISP traceroute shows zero packet loss; bucket transfer acceleration is disabled.",
        "summary": "Slow S3 bucket uploads (<100 KB/s) in eu-west-1; bucket transfer acceleration recommended.",
        "sector": "Cloud Infrastructure", "intent": "Performance Issue", "category": "Storage Network", "priority": "Low"
    },
    {
        "ticket_id": "TICK-2020",
        "ticket_text": "Healthcare portal user unable to view lab results PDF. Page hangs at loading spinner for 60 seconds before timing out. Network inspector reveals 504 Gateway Timeout from backend storage service.",
        "summary": "Lab results PDF failing to load with 504 Gateway Timeout from backend storage service.",
        "sector": "Healthcare", "intent": "Technical Support", "category": "EMR Portal", "priority": "High"
    }
]

def main():
    print("=" * 60)
    print("      TICKET SUMMARIZATION DaaS — PREPARE PHASE 2 DATASETS")
    print("=" * 60)

    settings = get_settings()
    splitter = DatasetSplitter(seed=settings.reproducibility.seed)

    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save raw dataset
    raw_file = raw_dir / "tickets_expanded_raw.jsonl"
    with open(raw_file, "w", encoding="utf-8") as f:
        for ticket in EXPANDED_TICKET_DATASET:
            f.write(json.dumps(ticket) + "\n")
    print(f"\n1. Raw dataset saved: {raw_file} ({len(EXPANDED_TICKET_DATASET)} records)")

    # 2. Split dataset into train, val, test
    train_split, val_split, test_split = splitter.split_dataset(
        records=EXPANDED_TICKET_DATASET,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    train_file = processed_dir / "train.jsonl"
    val_file = processed_dir / "val.jsonl"
    test_file = processed_dir / "test.jsonl"

    splitter.save_split(train_split, train_file)
    splitter.save_split(val_split, val_file)
    splitter.save_split(test_split, test_file)

    # 3. Print Token Statistics per split
    print("\n" + "=" * 60)
    print("            DATASET SPLIT TOKEN STATISTICS")
    print("=" * 60)
    splits_map = {
        "Train": train_split,
        "Validation": val_split,
        "Fixed Test Set": test_split
    }

    for name, records in splits_map.items():
        texts = [r["ticket_text"] for r in records]
        stats = calculate_token_statistics(texts, max_context_limit=settings.context.max_input_tokens)
        print(f"\n[{name.upper()} SPLIT — {stats.total_samples} samples]")
        print(f"  Min Token Length    : {stats.min_length}")
        print(f"  Max Token Length    : {stats.max_length}")
        print(f"  Mean Token Length   : {stats.mean_length:.2f}")
        print(f"  P50 (Median) Length : {stats.p50_length:.2f}")
        print(f"  P95 Token Length    : {stats.p95_length:.2f}")
        print(f"  Exceeding Max Context ({stats.max_context_limit}): {stats.exceeding_limit_count} ({stats.exceeding_limit_percentage}%)")

    print("=" * 60)

if __name__ == "__main__":
    main()
