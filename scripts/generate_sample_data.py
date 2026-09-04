import json
from pathlib import Path

SAMPLE_DATASET = [
    {
        "ticket_id": "TICK-1001",
        "ticket_text": "Customer reported being unable to login to their mobile banking application after the recent v4.2 update. They receive error code ERR-902 on iOS 17.4. User tried resetting password and reinstalling the application without success. Customer needs access urgently to process payroll.",
        "summary": "User unable to login to mobile banking app following v4.2 update on iOS (Error ERR-902). Password reset and reinstall failed; customer needs urgent access for payroll.",
        "sector": "Fintech",
        "intent": "Technical Support",
        "category": "Authentication",
        "priority": "High"
    },
    {
        "ticket_id": "TICK-1002",
        "ticket_text": "Order #88392 was placed on October 12th for express 2-day shipping, but tracking status has been stuck at 'In Transit - Carrier Origin Facility' for 5 days. Customer is requesting a full refund of shipping fees and an updated estimated delivery date.",
        "summary": "Order #88392 express shipping delayed 5 days in transit. Customer requests full shipping fee refund and updated delivery date.",
        "sector": "E-commerce",
        "intent": "Order Status",
        "category": "Shipping & Logistics",
        "priority": "Medium"
    },
    {
        "ticket_id": "TICK-1003",
        "ticket_text": "Enterprise client noticed a double billing charge of $1,499 on invoice #INV-2024-099. Both the annual subscription fee and monthly add-on were processed twice on their corporate credit card. Requesting immediate credit memo and refund to card.",
        "summary": "Enterprise customer charged twice ($1,499) on invoice #INV-2024-099 for annual and monthly subscription. Requesting immediate refund and credit memo.",
        "sector": "SaaS",
        "intent": "Billing Issue",
        "category": "Payment Discrepancy",
        "priority": "Urgent"
    },
    {
        "ticket_id": "TICK-1004",
        "ticket_text": "Subscribed user experiencing intermittent fiber broadband connection drops every 20-30 minutes in the Austin, TX area. Router power cycle temporary fixes the issue for 10 minutes before dropping again. Optical line terminal LED status indicates line noise.",
        "summary": "Intermittent fiber broadband connection drops in Austin, TX area. Line noise indicated on OLT; temporary router resets fail.",
        "sector": "Telecom",
        "intent": "Outage Reporting",
        "category": "Network Infrastructure",
        "priority": "High"
    },
    {
        "ticket_id": "TICK-1005",
        "ticket_text": "Patient inquiring about prescription refill authorization for medication ID #RX-99201. Pharmacist states insurance coverage denial due to missing prior authorization form from primary care physician. Patient needs PC provider follow-up.",
        "summary": "Prescription refill denied by insurance due to missing prior authorization form. Patient requests physician follow-up.",
        "sector": "Healthcare",
        "intent": "Service Request",
        "category": "Insurance Authorization",
        "priority": "Medium"
    },
    {
        "ticket_id": "TICK-1006",
        "ticket_text": "Cloud API customer getting HTTP 429 Too Many Requests when calling endpoint /v1/data/extract, despite having an active tier 3 SLA permit allowing up to 1,000 requests per minute. Current usage spike reached only 450 rpm.",
        "summary": "API client receiving HTTP 429 error at 450 RPM despite Tier 3 SLA permit allowing up to 1,000 RPM. Escalated for rate limiter threshold adjustment.",
        "sector": "Cloud Infrastructure",
        "intent": "API Error",
        "category": "Rate Limiting",
        "priority": "High"
    },
    {
        "ticket_id": "TICK-1007",
        "ticket_text": "User wants to request deletion of all personal identifiable information (PII) under GDPR Article 17 provisions. User account handle @johndoe99. Account was closed 30 days ago.",
        "summary": "Customer requesting complete GDPR PII data erasure for closed account @johndoe99.",
        "sector": "SaaS",
        "intent": "Compliance Request",
        "category": "Data Privacy",
        "priority": "Low"
    },
    {
        "ticket_id": "TICK-1008",
        "ticket_text": "Merchant payment gateway failing to process Visa transactions starting at 14:00 UTC. Mastercard and AMEX payments operating normally. Error response code 504 Gateway Timeout returned from acquiring bank.",
        "summary": "Visa payment processing failure with HTTP 504 timeouts via acquiring bank. AMEX and Mastercard unaffected.",
        "sector": "Fintech",
        "intent": "Outage Reporting",
        "category": "Payment Gateway",
        "priority": "Urgent"
    }
]

def generate_sample_dataset(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ticket in SAMPLE_DATASET:
            f.write(json.dumps(ticket) + "\n")
    print(f"Sample dataset generated successfully with {len(SAMPLE_DATASET)} records at: {output_path}")

if __name__ == "__main__":
    sample_path = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_tickets.jsonl"
    generate_sample_dataset(sample_path)
