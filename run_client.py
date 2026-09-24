import sys
import time
import argparse
import requests
import json

URL = "http://localhost:8000/api/v1/summarize"
HEALTH_URL = "http://localhost:8000/api/v1/health"

def wait_for_server(max_retries=30, delay=2):
    """Waits for the FastAPI server to complete startup and model loading."""
    print("Connecting to FastAPI server (http://localhost:8000)...")
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.get(HEALTH_URL, timeout=3)
            if res.status_code == 200 and res.json().get("model_loaded"):
                print("✅ Server is ready and model is loaded!\n")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        
        print(f"⏳ Server is starting up / loading model weights... Retry {attempt}/{max_retries} (waiting {delay}s)")
        time.sleep(delay)
    return False

def get_user_input():
    print("=" * 60)
    print("      TICKET SUMMARIZATION DAAS - INTERACTIVE CLIENT      ")
    print("=" * 60)
    
    ticket_id = input("\nEnter Ticket ID [default: TICK-USER-001]: ").strip()
    if not ticket_id:
        ticket_id = "TICK-USER-001"
        
    print("\nEnter Support Ticket Complaint Text:")
    ticket_text = input("> ").strip()
    
    while not ticket_text:
        print("⚠️ Ticket text cannot be empty! Please enter the ticket text:")
        ticket_text = input("> ").strip()
        
    sector = input("\nEnter Sector (e.g. Fintech, Healthcare, E-commerce) [optional]: ").strip()
    intent = input("Enter Intent (e.g. Technical Support, Billing) [optional]: ").strip()
    priority = input("Enter Priority (e.g. High, Medium, Low) [optional]: ").strip()

    return {
        "ticket_id": ticket_id,
        "ticket_text": ticket_text,
        "sector": sector if sector else None,
        "intent": intent if intent else None,
        "priority": priority if priority else None
    }

def main():
    parser = argparse.ArgumentParser(description="Run Ticket Summarization Client with custom input")
    parser.add_argument("--ticket_text", type=str, help="Customer support ticket text")
    parser.add_argument("--ticket_id", type=str, default="TICK-USER-001", help="Ticket ID")
    parser.add_argument("--sector", type=str, help="Sector")
    parser.add_argument("--intent", type=str, help="Intent")
    parser.add_argument("--priority", type=str, help="Priority")

    args = parser.parse_args()

    if args.ticket_text:
        payload = {
            "ticket_id": args.ticket_id,
            "ticket_text": args.ticket_text,
            "sector": args.sector,
            "intent": args.intent,
            "priority": args.priority
        }
    else:
        payload = get_user_input()

    if not wait_for_server():
        print("\n❌ Error: Service took too long to respond at http://localhost:8000.")
        print("Please ensure uvicorn is running: python -m uvicorn app.main:app --reload --port 8000")
        return

    print("\nSending ticket summarization request...")
    print("-" * 60)

    try:
        response = requests.post(URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        print("\n" + " SUMMARY RESPONSE ".center(60, "="))
        print(f"Ticket ID      : {data.get('ticket_id')}")
        print(f"Model          : {data.get('model')}")
        print(f"Latency        : {data.get('latency_ms')} ms")
        print(f"Input Tokens   : {data.get('input_tokens')}")
        print(f"Output Tokens  : {data.get('output_tokens')}")
        print("-" * 60)
        print("SUMMARY:")
        print(f"  {data.get('summary')}")
        
        struct = data.get("structured_summary")
        if struct:
            print("-" * 60)
            print("STRUCTURED ANALYSIS:")
            print(f"  • Core Issue       : {struct.get('core_issue')}")
            print(f"  • Customer Intent  : {struct.get('customer_intent')}")
            print(f"  • Key Action Items : {struct.get('key_action_items')}")
            
        print("-" * 60)
        print("FULL RAW JSON RESPONSE:")
        print(json.dumps(data, indent=2))
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during summarization request: {e}")

if __name__ == "__main__":
    main()


