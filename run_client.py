import sys
import time
import argparse
import requests
import json

URL = "http://localhost:8000/api/v1/summarize"
HEALTH_URL = "http://localhost:8000/api/v1/health"
HISTORY_URL = "http://localhost:8000/api/v1/history"

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
        
        print(f"⏳ Server starting up / loading model... Retry {attempt}/{max_retries} (waiting {delay}s)")
        time.sleep(delay)
    return False

def get_user_input():
    print("=" * 65)
    print("      TICKET SUMMARIZATION DAAS - INTERACTIVE CLIENT      ")
    print("=" * 65)
    
    print("\nEnter Support Ticket Complaint Text:")
    ticket_text = input("> ").strip()
    
    while not ticket_text:
        print("⚠️ Ticket text cannot be empty! Please enter the ticket text:")
        ticket_text = input("> ").strip()
        
    ticket_id = input("\nEnter Ticket ID [optional - press Enter to auto-assign]: ").strip()
    sector = input("Enter Sector (e.g. Fintech, Healthcare, E-commerce) [optional]: ").strip()
    intent = input("Enter Intent (e.g. Technical Support, Billing) [optional]: ").strip()
    priority = input("Enter Priority (e.g. High, Medium, Low) [optional - auto-classified if empty]: ").strip()

    payload = {
        "ticket_text": ticket_text,
        "sector": sector if sector else None,
        "intent": intent if intent else None,
        "priority": priority if priority else None
    }
    if ticket_id:
        payload["ticket_id"] = ticket_id

    return payload

def print_summary_response(data):
    print("\n" + " CONCISE SUMMARY RESULT ".center(65, "="))
    print(f"Ticket ID      : {data.get('ticket_id')} (Auto-Assigned)" if "TICK-20" in str(data.get('ticket_id')) else f"Ticket ID      : {data.get('ticket_id')}")
    print(f"Priority       : {data.get('priority')} (Model Classified)")
    print(f"Model          : {data.get('model')}")
    print(f"Latency        : {data.get('latency_ms')} ms")
    print(f"Input Tokens   : {data.get('input_tokens')}")
    print(f"Output Tokens  : {data.get('output_tokens')}")
    print("-" * 65)
    print("SUMMARY:")
    print(f"  {data.get('summary')}")
    
    struct = data.get("structured_summary")
    if struct:
        print("-" * 65)
        print("STRUCTURED ANALYSIS:")
        print(f"  • Core Issue       : {struct.get('core_issue')}")
        print(f"  • Customer Intent  : {struct.get('customer_intent')}")
        print(f"  • Key Action Items : {struct.get('key_action_items')}")
        
    print("=" * 65 + "\n")

def fetch_and_print_history():
    try:
        res = requests.get(HISTORY_URL, timeout=10)
        res.raise_for_status()
        data = res.json()
        items = data.get("items", [])
        total = data.get("total_count", 0)

        print("\n" + f" TICKET SUMMARIZATION HISTORY ({total} Records) ".center(70, "="))
        if not items:
            print("No history records found.")
            print("=" * 70 + "\n")
            return

        for idx, item in enumerate(items, 1):
            print(f"\n[#{idx}] Ticket ID: {item.get('ticket_id')} | Priority: {item.get('priority')} | Time: {item.get('timestamp')}")
            print(f"    Complaint : {item.get('ticket_text')[:120]}...")
            print(f"    Summary   : {item.get('summary')}")
            struct = item.get("structured_summary")
            if struct:
                print(f"    Core Issue: {struct.get('core_issue')}")
                print(f"    Action    : {struct.get('key_action_items')}")
            print("-" * 70)
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Error fetching history: {e}\n")

def clear_remote_history():
    try:
        res = requests.delete(HISTORY_URL, timeout=10)
        res.raise_for_status()
        print("\n✅ History cleared successfully.\n")
    except Exception as e:
        print(f"\n❌ Error clearing history: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="Run Ticket Summarization Client with custom input")
    parser.add_argument("--ticket_text", type=str, help="Customer support ticket text")
    parser.add_argument("--ticket_id", type=str, help="Ticket ID (optional)")
    parser.add_argument("--sector", type=str, help="Sector")
    parser.add_argument("--intent", type=str, help="Intent")
    parser.add_argument("--priority", type=str, help="Priority (auto-classified if omitted)")
    parser.add_argument("--history", action="store_true", help="View summarization history")

    args = parser.parse_args()

    if not wait_for_server():
        print("\n❌ Error: Service took too long to respond at http://localhost:8000.")
        print("Please ensure uvicorn is running: python -m uvicorn app.main:app --reload --port 8000")
        return

    if args.history:
        fetch_and_print_history()
        return

    if args.ticket_text:
        payload = {
            "ticket_text": args.ticket_text,
            "sector": args.sector,
            "intent": args.intent,
            "priority": args.priority
        }
        if args.ticket_id:
            payload["ticket_id"] = args.ticket_id

        print("\nSending ticket summarization request...")
        try:
            response = requests.post(URL, json=payload, timeout=60)
            response.raise_for_status()
            print_summary_response(response.json())
        except Exception as e:
            print(f"\n❌ Error during summarization request: {e}")
        return

    # Interactive Loop
    while True:
        print("=" * 65)
        print("            TICKET SUMMARIZATION DaaS INTERFACE            ")
        print("=" * 65)
        print("  [1] Summarize a Support Ticket")
        print("  [2] View Summarization History")
        print("  [3] Clear History")
        print("  [4] Exit")
        choice = input("\nSelect an option (1-4): ").strip()

        if choice == "1":
            payload = get_user_input()
            print("\nSending ticket summarization request...")
            try:
                response = requests.post(URL, json=payload, timeout=60)
                response.raise_for_status()
                print_summary_response(response.json())
            except Exception as e:
                print(f"\n❌ Error during summarization request: {e}")
        elif choice == "2":
            fetch_and_print_history()
        elif choice == "3":
            clear_remote_history()
        elif choice == "4":
            print("Exiting client.")
            break
        else:
            print("Invalid option. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
