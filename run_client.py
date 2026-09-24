"""
Standalone client runner to invoke the Ticket Summarization DaaS FastAPI endpoint
and display clean, formatted structured output in the terminal.
"""
import requests
import json

URL = "http://localhost:8000/api/v1/summarize"

payload = {
    "ticket_id": "TICK-1001",
    "ticket_text": "User unable to login to mobile banking app following v4.2 update on iOS 17.4. Receives Error Code ERR-902 credential rejection during 2FA prompt.",
    "sector": "Fintech",
    "intent": "Technical Support",
    "priority": "High"
}

def main():
    print("=" * 60)
    print("Sending ticket summarization request...")
    print("=" * 60)

    try:
        response = requests.post(URL, json=payload, timeout=30)
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

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to FastAPI server at http://localhost:8000.")
        print("Please start the server first by running:")
        print("   python -m uvicorn app.main:app --reload --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
