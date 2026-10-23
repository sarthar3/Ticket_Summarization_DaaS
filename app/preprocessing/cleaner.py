import re
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

class TicketData(BaseModel):
    ticket_id: str
    ticket_text: str
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    summary: Optional[str] = None
    sector: Optional[str] = "Cloud Infrastructure"
    domain: Optional[str] = "Cloud Infrastructure"
    intent: Optional[str] = "Multi-Service Outage / HTTP 503"
    issue_type: Optional[str] = "Multi-Service Outage / HTTP 503"
    category: Optional[str] = "Infrastructure Outage"
    urgency: Optional[str] = "Critical"
    priority: str = "P1 · Critical"
    priority_reason: Optional[str] = None
    customer_impact: Optional[str] = None
    recommended_actions: Optional[List[str]] = None
    issue_origin: Optional[str] = "COMPANY_SIDE"  # "CLIENT_SIDE" or "COMPANY_SIDE"
    client_guidance: Optional[Dict[str, Any]] = None
    escalation_details: Optional[Dict[str, Any]] = None

class TicketPreprocessor:
    """Preprocesses raw support tickets according to configurable limits and rules."""

    def __init__(self, max_input_tokens: int = 1024, max_output_tokens: int = 256, truncation_side: str = "right"):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.truncation_side = truncation_side

    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        # Replace multiple whitespaces/newlines with single spaces while preserving line breaks where helpful
        cleaned = text.strip()
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def classify_priority_details(self, ticket_text: str) -> Tuple[str, str]:
        """Automatically determines ticket priority and the underlying operational rationale using deterministic rules."""
        text_lower = ticket_text.lower()

        # Check for user impact scale (e.g. 500+ customers, hundreds of users, all users)
        impact_match = re.search(r"(?:more than|over|\b)(\d{2,}|hundreds?|thousands?)\+?\s*(?:customers?|users?|clients?|accounts?)", text_lower)
        high_user_impact = False
        if impact_match:
            val_str = impact_match.group(1)
            if val_str.isdigit() and int(val_str) >= 50:
                high_user_impact = True
            elif val_str in ["hundreds", "hundred", "thousands", "thousand"]:
                high_user_impact = True

        # Multi-service failure indicators
        service_keywords = ["portal", "mobile banking", "payment gateway", "internet banking", "api gateway", "checkout"]
        matched_services = [s for s in service_keywords if s in text_lower]
        is_multi_service = len(matched_services) >= 2 or ("services" in text_lower and any(w in text_lower for w in ["all stopped", "multiple", "down since", "unavailable"]))

        # Outage and error triggers
        has_503 = "503" in text_lower or "service unavailable" in text_lower
        has_outage_keywords = any(w in text_lower for w in [
            "stopped working", "widespread outage", "system down", "all users unable",
            "production crash", "major service outage", "services are down", "service outage",
            "completely down", "multi-service outage", "major outage"
        ])

        # P0 / P1 · Critical triggers: Outages, high scale impact, critical service failures, security breach, payroll deadlock
        if (is_multi_service and (has_503 or has_outage_keywords)) or (high_user_impact and (has_503 or has_outage_keywords or is_multi_service)):
            count_desc = impact_match.group(0).capitalize() if impact_match else "Multiple customers"
            return "P1 · Critical", f"Major widespread service outage: multiple core services returning 503 errors affecting {count_desc.lower()}"

        if has_503 and has_outage_keywords:
            return "P1 · Critical", "Widespread platform service unavailability (HTTP 503) affecting core business operations"

        if any(w in text_lower for w in ["not done by me", "unauthorized debit", "unauthorized transaction", "card stolen", "hacked", "fraudulent charge"]):
            return "P1 · Critical", "Reported unauthorized financial transaction / fraud alert requiring immediate account freeze"

        if any(w in text_lower for w in ["payroll", "cutoff window", "time-sensitive transfer"]):
            return "P1 · Critical", "Time-critical financial transfer / payroll access deadlock"

        if any(w in text_lower for w in ["security breach", "data leak", "zero-day", "ransomware", "unauthorized access"]):
            return "P1 · Critical", "Severe security incident / data confidentiality risk"

        if any(w in text_lower for w in ["system down", "widespread outage", "all users unable", "production crash"]):
            return "P1 · Critical", "Company-side core infrastructure outage affecting business operations"

        # P2 · High triggers: Single user authentication blockage, financial dispute, SLA breach
        if any(w in text_lower for w in ["unable to login", "cannot login", "invalid session", "password reset", "locked out"]):
            return "P2 · High", "Customer authentication blockage preventing service utilization"

        if any(w in text_lower for w in ["double billing", "charged twice", "deducted twice", "duplicate payment", "payment glitch"]):
            return "P2 · High", "Duplicate billing deduction requiring financial reconciliation"

        if any(w in text_lower for w in ["express", "carrier facility", "stuck in transit", "delayed for"]):
            return "P2 · High", "SLA breach on expedited delivery or delayed logistics"

        # Check if customer explicitly indicates absence of urgency (e.g. "no urgent", "not urgent")
        has_urgency_negation = any(neg in text_lower for neg in ["no urgent", "not urgent", "non-urgent", "not an emergency", "zero urgency"])
        if not has_urgency_negation and any(w in text_lower for w in ["urgent", "asap", "emergency", "error 500", "fatal error", "504 gateway timeout"]):
            return "P2 · High", "Elevated customer urgency with operational impact"

        # Routine transaction decline / insufficient balance check
        if any(w in text_lower for w in ["insufficient balance", "insufficient funds", "balance in account is", "getting failed as my balance", "balance is rs", "balance is inr"]):
            return "P3 · Medium", "Customer transaction decline: requested amount exceeds current available account balance"

        # P4 · Low triggers: Informational, feedback, documentation, general routine queries
        if any(w in text_lower for w in [
            "feedback", "suggestion", "feature request", "general question", "how do i",
            "documentation", "gdpr deletion", "branch hours", "ifsc code", "account statement format",
            "working hours", "where can i find", "requesting information", "annual percentage yield",
            "fixed deposit", "fixed deposits", "interest rate", "eligibility criteria",
            "routine inquiry", "routine support"
        ]) or has_urgency_negation:
            return "P4 / Low", "Standard customer support inquiry within standard SLA resolution window"

        # P3 · Medium default: Standard customer inquiry within SLA
        return "P3 · Medium", "Standard customer support inquiry within standard SLA resolution window"

    def classify_priority(self, ticket_text: str) -> str:
        """Automatically judges ticket priority based on complaint urgency indicators."""
        priority, _ = self.classify_priority_details(ticket_text)
        return priority

    def classify_domain_and_issue(self, ticket_text: str, sector: Optional[str] = None) -> Tuple[str, str, str, str, List[str]]:
        """Separates domain, issue type, urgency, customer impact, and recommended actions into distinct fields."""
        text_lower = ticket_text.lower()
        
        # Sector normalization
        if sector:
            sec_clean = sector.lower().strip()
            sector_mapping = {
                "banking & finance": "Banking & Financial Services",
                "banking": "Banking & Financial Services",
                "finance": "Banking & Financial Services",
                "healthcare": "Healthcare & Medical",
                "ecommerce": "E-Commerce & Retail",
                "e-commerce": "E-Commerce & Retail",
                "telecom": "Telecommunications",
                "it support": "IT & Tech Support",
                "cloud": "Cloud Infrastructure",
            }
            sector = sector_mapping.get(sec_clean, sector)

        priority_label, _ = self.classify_priority_details(ticket_text)
        urgency = "Critical" if "P1" in priority_label or "Critical" in priority_label else \
                  "High" if "P2" in priority_label or "High" in priority_label else \
                  "Medium" if "P3" in priority_label or "Medium" in priority_label else "Low"

        # 1. Outage / Availability / Multi-Service failure (Check first to avoid billing dispute trap)
        has_outage = any(w in text_lower for w in ["503", "service unavailable", "stopped working", "outage", "services are down", "down since"])
        service_mentions = [s for s in ["portal", "mobile banking", "payment gateway", "internet banking", "api"] if s in text_lower]

        if has_outage or len(service_mentions) >= 2:
            domain = sector if (sector and sector.lower() not in ["general", "general support"]) else "Cloud Infrastructure"
            issue_type = "Multi-Service Outage / HTTP 503" if ("503" in text_lower or len(service_mentions) >= 2) else "Service Outage & Availability"
            
            # Extract impact scale
            impact_match = re.search(r"(?:more than|over|\b)(\d{2,}|hundreds?|thousands?)\+?\s*(?:customers?|users?|clients?|accounts?)", text_lower)
            is_fin = any(k in (domain or "").lower() for k in ["bank", "finan", "payment", "money"])
            if impact_match:
                if is_fin:
                    customer_impact = f"More than {impact_match.group(1)} customers affected; banking access and transaction processing disrupted."
                else:
                    customer_impact = f"More than {impact_match.group(1)} users affected; application accessibility and service operations disrupted."
            else:
                if is_fin:
                    customer_impact = "Multiple customer services unavailable; transaction and login capabilities disrupted."
                else:
                    customer_impact = "Multiple users unable to access application; cloud service operations and connectivity disrupted."

            actions = [
                "Escalate to the incident response / SRE team.",
                "Investigate application health, load balancers, upstream dependencies, and recent deployments.",
                "Check logs, service metrics, and infrastructure health.",
                "Restore service and communicate incident updates."
            ]
            return domain, issue_type, urgency, customer_impact, actions

        # 2. Fraud / Unauthorized Transactions
        if any(w in text_lower for w in ["not done by me", "unauthorized", "fraud", "fraudulent", "scam", "stolen", "compromised", "hacked"]):
            domain = sector if (sector and sector.lower() not in ["general", "general support"]) else "Banking & Financial Security"
            issue_type = "Fraud Alert / Unauthorized Transaction"
            customer_impact = "Customer financial account potentially compromised; unauthorized debit reported."
            actions = [
                "Immediately freeze payment card and block online access credentials.",
                "Register formal dispute chargeback with payment settlement network.",
                "Escalate transaction telemetry to Fraud Risk and AML investigations unit."
            ]
            return domain, issue_type, urgency, customer_impact, actions

        # 3. Authentication & Access Failure
        if any(w in text_lower for w in ["unable to login", "cannot login", "password reset", "invalid session", "token", "locked out"]):
            domain = sector if (sector and sector.lower() not in ["general", "general support"]) else "Authentication & Client Access"
            issue_type = "Authentication & Access Failure"
            customer_impact = "Customer blocked from logging into account; cannot perform self-service actions."
            actions = [
                "Verify customer identity and session token validity.",
                "Invalidate stale session tokens in auth cache and trigger credential reset.",
                "Check authentication microservice error logs for widespread token expiration."
            ]
            return domain, issue_type, urgency, customer_impact, actions

        # 4. Billing & Payment Disputes (only when actual dispute/double billing, not when payment gateway is down)
        if any(w in text_lower for w in ["double billing", "charged twice", "deducted twice", "duplicate charge", "refund request", "incorrect charge"]):
            domain = sector if (sector and sector.lower() not in ["general", "general support"]) else "Billing & Payments"
            issue_type = "Billing & Payment Dispute"
            customer_impact = "Financial debit discrepancy reported; customer balance affected."
            actions = [
                "Reconcile payment transaction ID with payment gateway ledger.",
                "Initiate automated reversal for duplicate or unfulfilled debit.",
                "Issue confirmation receipt and updated statement to customer."
            ]
            return domain, issue_type, urgency, customer_impact, actions

        # 5. Transaction Failure & Insufficient Funds
        if any(w in text_lower for w in [
            "insufficient balance", "insufficient funds", "balance in account is", "account balance is",
            "balance is rs", "balance is inr", "balance is $", "getting failed as my balance",
            "transaction failed", "transaction is getting failed", "transaction declining", "payment failed",
            "transaction declined", "payment declined", "declined transaction", "failed transaction",
            "low balance", "not enough balance"
        ]):
            domain = sector if (sector and sector.lower() not in ["general", "general support"]) else "Banking & Financial Services"
            issue_type = "Transaction Failure / Insufficient Funds"
            customer_impact = "Customer transaction declined due to requested amount exceeding available account balance."
            actions = [
                "Verify customer ledger and available balance against requested transaction amount.",
                "Advise customer to deposit required funds or utilize an alternate payment instrument.",
                "Confirm if any active account holds, pending debits, or daily transfer limits are restricting balance."
            ]
            return domain, issue_type, urgency, customer_impact, actions

        # 5. Logistics & Delivery Delay
        if any(w in text_lower for w in ["tracking", "order", "delivery", "shipping", "in transit", "carrier"]):
            domain = sector if (sector and sector.lower() not in ["general", "general support"]) else "Logistics & Fulfillment"
            issue_type = "Logistics & Delivery Delay"
            customer_impact = "Customer order delivery delayed beyond estimated SLA window."
            actions = [
                "Track package location with carrier logistics API.",
                "Escalate warehouse dispatch status if uncollected.",
                "Update customer tracking link and waive express shipping fee if applicable."
            ]
            return domain, issue_type, urgency, customer_impact, actions

        # 6. API Infrastructure & Rate Limiting
        if any(w in text_lower for w in ["429", "504", "timeout", "rate limit", "api", "latency"]):
            domain = sector if (sector and sector.lower() not in ["general", "general support"]) else "Cloud Infrastructure"
            issue_type = "API Ingress & Rate Limiting"
            customer_impact = "Enterprise API requests rejected or timing out; downstream integration degraded."
            actions = [
                "Review ingress gateway rate limiter quotas and tenant tier configuration.",
                "Autoscale telemetry and proxy worker pods.",
                "Notify account manager of high traffic volume."
            ]
            return domain, issue_type, urgency, customer_impact, actions

        # 7. Telecom & Connectivity
        if any(w in text_lower for w in ["esim", "sim", "network", "fiber", "broadband"]):
            domain = sector if (sector and sector.lower() not in ["general", "general support"]) else "Telecommunications"
            issue_type = "Telecom & Connectivity Request"
            customer_impact = "Customer network service or device profile conversion pending."
            actions = [
                "Check carrier SM-DP+ provisioning server and SMTP dispatch queue.",
                "Trigger verified eSIM QR profile dispatch to customer email."
            ]
            return domain, issue_type, urgency, customer_impact, actions

        # 8. General Support / Informational
        domain = sector if (sector and sector.lower() not in ["general", "general support"]) else "General Support"
        issue_type = "General Customer Inquiry"
        customer_impact = "Routine customer inquiry with no active operational or service disruption."
        actions = [
            "Review customer account details and provide standard documentation / support response.",
            "Verify customer satisfaction and close inquiry."
        ]
        return domain, issue_type, urgency, customer_impact, actions

    def classify_origin_and_guidance(
        self,
        ticket_text: str,
        issue_type: str,
        domain: str,
        priority: str
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Classifies whether the issue is CLIENT_SIDE (requires client self-service guidance)
        or COMPANY_SIDE (requires engineering / company authority escalation via email)."""
        text_lower = ticket_text.lower()

        # Indicators where action must be taken by the client
        client_triggers = [
            "insufficient balance", "insufficient funds", "balance in account is", "balance is rs",
            "low balance", "not enough balance", "balance is inr", "balance is $",
            "how do i", "how to", "forgot password", "reset password", "change password",
            "forgot pin", "atm pin", "kyc", "document submission", "branch hours", "ifsc code",
            "routine inquiry", "information request", "daily limit reached", "exceeded limit"
        ]

        # Overriding company-side triggers (if system is broken or down, it's company-side)
        company_override = any(w in text_lower for w in [
            "503", "service unavailable", "stopped working", "widespread outage",
            "system down", "all users unable", "double billing", "charged twice",
            "deducted twice", "glitch", "crash", "invalid session", "webhook timeout",
            "unauthorized debit", "fraud", "stolen"
        ])

        if any(w in text_lower for w in client_triggers) and not company_override:
            issue_origin = "CLIENT_SIDE"
            if any(w in text_lower for w in ["insufficient balance", "insufficient funds", "balance in account is", "low balance", "not enough balance", "balance is rs", "balance is inr"]):
                summary = "Customer balance is below transaction amount. Client action required to deposit funds or switch payment instrument."
                steps = [
                    "1. Check your live account balance in your Mobile Banking app or NetBanking portal.",
                    "2. Deposit or transfer additional funds (via UPI, IMPS, or bank transfer) to cover the transaction value.",
                    "3. Ensure your daily outgoing transaction limit has not been exceeded.",
                    "4. Retry your transfer or select an alternate payment instrument (e.g. credit card or secondary account)."
                ]
                msg = (
                    "Dear Customer, your transaction could not be processed because the transfer amount "
                    "exceeds your currently available account balance. Please deposit required funds or select "
                    "an alternative payment instrument to complete your transaction successfully."
                )
            elif any(w in text_lower for w in ["password", "pin", "login credentials"]):
                summary = "Customer credential reset required. Follow self-service password reset steps."
                steps = [
                    "1. On the login screen, select 'Forgot Password / Reset PIN'.",
                    "2. Enter your registered mobile number or email ID to receive a secure OTP.",
                    "3. Enter the OTP and create a new secure alphanumeric password.",
                    "4. Sign back in using your new credentials."
                ]
                msg = (
                    "Dear Customer, you can securely reset your access credentials directly by clicking 'Forgot Password' "
                    "on the login screen. Follow the on-screen OTP verification steps to regain immediate account access."
                )
            else:
                summary = "Standard customer guidance inquiry. Refer to customer self-service guidelines."
                steps = [
                    "1. Access the self-service documentation portal for detailed instructions.",
                    "2. Verify that your registered KYC documents and contact details are up to date.",
                    "3. Reach out to our standard support channel if further clarification is needed."
                ]
                msg = (
                    "Dear Customer, thank you for reaching out. Please refer to our self-service portal for "
                    "step-by-step assistance with your inquiry, or contact our support team if you require further help."
                )

            client_guidance = {
                "summary": summary,
                "steps": steps,
                "message_template": msg
            }
            return issue_origin, client_guidance, None

        # Otherwise it is a COMPANY_SIDE issue requiring authority escalation
        issue_origin = "COMPANY_SIDE"
        try:
            from app.utils.officials_store import get_official_for_category
            official = get_official_for_category(domain or issue_type)
        except Exception:
            official = {
                "name": "Vikram Sharma",
                "designation": "Head of SRE & Cloud Infrastructure",
                "email": "vikram.sre@company.com",
                "department": "Cloud Infrastructure"
            }

        mail_subject = f"[COMPANY INCIDENT ESCALATION] {issue_type} - Action Required"
        mail_body = (
            f"Dear {official['name']} ({official['designation']}),\n\n"
            f"The TicketSummarizer DaaS engine has identified a company-side platform issue requiring your attention:\n\n"
            f"• Issue Domain: {domain}\n"
            f"• Issue Classification: {issue_type}\n"
            f"• Assigned Severity: {priority}\n"
            f"• Primary Incident Text: \"{ticket_text[:200]}...\"\n\n"
            f"Recommended Next Steps:\n"
            f"1. Review service telemetry and microservice health logs.\n"
            f"2. Initiate incident triage bridge with the engineering / operations team.\n"
            f"3. Acknowledge this escalation in the incident monitoring dashboard.\n\n"
            f"Regards,\nDaaS Incident Routing Engine"
        )

        escalation_details = {
            "escalation_required": True,
            "target_official": official,
            "email_subject": mail_subject,
            "email_body": mail_body,
            "department": official.get("department", domain)
        }
        return issue_origin, None, escalation_details

    def classify_intent(self, ticket_text: str) -> str:
        """Automatically infers customer intent / issue type category from complaint text."""
        _, issue_type, _, _, _ = self.classify_domain_and_issue(ticket_text)
        return issue_type

    def detect_distribution(
        self,
        ticket_text: str,
        sector: Optional[str] = None,
        domain: Optional[str] = None,
        priority_label: Optional[str] = None,
        issue_type: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Detects whether a ticket requires Teacher LLM (Groq 120B) or Student Model (Qwen2.5-1.5B).
        ROUTING SAFETY POLICY:
        1. High Severity / P1 Outages: Route to Teacher LLM for high-reliability synthesis & incident plan.
        2. Novel / Out-of-Distribution: Route to Teacher LLM for specialized domain reasoning.
        3. High Complexity / Multi-Issue Uncertainty: Route to Teacher LLM for multi-point decomposition.
        4. Standard In-Domain Low/Moderate Severity: Route to Student Model for rapid local inference.
        """
        text_lower = ticket_text.lower()
        effective_sector = (domain or sector or "").lower()

        # Rule 1: Severity Escalation (P1 Outages or severe security/fraud incidents)
        resolved_priority = priority_label
        if not resolved_priority:
            resolved_priority, _ = self.classify_priority_details(ticket_text)
        if "P1" in resolved_priority or "Critical" in resolved_priority:
            return True, (
                f"Critical Severity Safety Escalation ({resolved_priority}): High customer impact / widespread service outage "
                "routed to Teacher LLM (Groq 120B) for high-reliability synthesis and SRE-grade action plan."
            )

        # Rule 2: Novel/Unseen domain or sector triggers
        unseen_sectors = ["healthcare", "medical", "clinical", "biotech", "aerospace", "defense", "legal", "compliance", "iot hardware"]
        if any(s in effective_sector for s in unseen_sectors):
            return True, f"Novel Out-of-Distribution Sector detected ('{domain or sector}'): routed to Teacher LLM (Groq 120B) for high-precision reasoning."

        # Rule 3: High complexity / novel technical terminology
        unseen_technical_terms = [
            "distributed consensus", "raft cluster", "byzantine", "zero-day", "memory leak in kernel",
            "cross-border currency swap", "fx escrow", "hipaa", "fhir protocol", "pci-dss audit",
            "firmware bricked", "can-bus", "telematics ecu", "deadlock in worker thread", "quantum key"
        ]
        if any(term in text_lower for term in unseen_technical_terms):
            return True, "Complex novel technical incident / specialized terminology detected: routed to Teacher LLM (Groq 120B)."

        # Rule 4: Multi-issue complex tickets (long text with 3+ disjoint questions/complaints)
        words = ticket_text.split()
        if len(words) > 85 and any(w in text_lower for w in ["furthermore", "in addition to this", "secondly", "moreover", "on top of that"]):
            return True, "Multi-issue compound customer grievance detected: routed to Teacher LLM (Groq 120B) for multi-point decomposition."

        # Rule 5: Standard known in-domain patterns with moderate/low severity
        return False, f"Standard In-Domain Support Pattern ({resolved_priority}): routed to Student Model (Qwen2.5-1.5B) for rapid local inference."

    def generate_ticket_id(self) -> str:
        """Automatically generates a chronological sequential ticket ID."""
        try:
            from app.utils.history_store import get_next_ticket_id
            return get_next_ticket_id()
        except Exception:
            date_str = datetime.now().strftime("%Y%m%d")
            short_code = uuid.uuid4().hex[:5].upper()
            return f"TICK-{date_str}-{short_code}"

    def validate_and_normalize(self, raw_ticket: Dict[str, Any]) -> TicketData:
        """Validates ticket dictionary structure, generates missing ticket_id, and auto-classifies priority."""
        ticket_id = str(raw_ticket.get("ticket_id") or "").strip()
        if not ticket_id or ticket_id.upper() in ["AUTO", "NONE", "AUTO-ASSIGN"]:
            ticket_id = self.generate_ticket_id()

        ticket_text = raw_ticket.get("ticket_text")
        if ticket_text is None or str(ticket_text).strip() == "":
            raise ValueError(f"Ticket '{ticket_id}' missing or has empty 'ticket_text'.")

        cleaned_text = self.clean_text(str(ticket_text))
        summary = self.clean_text(str(raw_ticket["summary"])) if raw_ticket.get("summary") else None

        # Auto-classify domain, issue type, urgency, customer impact, actions
        sector_input = raw_ticket.get("sector")
        domain, issue_type, urgency, customer_impact, actions = self.classify_domain_and_issue(cleaned_text, sector_input)

        # Auto-classify priority and priority reason
        provided_priority = str(raw_ticket.get("priority") or "").strip()
        if provided_priority:
            priority = provided_priority
            priority_reason = f"Manual override: {provided_priority}"
        else:
            priority, priority_reason = self.classify_priority_details(cleaned_text)

        # Classify issue origin (CLIENT_SIDE vs COMPANY_SIDE) and generate guidance / escalation
        issue_origin, client_guidance, escalation_details = self.classify_origin_and_guidance(
            ticket_text=cleaned_text,
            issue_type=issue_type,
            domain=domain,
            priority=priority
        )

        customer_name = str(raw_ticket.get("customer_name") or "").strip() or None
        customer_email = str(raw_ticket.get("customer_email") or "").strip() or None
        customer_phone = str(raw_ticket.get("customer_phone") or "").strip() or None

        provided_intent = str(raw_ticket.get("intent") or "").strip()
        intent = provided_intent if (provided_intent and provided_intent.lower() != "inquiry") else issue_type

        return TicketData(
            ticket_id=ticket_id,
            ticket_text=cleaned_text,
            customer_name=customer_name,
            customer_email=customer_email,
            customer_phone=customer_phone,
            summary=summary,
            sector=domain,
            domain=domain,
            intent=intent,
            issue_type=issue_type,
            category=raw_ticket.get("category") or issue_type,
            urgency=urgency,
            priority=priority,
            priority_reason=priority_reason,
            customer_impact=customer_impact,
            recommended_actions=actions,
            issue_origin=issue_origin,
            client_guidance=client_guidance,
            escalation_details=escalation_details
        )
