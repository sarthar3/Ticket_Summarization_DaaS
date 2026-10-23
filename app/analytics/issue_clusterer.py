import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AffectedTicket(BaseModel):
    ticket_id: str
    timestamp: str
    formatted_time: str
    customer_name: str
    customer_email: str
    customer_phone: str
    priority: str
    core_issue: str
    summary: str
    raw_query: str

class CommonIssueCluster(BaseModel):
    cluster_id: str
    issue_title: str
    category: str
    affected_count: int
    affected_pct: float
    severity: str  # "CRITICAL_OUTAGE", "HIGH_IMPACT", "MODERATE"
    is_company_wide_issue: bool  # True if >= 2 users
    issue_origin: str = "COMPANY_SIDE"  # "CLIENT_SIDE" or "COMPANY_SIDE"
    common_issue_summary: str
    company_root_cause_diagnosis: str
    recommended_action: str
    unified_solution: str = ""
    escalation_official: Optional[Dict[str, Any]] = None
    escalation_email_subject: Optional[str] = None
    escalation_email_body: Optional[str] = None
    client_guidance: Optional[Dict[str, Any]] = None
    resolution_applied: bool = False
    resolution_timestamp: Optional[str] = None
    affected_tickets: List[AffectedTicket]

class CompanyIssuesReport(BaseModel):
    total_tickets_analyzed: int
    total_customers_affected: int
    unique_issue_clusters_count: int
    company_outages_detected: int
    has_active_outage: bool
    has_unverified_outage_report: bool = False
    unverified_outage_count: int = 0
    company_status: str = "HEALTHY"
    executive_brief: str
    generated_at: str
    clusters: List[CommonIssueCluster]
    chronological_queue: List[AffectedTicket]


DOMAIN_ISSUE_PROFILES = [
    {
        "id_prefix": "CLOUD-OUTAGE",
        "title": "Multi-Service Cloud & Banking Infrastructure Outage (HTTP 503)",
        "category": "Cloud Infrastructure",
        "keywords": [
            "503", "http 503", "service unavailable", "stopped working", "portal",
            "payment gateway", "mobile banking", "multi-service", "outage", "services are down",
            "down since", "500 customers", "internet banking", "business operations"
        ],
        "root_cause": (
            "Upstream microservices (internet banking portal, mobile banking application, and payment gateway) "
            "are failing health probes and returning HTTP 503 Service Unavailable errors across services. "
            "Ingress application load balancers are saturating backend connection pools under peak customer traffic."
        ),
        "action": (
            "1. Immediately escalate to SRE / Incident Response bridge.\n"
            "2. Investigate application health, load balancers, upstream dependencies, and recent deployments.\n"
            "3. Check logs, service metrics, and infrastructure health.\n"
            "4. Restore service and communicate incident updates."
        )
    },
    {
        "id_prefix": "AUTH-OUTAGE",
        "title": "Mobile Banking & App Session Invalidation Failure",
        "category": "Authentication & Mobile Client",
        "keywords": [
            "login", "sign in", "unable to login", "cannot login", "password reset",
            "session", "invalid session", "token", "auth", "ios 17", "v4.2", "update",
            "payroll transfer", "locked out", "credentials"
        ],
        "root_cause": (
            "Authentication gateway microservice is invalidating JWT session tokens prematurely "
            "following the recent v4.2 client release. The recurrent failures across independent "
            "customers confirm a company-side authorization service regression."
        ),
        "action": (
            "1. Roll back authentication gateway patch to v4.1.9 immediately.\n"
            "2. Flush stale session key blacklist in Redis cache.\n"
            "3. Broadcast high-priority incident status to customer support and mobile leads."
        )
    },
    {
        "id_prefix": "PAYMENT-FAIL",
        "title": "Payment Gateway Timeout & Double-Deduction Anomaly",
        "category": "Billing & Payment Processing",
        "keywords": [
            "payment", "charged", "double billing", "deducted", "gateway timeout",
            "refund", "checkout", "invoice", "bank statement", "double charge",
            "money deducted", "transaction failed", "unpaid"
        ],
        "root_cause": (
            "Upstream bank payment gateway webhook callbacks are experiencing network timeouts "
            "during peak checkout. The customer's bank executes the debit, but the internal billing "
            "service fails to receive the completion callback, leaving the order in an unconfirmed state."
        ),
        "action": (
            "1. Replay dropped webhook transaction events via the payment partner fallback queue.\n"
            "2. Trigger automated ledger reconciliation for all orders in the last 4 hours.\n"
            "3. Issue automatic refund/credit confirmations to affected customers within 60 minutes."
        )
    },
    {
        "id_prefix": "SECURITY-FRAUD",
        "title": "Unauthorized Debit & Suspected Account Compromise Alert",
        "category": "Fraud & Account Security",
        "keywords": [
            "not done by me", "unauthorized", "fraud", "fraudulent", "did not authorize",
            "stolen card", "card compromised", "scam", "didn't do this", "unknown debit",
            "unrecognized charge", "hacked account"
        ],
        "root_cause": (
            "Customer reports unauthorized financial debit occurring without account holder authorization, "
            "indicating possible payment instrument compromise, credential harvesting, or fraudulent switch routing."
        ),
        "action": (
            "1. Immediately place a hotlist/freeze on customer card and online banking credentials.\n"
            "2. Register formal dispute chargeback with issuing/acquiring payment switch.\n"
            "3. Escalate transaction details and IP telemetry to Fraud Risk & AML team."
        )
    },
    {
        "id_prefix": "LOGISTICS-DELAY",
        "title": "Warehouse Dispatch Stagnation & Carrier Tracking Freeze",
        "category": "Logistics & Fulfillment",
        "keywords": [
            "order", "tracking", "carrier", "in transit", "stuck", "shipping",
            "backorder", "sku", "warehouse", "delivery", "carrier origin facility",
            "refund of shipping", "express 2-day"
        ],
        "root_cause": (
            "ERP-to-Carrier inventory synchronization queue backlog. High-priority packages were "
            "manifested but remained uncollected at the primary regional warehouse origin facility."
        ),
        "action": (
            "1. Force manual sync on the warehouse ERP dispatch queue.\n"
            "2. Escalate logistics priority ticket with tier-1 carrier operations.\n"
            "3. Automatically waive and refund express shipping charges for affected accounts."
        )
    },
    {
        "id_prefix": "INFRA-THROTTLE",
        "title": "Cloud API Ingress 429 Throttling & 504 Gateway Timeouts",
        "category": "Cloud Infrastructure",
        "keywords": [
            "429", "504", "rate limit", "too many requests", "gateway timeout",
            "telemetry", "endpoint", "sla", "requests/minute", "spike", "storage microservice"
        ],
        "root_cause": (
            "Ingress API gateway rate limiting rules misconfigured for enterprise Tier 3 SLA accounts, "
            "triggering premature 429 throttling below contractual capacity limits."
        ),
        "action": (
            "1. Dynamically raise ingress rate limiter ceiling to 2,000 RPM for Tier 3 tenants.\n"
            "2. Horizontal autoscaling of API telemetry proxy pods from 4 to 12 replicas."
        )
    },
    {
        "id_prefix": "HEALTHCARE-EXPORT",
        "title": "EMR Physician Dashboard Export Timeout",
        "category": "Healthcare & Medical Systems",
        "keywords": [
            "emr", "portal", "physician", "lab results", "pdf", "storage microservice",
            "gateway timeout", "patient"
        ],
        "root_cause": (
            "Storage microservice thread exhaustion when aggregating large multi-page patient "
            "laboratory pathology PDFs."
        ),
        "action": (
            "1. Shift PDF generation to asynchronous background Celery worker queue.\n"
            "2. Direct physician portal to deliver instant pre-signed download links when rendered."
        )
    },
    {
        "id_prefix": "TELECOM-ESIM",
        "title": "eSIM Automated Provisioning & QR Dispatch Queue Delay",
        "category": "Telecommunications",
        "keywords": [
            "esim", "sim", "fiber", "connection drops", "physical sim", "qr profile",
            "activation email", "lost device"
        ],
        "root_cause": (
            "Provisioning server SMTP queue rate limit preventing automated delivery of eSIM QR activation profiles."
        ),
        "action": (
            "1. Clear provisioning SMTP queue backlog and verify carrier SM-DP+ server status.\n"
            "2. Trigger manual one-time QR dispatch to customer's verified email address."
        )
    },
    {
        "id_prefix": "SECURITY-BRUTEFORCE",
        "title": "Distributed SSO Credential Stuffing & Brute-Force Alert",
        "category": "Cybersecurity & SOC",
        "keywords": [
            "brute-force", "credential stuffing", "unauthorized", "soc compliance",
            "sso admin", "cloudflare", "waf", "subnet", "security breach"
        ],
        "root_cause": (
            "Coordinated botnet credential stuffing targeting administrative SSO login endpoints."
        ),
        "action": (
            "1. Enforce instantaneous IP CIDR blocklist via Cloudflare WAF.\n"
            "2. Require mandatory hardware/authenticator app MFA for all admin accounts."
        )
    }
]

_applied_resolutions: Dict[str, Dict[str, Any]] = {}

def record_cluster_solution(cluster_id: str, solution_text: Optional[str] = None) -> Dict[str, Any]:
    """Records that a unified single solution has been applied to all tickets in a cluster."""
    now_str = datetime.now().strftime("%I:%M %p, %b %d, %Y")
    sol = solution_text or "Unified operational resolution applied across all affected tickets."
    _applied_resolutions[cluster_id] = {
        "solution": sol,
        "timestamp": now_str
    }
    return _applied_resolutions[cluster_id]

class IssueClusterer:
    """Clustering engine that aggregates customer tickets, isolates recurring company-side issues,
    and produces high-level executive root-cause summaries."""

    def __init__(self):
        self.profiles = DOMAIN_ISSUE_PROFILES

    def _normalize_tokens(self, text: str) -> set:
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        words = clean.split()
        stopwords = {
            "the", "and", "to", "of", "a", "in", "is", "that", "for", "i", "it",
            "not", "on", "with", "he", "as", "you", "at", "this", "but", "his",
            "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
            "my", "one", "all", "would", "there", "their", "what", "so", "up",
            "out", "if", "about", "who", "get", "which", "go", "me", "customer",
            "reported", "user", "reported", "needs", "support", "ticket", "issue"
        }
        return {w for w in words if len(w) > 2 and w not in stopwords}

    def _format_time_str(self, timestamp_val: Any) -> str:
        """Produces a clean human-readable time like '10:00 AM, Oct 15, 2026'."""
        if not timestamp_val:
            return datetime.now().strftime("%I:%M %p, %b %d, %Y")
        try:
            val = str(timestamp_val).replace("Z", "+00:00")
            dt = datetime.fromisoformat(val)
            return dt.strftime("%I:%M %p, %b %d, %Y")
        except Exception:
            return str(timestamp_val)

    def _extract_core_issue(self, ticket: Dict[str, Any]) -> str:
        structured = ticket.get("structured_summary") or {}
        if isinstance(structured, dict) and structured.get("core_issue"):
            return structured["core_issue"]
        summary = ticket.get("summary") or ""
        if summary:
            return summary.split(".")[0].strip() + "."
        text = ticket.get("ticket_text") or ""
        return (text[:120].strip() + "...") if len(text) > 120 else text

    def cluster_tickets(self, raw_tickets: List[Dict[str, Any]]) -> CompanyIssuesReport:
        if not raw_tickets:
            now_str = datetime.now().strftime("%I:%M %p, %b %d, %Y")
            return CompanyIssuesReport(
                total_tickets_analyzed=0,
                total_customers_affected=0,
                unique_issue_clusters_count=0,
                company_outages_detected=0,
                has_active_outage=False,
                has_unverified_outage_report=False,
                unverified_outage_count=0,
                company_status="HEALTHY",
                executive_brief="No customer tickets currently available in system history.",
                generated_at=now_str,
                clusters=[],
                chronological_queue=[]
            )

        # 1. Transform raw ticket items into standard AffectedTicket objects
        transformed_tickets: List[AffectedTicket] = []
        for item in raw_tickets:
            t_id = str(item.get("ticket_id") or "TCK-UNKNOWN")
            ts = str(item.get("timestamp") or "")
            formatted_time = item.get("created_at_formatted") or self._format_time_str(ts)
            c_name = str(item.get("customer_name") or "Valued Customer").strip()
            c_email = str(item.get("customer_email") or "customer@example.com").strip()
            c_phone = str(item.get("customer_phone") or "N/A").strip()
            priority = str(item.get("priority") or "Medium")
            core_issue = self._extract_core_issue(item)
            summary = str(item.get("summary") or core_issue)
            raw_text = str(item.get("ticket_text") or "")

            transformed_tickets.append(AffectedTicket(
                ticket_id=t_id,
                timestamp=ts,
                formatted_time=formatted_time,
                customer_name=c_name,
                customer_email=c_email,
                customer_phone=c_phone,
                priority=priority,
                core_issue=core_issue,
                summary=summary,
                raw_query=raw_text
            ))

        # 2. Match tickets into predefined domain profiles or generic clusters
        matched_groups: Dict[str, List[AffectedTicket]] = {p["id_prefix"]: [] for p in self.profiles}
        unmatched_tickets: List[AffectedTicket] = []

        for ticket in transformed_tickets:
            searchable_text = f"{ticket.raw_query} {ticket.core_issue} {ticket.summary}".lower()
            matched_profile = None
            highest_score = 0

            for profile in self.profiles:
                score = 0
                for kw in profile["keywords"]:
                    if kw.lower() in searchable_text:
                        score += 2 if len(kw.split()) > 1 else 1

                if score > highest_score and score >= 2:
                    highest_score = score
                    matched_profile = profile

            if matched_profile:
                matched_groups[matched_profile["id_prefix"]].append(ticket)
            else:
                unmatched_tickets.append(ticket)

        # 3. Dynamic clustering for unmatched tickets using token similarity
        dynamic_clusters: List[List[AffectedTicket]] = []
        for ticket in unmatched_tickets:
            t_tokens = self._normalize_tokens(f"{ticket.raw_query} {ticket.summary}")
            placed = False
            for group in dynamic_clusters:
                # Compare with first ticket in group
                leader_tokens = self._normalize_tokens(f"{group[0].raw_query} {group[0].summary}")
                inter = len(t_tokens.intersection(leader_tokens))
                union = len(t_tokens.union(leader_tokens))
                jaccard = (inter / union) if union > 0 else 0
                if jaccard >= 0.28:
                    group.append(ticket)
                    placed = True
                    break
            if not placed:
                dynamic_clusters.append([ticket])

        # 4. Construct CommonIssueCluster objects
        total_tickets = len(transformed_tickets)
        clusters: List[CommonIssueCluster] = []

        # Process predefined clusters
        cluster_seq = 1
        for profile in self.profiles:
            group = matched_groups[profile["id_prefix"]]
            if not group:
                continue

            count = len(group)
            pct = round((count / total_tickets) * 100.0, 1)
            is_company_issue = count >= 2

            if count >= 3:
                severity = "CRITICAL_OUTAGE"
            elif count == 2:
                severity = "HIGH_IMPACT"
            else:
                severity = "MODERATE"

            # Synthesize customer list for summary
            user_names = [t.customer_name for t in group[:4]]
            user_list_str = ", ".join(user_names) + (f" and {count - 4} others" if count > 4 else "")
            
            if is_company_issue:
                common_summary = (
                    f"⚠️ COMMON COMPANY ISSUE IDENTIFIED: {count} customers ({user_list_str}) have independently "
                    f"reported identical incidents regarding '{profile['title']}'. "
                    f"This recurrence confirms a company-side platform defect rather than user error."
                )
            else:
                common_summary = (
                    f"Single isolated customer inquiry from {user_names[0]} regarding '{profile['title']}'."
                )

            from app.utils.officials_store import get_official_for_category
            official = get_official_for_category(profile["category"])
            cid = f"CLUSTER-0{cluster_seq}"

            # Escalation email template
            ticket_id_list = ", ".join(t.ticket_id for t in group[:6])
            email_subject = f"[COMPANY OUTAGE ESCALATION] {profile['title']} — {count} Customer(s) Impacted"
            email_body = (
                f"Dear {official['name']} ({official['designation']}),\n\n"
                f"The TicketSummarizer DaaS Incident Intelligence engine has identified a company-side platform issue:\n\n"
                f"• Incident Title: {profile['title']}\n"
                f"• Category / Department: {profile['category']}\n"
                f"• Impact Scale: {count} customer(s) affected ({pct}% of ticket volume)\n"
                f"• Associated Ticket IDs: {ticket_id_list}\n"
                f"• Root Cause Diagnosis: {profile['root_cause']}\n\n"
                f"Unified SRE Action Plan:\n{profile['action']}\n\n"
                f"Please acknowledge receipt and coordinate resolution across the operational team.\n\n"
                f"Regards,\nDaaS Outage Intelligence Bridge"
            )

            is_res = cid in _applied_resolutions
            res_meta = _applied_resolutions.get(cid, {})

            clusters.append(CommonIssueCluster(
                cluster_id=cid,
                issue_title=profile["title"],
                category=profile["category"],
                affected_count=count,
                affected_pct=pct,
                severity=severity,
                is_company_wide_issue=is_company_issue,
                issue_origin="COMPANY_SIDE",
                common_issue_summary=common_summary,
                company_root_cause_diagnosis=profile["root_cause"],
                recommended_action=profile["action"],
                unified_solution=res_meta.get("solution", profile["action"]),
                escalation_official=official,
                escalation_email_subject=email_subject,
                escalation_email_body=email_body,
                resolution_applied=is_res,
                resolution_timestamp=res_meta.get("timestamp"),
                affected_tickets=group
            ))
            cluster_seq += 1

        # Process dynamic clusters
        for dyn_group in dynamic_clusters:
            count = len(dyn_group)
            pct = round((count / total_tickets) * 100.0, 1)
            is_company_issue = count >= 2
            severity = "CRITICAL_OUTAGE" if count >= 3 else "HIGH_IMPACT" if count == 2 else "MODERATE"

            first_t = dyn_group[0]
            title = f"Customer Query: {first_t.core_issue[:55]}..."
            user_names = [t.customer_name for t in dyn_group[:4]]
            user_list_str = ", ".join(user_names)

            # Check if this dynamic cluster is client-side (e.g. balance, password)
            combined_text = " ".join(t.raw_query for t in dyn_group).lower()
            is_client_side = any(k in combined_text for k in [
                "insufficient balance", "insufficient funds", "balance in account is", "low balance",
                "not enough balance", "balance is rs", "how do i", "forgot password", "reset password",
                "kyc", "routine inquiry"
            ]) and not any(k in combined_text for k in ["503", "outage", "stopped working", "system down", "crash"])

            cid = f"CLUSTER-0{cluster_seq}"
            from app.utils.officials_store import get_official_for_category

            if is_client_side:
                issue_origin = "CLIENT_SIDE"
                diag = "Routine customer self-service inquiry / account balance threshold."
                act = "Transmit automated step-by-step guidance instructions to all affected customers."
                unified_sol = "1. Transmit standard self-service instructions to customer.\n2. Advise balance top-up or secondary payment instrument.\n3. Close inquiries upon verification."
                client_guidance = {
                    "summary": "Customer action required to deposit funds or perform self-service credentials update.",
                    "steps": [
                        "1. Verify account ledger and current available balance.",
                        "2. Deposit required funds via UPI / NetBanking if balance is lower than transfer amount.",
                        "3. Retry transaction or choose alternate payment method."
                    ]
                }
                official = None
                email_subject = None
                email_body = None
            else:
                issue_origin = "COMPANY_SIDE"
                client_guidance = None
                official = get_official_for_category("General")
                ticket_id_list = ", ".join(t.ticket_id for t in dyn_group[:6])
                email_subject = f"[COMPANY ISSUE ESCALATION] {first_t.core_issue[:40]} — {count} Customer(s)"
                email_body = (
                    f"Dear {official['name']} ({official['designation']}),\n\n"
                    f"A recurring company-side issue has been flagged:\n"
                    f"• Problem: {first_t.core_issue}\n"
                    f"• Affected Tickets: {ticket_id_list}\n"
                    f"• Impact Count: {count} customer(s)\n\n"
                    f"Please inspect telemetry logs and assign an operational owner.\n\n"
                    f"Regards,\nDaaS Intelligence Bridge"
                )
                if is_company_issue:
                    common_summary = (
                        f"⚠️ RECURRING SYSTEM ISSUE: {count} customers ({user_list_str}) reported similar complaints "
                        f"concerning: '{first_t.core_issue}'. Company operational review recommended."
                    )
                    diag = f"Multiple customer tickets share high semantic similarity regarding '{first_t.core_issue}'."
                    act = "Investigate related backend services and customer telemetry logs."
                    unified_sol = "1. Initiate service diagnostics for reported subsystem.\n2. Review proxy logs and error rate metrics.\n3. Dispatch incident resolution notice to affected tickets."
                else:
                    common_summary = f"Customer query reported by {user_names[0]}: {first_t.core_issue}"
                    diag = "Isolated user inquiry."
                    act = "Standard support ticket resolution."
                    unified_sol = "Review customer query details and execute standard ticket resolution."

            is_res = cid in _applied_resolutions
            res_meta = _applied_resolutions.get(cid, {})

            clusters.append(CommonIssueCluster(
                cluster_id=cid,
                issue_title=title,
                category="General Service Inquiry" if not is_client_side else "Client Account Management",
                affected_count=count,
                affected_pct=pct,
                severity=severity,
                is_company_wide_issue=is_company_issue,
                issue_origin=issue_origin,
                common_issue_summary=common_summary,
                company_root_cause_diagnosis=diag,
                recommended_action=act,
                unified_solution=res_meta.get("solution", unified_sol),
                escalation_official=official,
                escalation_email_subject=email_subject,
                escalation_email_body=email_body,
                client_guidance=client_guidance,
                resolution_applied=is_res,
                resolution_timestamp=res_meta.get("timestamp"),
                affected_tickets=dyn_group
            ))
            cluster_seq += 1

        # Sort clusters: Outages / multi-user issues first, then by count descending
        clusters.sort(key=lambda c: (1 if c.is_company_wide_issue else 0, c.affected_count), reverse=True)

        # 5. Outage stats and Executive Brief
        outage_count = sum(1 for c in clusters if c.is_company_wide_issue)
        has_outage = outage_count > 0

        # Unverified single-ticket outage reports (P1 Critical or availability outages without multi-ticket correlation yet)
        unverified_outage_clusters = [
            c for c in clusters if not c.is_company_wide_issue and (
                any("P1" in t.priority or "Critical" in t.priority for t in c.affected_tickets) or
                "outage" in c.issue_title.lower() or "503" in c.issue_title.lower() or "http" in c.issue_title.lower()
            )
        ]
        has_unverified = len(unverified_outage_clusters) > 0
        unverified_count = len(unverified_outage_clusters)

        if has_outage:
            company_status = "CRITICAL_OUTAGE"
            top_cluster = clusters[0]
            brief = (
                f"🚨 SYSTEMIC COMPANY-SIDE OUTAGE ALERT: {outage_count} widespread issue cluster(s) detected affecting "
                f"{sum(c.affected_count for c in clusters if c.is_company_wide_issue)} customers. "
                f"Top recurring issue is '{top_cluster.issue_title}' affecting {top_cluster.affected_count} users "
                f"({top_cluster.affected_pct}% of total tickets). Immediate engineering mitigation required."
            )
        elif has_unverified:
            company_status = "UNVERIFIED_OUTAGE_REPORT"
            unv = unverified_outage_clusters[0]
            brief = (
                f"⚠️ UNVERIFIED OUTAGE REPORT UNDER INVESTIGATION: A customer ticket has reported a high-severity incident "
                f"('{unv.issue_title}'). SRE monitoring active; awaiting corroborating customer reports or telemetry confirmation "
                f"before declaring full systemic company outage."
            )
        else:
            company_status = "HEALTHY"
            brief = (
                f"All {total_tickets} processed ticket(s) are operating within normal parameters with no common systemic company-side outages detected."
            )

        # Sort chronological queue: chronological order (earliest to latest or latest to earliest)
        chronological_queue = list(transformed_tickets)

        now_str = datetime.now().strftime("%I:%M %p, %b %d, %Y")
        return CompanyIssuesReport(
            total_tickets_analyzed=total_tickets,
            total_customers_affected=total_tickets,
            unique_issue_clusters_count=len(clusters),
            company_outages_detected=outage_count,
            has_active_outage=has_outage,
            has_unverified_outage_report=has_unverified,
            unverified_outage_count=unverified_count,
            company_status=company_status,
            executive_brief=brief,
            generated_at=now_str,
            clusters=clusters,
            chronological_queue=chronological_queue
        )

# Global singleton
_clusterer_instance: Optional[IssueClusterer] = None

def get_issue_clusterer() -> IssueClusterer:
    global _clusterer_instance
    if _clusterer_instance is None:
        _clusterer_instance = IssueClusterer()
    return _clusterer_instance
