"""Curated synthetic demo tickets for the MVP UI."""

from __future__ import annotations

DEMO_TICKETS = [
    {
        "title": "Security incident on the account portal",
        "subject": "Critical security incident affecting the account portal",
        "language": "en",
        "body": (
            "Multiple users report that the account portal is unavailable after suspicious login "
            "activity. We need urgent investigation, containment guidance, and an update on "
            "service restoration steps."
        ),
    },
    {
        "title": "Invoice mismatch after subscription renewal",
        "subject": "Unexpected invoice amount after renewal",
        "language": "en",
        "body": (
            "Our latest invoice shows duplicate charges after renewing the enterprise plan. "
            "Please review the billing details, explain the discrepancy, and advise on the refund process."
        ),
    },
    {
        "title": "Product issue after firmware update",
        "subject": "Firmware update causes device instability",
        "language": "en",
        "body": (
            "Since the latest firmware update, the device starts but quickly loses connectivity "
            "and becomes unreliable. Please investigate the root cause and provide a short-term fix recommendation."
        ),
    },
    {
        "title": "Return request for defective accessories",
        "subject": "Request to return defective accessories",
        "language": "en",
        "body": (
            "We received several accessories with visible defects and need instructions for the return "
            "and replacement process, including shipping labels and processing time."
        ),
    },
    {
        "title": "Employee portal access request",
        "subject": "Need access to the employee onboarding portal",
        "language": "en",
        "body": (
            "A new team member cannot access the onboarding portal and related HR documents. Please "
            "help us restore access and confirm the required permissions."
        ),
    },
    {
        "title": "Sicherheitsvorfall im Kundenportal",
        "subject": "Dringender Sicherheitsvorfall im Kundenportal",
        "language": "de",
        "body": (
            "Mehrere Nutzer melden verdächtige Anmeldeversuche und den Ausfall des Kundenportals. "
            "Bitte prüfen Sie den Vorfall umgehend, geben Sie Maßnahmen zur Eindämmung vor und "
            "informieren Sie über die nächsten Schritte zur Wiederherstellung."
        ),
    },
    {
        "title": "Falscher Rechnungsbetrag nach Vertragsverlängerung",
        "subject": "Rechnung nach Verlängerung scheint zu hoch",
        "language": "de",
        "body": (
            "Nach der Verlängerung unseres Enterprise-Vertrags wurde ein deutlich höherer Betrag als "
            "erwartet berechnet. Bitte prüfen Sie die Rechnung, erklären Sie die Abweichung und teilen "
            "Sie uns mit, wie eine Korrektur oder Erstattung erfolgt."
        ),
    },
    {
        "title": "Instabile Geräte nach Firmware-Update",
        "subject": "Firmware-Update verursacht Verbindungsabbrüche",
        "language": "de",
        "body": (
            "Seit dem letzten Firmware-Update verlieren mehrere Geräte nach kurzer Zeit die Verbindung "
            "und arbeiten unzuverlässig. Bitte analysieren Sie die Ursache und empfehlen Sie eine "
            "kurzfristige Lösung."
        ),
    },
    {
        "title": "Rücksendung defekter Zubehörteile",
        "subject": "Bitte um Rücksendung und Austausch defekter Zubehörteile",
        "language": "de",
        "body": (
            "Wir haben mehrere Zubehörteile mit sichtbaren Mängeln erhalten und benötigen eine "
            "Rücksendeanleitung. Bitte senden Sie uns Informationen zu Austausch, Versandlabel und "
            "Bearbeitungsdauer."
        ),
    },
    {
        "title": "Zugriff auf das HR-Onboarding-Portal",
        "subject": "Neuer Mitarbeiter hat keinen Zugriff auf das HR-Portal",
        "language": "de",
        "body": (
            "Ein neuer Mitarbeiter kann sich nicht im Onboarding-Portal anmelden und keine wichtigen "
            "HR-Dokumente einsehen. Bitte stellen Sie den Zugriff wieder her und bestätigen Sie die "
            "erforderlichen Berechtigungen."
        ),
    },
]


def get_demo_ticket(index: int = 0) -> tuple[int, dict[str, str]]:
    """Return one deterministic demo ticket by index."""
    if not DEMO_TICKETS:
        raise ValueError("No demo tickets are configured.")
    resolved_index = index % len(DEMO_TICKETS)
    return resolved_index, DEMO_TICKETS[resolved_index].copy()
