"""
AABS Control Tower - Claude AI Service

Real AI analysis using the Anthropic SDK with prompt caching.
Requires: ANTHROPIC_API_KEY environment variable.
"""

import os
import json
import anthropic
from datetime import datetime
from typing import Optional

# Stable system prompt — cached on first request, reused across all calls.
# Keep this frozen; dynamic context goes in the user message.
_SYSTEM_PROMPT = """You are the AABS Control Tower decision intelligence engine. \
You analyze supply chain data — orders, logistics corridors, market signals, and \
financial variance — to help operations managers act decisively.

Rules:
- Be concise: 2-4 sentences per analysis unless JSON is requested
- Lead with the risk or finding, follow with business impact ($, %, urgency), \
  end with one specific recommended action
- Use exact numbers from the data provided
- Never start with "I" or refer to yourself as an AI
- When returning JSON, respond with only valid JSON and no markdown fences"""


class ClaudeAIService:
    """
    Real AI service backed by Claude claude-opus-4-7.

    The system prompt is sent with cache_control so it is cached after the
    first request, cutting input token cost ~90% on all subsequent calls.
    """

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.available = True
                self.model = "claude-opus-4-7"
            except Exception:
                self.client = None
                self.available = False
                self.model = "disabled"
        else:
            self.client = None
            self.available = False
            self.model = "disabled"

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _call(self, user_message: str, max_tokens: int = 256) -> str:
        """Make a prompt-cached Claude API call. Returns plain text."""
        if not self.available:
            return "AI analysis unavailable. Set ANTHROPIC_API_KEY to enable."
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
            return next(
                (b.text for b in response.content if b.type == "text"), ""
            )
        except anthropic.AuthenticationError:
            self.available = False
            return "Invalid API key. Check ANTHROPIC_API_KEY."
        except anthropic.RateLimitError:
            return "Rate limit reached. Please wait before retrying."
        except anthropic.APIStatusError as e:
            return f"API error {e.status_code}: {e.message[:80]}"
        except Exception as e:
            return f"Analysis unavailable: {str(e)[:80]}"

    def _call_json(self, user_message: str, max_tokens: int = 400, fallback: dict = None) -> dict:
        """Call Claude expecting a JSON response. Falls back to `fallback` on parse failure."""
        raw = self._call(user_message, max_tokens=max_tokens)
        try:
            return json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            return fallback or {}

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def analyze_order_risk(self, order_data: dict) -> str:
        order_id = order_data.get("order_id", "N/A")
        customer = order_data.get("customer", "Unknown")
        value = order_data.get("value", 0)
        risk_score = order_data.get("risk_score", 0)
        line_items = order_data.get("line_items", 1)
        product_diversity = order_data.get("product_diversity", 1)
        country = order_data.get("country", "Unknown")

        prompt = (
            f"Analyze this high-risk order and provide a 2-3 sentence risk assessment.\n"
            f"Order #{order_id} | Customer: {customer} | Country: {country}\n"
            f"Value: ${value:,.0f} | Risk Score: {risk_score * 100:.0f}% "
            f"| Line Items: {line_items} | Product Types: {product_diversity}\n\n"
            f"Identify the primary risk driver, quantify the business impact, "
            f"and recommend one specific action."
        )
        return self._call(prompt, max_tokens=150)

    def analyze_corridor_impact(self, traffic_data: list, affected_orders: dict) -> str:
        congested = [t for t in traffic_data if t.get("level") in ["heavy", "severe"]]
        total_value = sum(v.get("value", 0) for v in affected_orders.values())
        total_count = sum(v.get("count", 0) for v in affected_orders.values())
        corridor_summary = "; ".join(
            f"{t['corridor']} ({t['delay_ratio']}x, {t['level']})" for t in congested
        )

        prompt = (
            f"Analyze logistics corridor disruptions and their supply chain impact.\n"
            f"Congested corridors: {corridor_summary}\n"
            f"Affected orders: {total_count} orders worth ${total_value:,.0f}\n\n"
            f"Provide a 2-3 sentence assessment of delivery risk and one mitigation recommendation."
        )
        return self._call(prompt, max_tokens=150)

    def generate_executive_brief(self, metrics: dict, alerts: list, ml_summary: dict = None) -> str:
        critical = [a for a in alerts if a.get("sev") == "CRITICAL"]
        at_risk = metrics.get("at_risk_value", 0)
        total = max(metrics.get("total_value", 1), 1)
        high_risk_count = metrics.get("high_risk_count", 0)

        prompt = (
            f"Generate a 3-sentence executive briefing for the supply chain control tower.\n"
            f"Pipeline: ${total / 1e6:.1f}M total | ${at_risk / 1e6:.2f}M at risk "
            f"({at_risk / total * 100:.0f}% exposure)\n"
            f"Critical alerts: {len(critical)} | High-risk orders: {high_risk_count}\n"
            f"Date: {datetime.now().strftime('%B %d, %Y %H:%M')}\n\n"
            f"Lead with overall risk posture, include the most urgent issue, "
            f"end with a recommended executive action."
        )
        return self._call(prompt, max_tokens=200)

    def generate_smart_recommendation(self, context: dict) -> str:
        prompt = (
            f"Based on this supply chain context, generate one specific actionable recommendation:\n"
            f"{str(context)[:500]}\n\n"
            f"Provide a single concrete recommendation with an expected outcome."
        )
        return self._call(prompt, max_tokens=120)

    def generate_consequence_analysis(self, issue: dict) -> str:
        prompt = (
            f"Analyze the business consequences of this supply chain issue if unaddressed:\n"
            f"{str(issue)[:400]}\n\n"
            f"Describe the 24-hour and 72-hour consequences in 2 sentences. "
            f"Include estimated financial impact."
        )
        return self._call(prompt, max_tokens=120)

    def generate_escalation_card(self, order_data: dict) -> dict:
        order_id = order_data.get("order_id", "N/A")
        value = order_data.get("value", 0)
        risk_score = order_data.get("risk_score", 0)
        customer = order_data.get("customer", "Unknown")

        prompt = (
            f"Generate escalation data for Order #{order_id} "
            f"(${value:,.0f}, {risk_score * 100:.0f}% risk, Customer: {customer}).\n"
            f"Return a JSON object with exactly these keys: "
            f"root_cause, action, deadline, owner, fallback, recovery_probability (integer 0-100)."
        )
        fallback = {
            "root_cause": f"High-value order with {risk_score * 100:.0f}% risk probability",
            "action": "Contact customer to confirm delivery requirements and verify address",
            "deadline": "Within 24 hours",
            "owner": "Logistics Team",
            "fallback": "Escalate to account manager if no response",
            "recovery_probability": 75,
        }
        data = self._call_json(prompt, max_tokens=200, fallback=fallback)
        data.update({
            "order_id": order_id,
            "customer": customer,
            "value": value,
            "risk_score": risk_score,
            "risk_level": "CRITICAL" if risk_score > 0.7 else "HIGH",
        })
        return data

    def generate_mitigation_playbook(self, issue_type: str, context: dict) -> dict:
        prompt = (
            f"Generate a mitigation playbook for: {issue_type}\n"
            f"Context: {str(context)[:300]}\n"
            f"Return JSON with exactly these keys: title, steps (list of 3-5 strings), "
            f"timeline, owner, success_metric."
        )
        return self._call_json(
            prompt,
            max_tokens=300,
            fallback={
                "title": f"{issue_type} Mitigation",
                "steps": ["Assess scope", "Notify stakeholders", "Implement fix", "Monitor outcome"],
                "timeline": "24-48 hours",
                "owner": "Operations Team",
                "success_metric": "Issue resolved without revenue impact",
            },
        )

    def generate_tradeoff_summary(self, option_a: dict, option_b: dict) -> dict:
        prompt = (
            f"Compare these two supply chain options and return a tradeoff analysis.\n"
            f"Option A: {str(option_a)[:200]}\n"
            f"Option B: {str(option_b)[:200]}\n"
            f"Return JSON with: recommendation, a_pros (list), a_cons (list), "
            f"b_pros (list), b_cons (list), confidence (integer 0-100)."
        )
        return self._call_json(
            prompt,
            max_tokens=250,
            fallback={
                "recommendation": "Evaluate based on current capacity and customer priority",
                "a_pros": ["Revenue protection"],
                "a_cons": ["Resource intensive"],
                "b_pros": ["Prevents cascading delays"],
                "b_cons": ["Delayed revenue impact"],
                "confidence": 70,
            },
        )

    def generate_daily_action_plan(self, metrics: dict, top_orders: list, alerts: list) -> dict:
        high_risk_count = metrics.get("high_risk_count", 0)
        at_risk = metrics.get("at_risk_value", 0)
        critical_count = len([a for a in alerts if a.get("sev") == "CRITICAL"])

        prompt = (
            f"Generate a daily action plan for supply chain operations.\n"
            f"High-risk orders: {high_risk_count} | At-risk value: ${at_risk / 1e6:.2f}M "
            f"| Critical alerts: {critical_count}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"Return JSON: {{date, generated_at, total_at_risk, orders_to_review, "
            f"actions: [{{action, priority, owner, deadline}}]}} with 3-5 actions."
        )
        return self._call_json(
            prompt,
            max_tokens=400,
            fallback={
                "date": datetime.now().strftime("%Y-%m-%d"),
                "generated_at": datetime.now().strftime("%H:%M"),
                "total_at_risk": at_risk,
                "orders_to_review": high_risk_count,
                "actions": [
                    {
                        "action": f"Review {high_risk_count} high-risk orders",
                        "priority": "CRITICAL",
                        "owner": "Logistics",
                        "deadline": "Today 12:00",
                    },
                    {
                        "action": f"Resolve {critical_count} critical alerts",
                        "priority": "HIGH",
                        "owner": "Operations",
                        "deadline": "Today 14:00",
                    },
                    {
                        "action": "Update executive dashboard",
                        "priority": "MEDIUM",
                        "owner": "Management",
                        "deadline": "Today 17:00",
                    },
                ],
            },
        )
