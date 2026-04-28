from typing import Optional, List
from agents.agent import Agent
from agents.job_listings import JobListing, JobOpportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from openai import OpenAI
import json


class AutonomousPlanningAgent(Agent):
    """
    Full agent loop with tool calling: an LLM autonomously decides which
    tools to call (scan, estimate, notify) to find valuation opportunities.
    """

    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "gpt-4o"

    def __init__(self, collection):
        self.log("Autonomous Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.openai = OpenAI()
        self.memory = None
        self.opportunity = None
        self.log("Autonomous Planning Agent is ready")

    def scan_for_job_listings(self) -> str:
        self.log("Autonomous Planning Agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No startup profiles found"

    def estimate_market_salary(self, description: str) -> str:
        self.log("Autonomous Planning Agent is estimating valuation via Ensemble")
        estimate = self.ensemble.estimate(description)
        return f"The estimated valuation for '{description[:80]}...' is ${estimate:,.0f}"

    def notify_user_of_opportunity(
        self, description: str, offered_salary: float, estimated_salary: float, url: str
    ) -> str:
        if self.opportunity:
            self.log("Already notified user this run; ignoring duplicate")
        else:
            self.log("Autonomous Planning Agent is notifying user")
            self.messenger.notify(description, offered_salary, estimated_salary, url)
            listing = JobListing(
                job_title=description[:100],
                company="",
                salary=offered_salary,
                location="",
                description=description,
                url=url,
            )
            premium = offered_salary - estimated_salary
            self.opportunity = JobOpportunity(
                listing=listing, estimate=estimated_salary, premium=premium
            )
        return "Notification sent"

    scan_function = {
        "name": "scan_for_job_listings",
        "description": "Scans startup/news RSS feeds and returns promising startup profiles with valuation signals",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_market_salary",
        "description": "Given a startup profile, estimate a likely valuation",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The startup profile/description to estimate valuation for",
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_opportunity",
        "description": "Send the user a push notification about the single best valuation opportunity; only call once",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The startup profile scraped from the internet",
                },
                "offered_salary": {
                    "type": "number",
                    "description": "The reported valuation of this startup profile",
                },
                "estimated_salary": {
                    "type": "number",
                    "description": "The estimated valuation for this startup",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of the startup profile/article",
                },
            },
            "required": ["description", "offered_salary", "estimated_salary", "url"],
            "additionalProperties": False,
        },
    }

    def get_tools(self):
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tool_call(self, message):
        mapping = {
            "scan_for_job_listings": self.scan_for_job_listings,
            "estimate_market_salary": self.estimate_market_salary,
            "notify_user_of_opportunity": self.notify_user_of_opportunity,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append(
                {"role": "tool", "content": result, "tool_call_id": tool_call.id}
            )
        return results

    system_message = (
        "You find promising startup valuation opportunities by scanning profiles and estimating "
        "whether reported valuation deviates materially from estimated valuation, then notify the user of the best one."
    )
    user_message = """
    First, use your tool to scan for startup profiles. Then for each listing, use your tool to estimate valuation.
    Then pick the single most compelling opportunity where the reported valuation is much higher than the estimated valuation,
    and use your tool to notify the user.
    Then just reply OK to indicate success.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    def plan(self, memory: List[str] = []) -> Optional[JobOpportunity]:
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory
        self.opportunity = None
        messages = self.messages[:]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model=self.MODEL, messages=messages, tools=self.get_tools()
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                results = self.handle_tool_call(message)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity
