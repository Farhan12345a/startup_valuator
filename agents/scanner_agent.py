from typing import Optional, List
from openai import OpenAI
from agents.job_listings import ScrapedJob, JobSelection
from agents.agent import Agent


class ScannerAgent(Agent):
    MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = """You identify and summarize the 5 most promising startup profiles from a list.
    Select entries that have the most detailed company description plus funding rounds, traction metrics, and valuation clues.
    Respond strictly in JSON. You should provide the salary as a number derived from the description.
    Here, salary field means startup valuation in USD.
    If a valuation range is given, use the midpoint.
    If valuation isn't clear, do not include that listing.
    Most important is that you respond with the 5 listings that have detailed company, funding, and traction context.
    Focus on business signals, not generic PR text.
    """

    USER_PROMPT_PREFIX = """Respond with the most promising 5 startup profiles from this list,
    selecting those with detailed company context and a clear valuation greater than 0.
    Summarize each startup focusing on what they build, funding rounds, and traction metrics.
    Remember to respond with a short paragraph in the description field for each listing.

    Startup Profiles:

    """

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 listings, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        self.log("Scanner Agent is initializing")
        self.openai = OpenAI()
        self.log("Scanner Agent is ready")

    def fetch_jobs(self, memory) -> List[ScrapedJob]:
        self.log("Scanner Agent is fetching startup profiles from RSS feeds")
        urls = [opp.listing.url for opp in memory]
        scraped = ScrapedJob.fetch()
        result = [s for s in scraped if s.url not in urls]
        self.log(f"Scanner Agent received {len(result)} new startup profiles")
        return result

    def make_user_prompt(self, scraped) -> str:
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([s.describe() for s in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: List[str] = []) -> Optional[JobSelection]:
        """
        Scrape RSS feeds and use structured outputs to parse startup profiles.
        """
        scraped = self.fetch_jobs(memory)
        if scraped:
            user_prompt = self.make_user_prompt(scraped)
            self.log("Scanner Agent is calling OpenAI using Structured Outputs")
            result = self.openai.beta.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=JobSelection,
            )
            result = result.choices[0].message.parsed
            result.listings = [l for l in result.listings if l.salary > 0]
            self.log(
                f"Scanner Agent received {len(result.listings)} startup profiles with valuation>0"
            )
            return result
        return None

    def test_scan(self, memory: List[str] = []) -> Optional[JobSelection]:
        """Return test data for development without live RSS feeds."""
        results = {
            "listings": [
                {
                    "job_title": "NebulaAI",
                    "company": "AI Infrastructure",
                    "salary": 320000000,
                    "location": "San Francisco, CA",
                    "description": "Builds LLM orchestration tooling for enterprise teams. Series B with $22M ARR and 180% YoY growth.",
                    "url": "https://example.com/startup/nebulaai",
                },
                {
                    "job_title": "LedgerFlow",
                    "company": "Fintech",
                    "salary": 140000000,
                    "location": "Remote",
                    "description": "Provides API-first treasury workflows for SMBs. Raised a $30M Series A and processes $1.2B annualized payment volume.",
                    "url": "https://example.com/startup/ledgerflow",
                },
                {
                    "job_title": "CureMesh",
                    "company": "Healthtech",
                    "salary": 210000000,
                    "location": "Boston, MA",
                    "description": "Clinical operations platform for specialty providers. Series C with 900 clinics onboarded and 3.1M patient workflows per month.",
                    "url": "https://example.com/startup/curemesh",
                },
                {
                    "job_title": "OrbitSupply",
                    "company": "Logistics SaaS",
                    "salary": 95000000,
                    "location": "Seattle, WA",
                    "description": "Forecasting and procurement stack for mid-market manufacturers. Bootstrapped to $9M ARR before taking growth capital.",
                    "url": "https://example.com/startup/orbitsupply",
                },
            ]
        }
        return JobSelection(**results)
