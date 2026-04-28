from pydantic import BaseModel, Field
from typing import List, Dict, Self
from bs4 import BeautifulSoup
import re
import feedparser
import requests
import time

feeds = [
    "https://techcrunch.com/category/startups/feed/",
    "https://www.eu-startups.com/feed/",
    "https://www.saastr.com/feed/",
]


def extract_text(html_snippet: str) -> str:
    """Clean HTML and extract readable text."""
    soup = BeautifulSoup(html_snippet, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


class ScrapedJob:
    """A raw startup article/profile retrieved from an RSS feed."""

    title: str
    summary: str
    url: str
    details: str

    def __init__(self, entry: Dict[str, str]):
        self.title = entry.get("title", "Unknown Startup")
        raw_summary = entry.get("summary", entry.get("description", ""))
        self.summary = extract_text(raw_summary)
        links = entry.get("links", [{}])
        self.url = entry.get("link", links[0].get("href", "") if links else "")
        self.details = ""
        try:
            resp = requests.get(
                self.url, timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (compatible; StartupValuationBot/1.0)"},
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                content = (
                    soup.find("div", class_="content")
                    or soup.find("article")
                    or soup.find("main")
                )
                if content:
                    self.details = content.get_text(separator=" ", strip=True)[:1000]
        except Exception:
            pass
        self.truncate()

    def truncate(self):
        self.title = self.title[:200]
        self.summary = self.summary[:800]
        self.details = self.details[:1000]

    def __repr__(self):
        return f"<{self.title}>"

    def describe(self):
        parts = [f"Title: {self.title}"]
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.details:
            parts.append(f"Details: {self.details}")
        parts.append(f"URL: {self.url}")
        return "\n".join(parts)

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        jobs = []
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    try:
                        jobs.append(cls(entry))
                        time.sleep(0.1)
                    except Exception:
                        continue
            except Exception:
                continue
        return jobs


class JobListing(BaseModel):
    """A structured startup profile produced by structured outputs."""

    job_title: str = Field(
        description="The startup name (e.g. 'Acme AI', 'Nimbus Bio')."
    )
    company: str = Field(description="Primary sector or category (e.g. fintech, devtools).")
    salary: float = Field(
        description="The reported or implied startup valuation in USD. "
        "If only a range is available, use the midpoint. Use 0 if unknown."
    )
    location: str = Field(description="The startup HQ, region, or 'Global/Remote' if unclear.")
    description: str = Field(
        description="A concise 2-3 sentence summary with company description, funding rounds, and traction metrics."
    )
    url: str = Field(description="The URL of the startup profile/article, as provided in the input")


class JobSelection(BaseModel):
    """A curated list of startup profiles."""

    listings: List[JobListing] = Field(
        description="Your selection of the 5 startup profiles that have the most detailed "
        "description and the clearest valuation information."
    )


class JobOpportunity(BaseModel):
    """A startup where reported valuation differs significantly from estimated valuation."""

    listing: JobListing
    estimate: float
    premium: float
