import re
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from agents.agent import Agent


class FrontierAgent(Agent):
    """
    RAG-based valuation estimator: encodes the startup description, looks up similar
    startup profiles in ChromaDB, and asks a frontier model to estimate valuation with
    that context.
    """

    name = "Frontier Agent"
    color = Agent.BLUE
    MODEL = "gpt-4o-mini"

    def __init__(self, collection):
        self.log("Initializing Frontier Agent")
        self.client = OpenAI()
        self.MODEL = "gpt-4o"
        self.log("Frontier Agent is setting up with OpenAI")
        self.collection = collection
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], salaries: List[float]) -> str:
        message = (
            "To provide context, here are similar startups and their reported valuations.\n\n"
        )
        for similar, salary in zip(similars, salaries):
            message += f"Similar startup:\n{similar}\nReported valuation is ${salary:,.0f}\n\n"
        return message

    def messages_for(
        self, description: str, similars: List[str], salaries: List[float]
    ) -> List[Dict[str, str]]:
        message = (
            "Estimate the likely startup valuation in USD for this profile. "
            "Respond with only the number, no explanation.\n\n"
            f"{description}\n\n"
        )
        message += self.make_context(similars, salaries)
        return [{"role": "user", "content": message}]

    def find_similars(self, description: str):
        self.log(
            "Frontier Agent is performing a RAG search of ChromaDB to find 5 similar startup profiles"
        )
        vector = self.model.encode([description])
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(), n_results=5
        )
        documents = results["documents"][0][:]
        salaries = [m["salary"] for m in results["metadatas"][0][:]]
        self.log("Frontier Agent has found similar startup profiles")
        return documents, salaries

    def get_salary(self, s) -> float:
        s = s.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.?\d+", s)
        return float(match.group()) if match else 0.0

    def estimate(self, description: str) -> float:
        documents, salaries = self.find_similars(description)
        self.log(
            f"Frontier Agent is calling {self.MODEL} with context from 5 similar startup profiles"
        )
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.messages_for(description, documents, salaries),
            seed=42,
        )
        reply = response.choices[0].message.content
        result = self.get_salary(reply)
        self.log(f"Frontier Agent completed - predicting ${result:,.0f}")
        return result
