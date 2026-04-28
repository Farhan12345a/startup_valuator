import modal
from agents.agent import Agent


class SpecialistAgent(Agent):
    """
    Calls the fine-tuned valuation prediction LLM running remotely on Modal.
    Mirrors the course's SpecialistAgent but for valuation estimation.
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        self.log("Specialist Agent is initializing - connecting to Modal")
        SalaryPredictor = modal.Cls.from_name("startup-valuation-service", "SalaryPredictor")
        self.predictor = SalaryPredictor()

    def estimate(self, description: str) -> float:
        """
        Make a remote call to estimate valuation for this startup description.
        """
        self.log("Specialist Agent is calling remote fine-tuned model")
        result = self.predictor.estimate.remote(description)
        self.log(f"Specialist Agent completed - estimating valuation ${result:,.0f}")
        return result
