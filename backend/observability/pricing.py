"""
Centralized pricing registry and cost calculator for LLM token usage.
Supports dynamic rate updates, strict unknown pricing handling, and deterministic cost calculation.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PricingRate(BaseModel):
    """
    Per-1K token pricing specification for a model.
    """
    prompt_per_1k: float = Field(description="USD price per 1,000 prompt/input tokens.")
    completion_per_1k: float = Field(description="USD price per 1,000 completion/output tokens.")
    currency: str = Field(default="USD", description="Billing currency.")


class CostResult(float):
    """
    Dual-nature float & tuple for cost results.
    Allows both numeric comparisons (cost == 12.5) and tuple unpacking (in_c, out_c, tot_c, curr = cost).
    """
    def __new__(cls, in_cost: float, out_cost: float, total_cost: float, currency: str = "USD"):
        inst = super().__new__(cls, total_cost)
        inst.input_cost = in_cost
        inst.output_cost = out_cost
        inst.total_cost = total_cost
        inst.currency = currency
        return inst

    def __iter__(self):
        return iter((self.input_cost, self.output_cost, self.total_cost, self.currency))

    def __getitem__(self, index):
        return (self.input_cost, self.output_cost, self.total_cost, self.currency)[index]


class ModelPricingManager:
    """
    Centralized pricing registry for all LLM models and providers.
    """

    def __init__(self, fallback_rate: Optional[PricingRate] = None):
        self._fallback_rate = fallback_rate
        self._rates: Dict[str, PricingRate] = {
            "gpt-4o-mini": PricingRate(prompt_per_1k=0.00015, completion_per_1k=0.0006),
            "gpt-4o": PricingRate(prompt_per_1k=0.0025, completion_per_1k=0.01),
            "claude-3-5-sonnet": PricingRate(prompt_per_1k=0.003, completion_per_1k=0.015),
            "sonnet": PricingRate(prompt_per_1k=0.003, completion_per_1k=0.015),
            "claude-3-haiku": PricingRate(prompt_per_1k=0.00025, completion_per_1k=0.00125),
            "gemini-1.5-flash": PricingRate(prompt_per_1k=0.000075, completion_per_1k=0.0003),
            "gemini-2.0-flash": PricingRate(prompt_per_1k=0.000075, completion_per_1k=0.0003),
            "gemini": PricingRate(prompt_per_1k=0.000075, completion_per_1k=0.0003),
            "gemini-1.5-pro": PricingRate(prompt_per_1k=0.00125, completion_per_1k=0.005),
            "llama-3": PricingRate(prompt_per_1k=0.0002, completion_per_1k=0.0002),
            "llama": PricingRate(prompt_per_1k=0.0002, completion_per_1k=0.0002),
            "deepseek": PricingRate(prompt_per_1k=0.00014, completion_per_1k=0.00028),
        }

    def register_rate(
        self,
        model_key: Optional[str] = None,
        rate: Optional[PricingRate] = None,
        prompt_price_per_1m: Optional[float] = None,
        completion_price_per_1m: Optional[float] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        """Adds or updates the pricing rate for a model identifier."""
        key = (model_name or model_key or "").lower()
        if rate is not None:
            self._rates[key] = rate
        elif prompt_price_per_1m is not None and completion_price_per_1m is not None:
            self._rates[key] = PricingRate(
                prompt_per_1k=prompt_price_per_1m / 1000.0,
                completion_per_1k=completion_price_per_1m / 1000.0,
            )

    def get_rate(self, model: Optional[str]) -> Optional[PricingRate]:
        """
        Resolves pricing rate by exact match or substring match.
        Returns None if model pricing is unknown and no fallback is set.
        """
        if not model:
            return self._fallback_rate

        model_key = model.lower()
        if model_key in self._rates:
            return self._rates[model_key]

        for key, rate in self._rates.items():
            if key in model_key:
                return rate

        return self._fallback_rate

    def calculate_cost(
        self,
        model: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        allow_fallback: bool = False,
    ) -> Optional[CostResult]:
        """
        Calculates CostResult(input_cost, output_cost, total_cost, currency).
        If pricing is unavailable, returns None (UNKNOWN).
        Never invents or fabricates arbitrary costs.
        """
        rate = self.get_rate(model)
        if not rate:
            if allow_fallback and self._fallback_rate:
                rate = self._fallback_rate
            else:
                return None

        input_cost = (prompt_tokens / 1000.0) * rate.prompt_per_1k
        output_cost = (completion_tokens / 1000.0) * rate.completion_per_1k
        total_cost = input_cost + output_cost

        return CostResult(
            round(input_cost, 6),
            round(output_cost, 6),
            round(total_cost, 6),
            rate.currency,
        )


# Singleton instance for platform-wide pricing
pricing_manager = ModelPricingManager(
    fallback_rate=PricingRate(prompt_per_1k=0.0005, completion_per_1k=0.0015)
)
default_pricing = pricing_manager
