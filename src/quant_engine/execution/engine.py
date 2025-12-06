from quant_engine.utils.logger import (
    get_logger, log_debug, log_info, log_warn, log_error
)
from quant_engine.utils.timer import timed_block


class ExecutionEngine:
    """Execution layer: policy → router → slippage → matching.
    Does NOT touch portfolio — higher layer must handle fills."""

    def __init__(self, policy, router, slippage_model, matcher):
        self.policy = policy
        self.router = router
        self.slippage = slippage_model
        self.matcher = matcher

        self._logger = get_logger(__name__)

    def execute(self, decision, market_state):
        """Execute order through the full execution pipeline.
        Returns a Fill or None.
        """

        with timed_block("execution.total"):

            # -----------------------
            # 1. Policy → Order
            # -----------------------
            with timed_block("execution.policy"):
                order = self.policy.generate(decision, market_state)
                log_debug(
                    self._logger,
                    "Policy generated order",
                    stage="policy",
                    order=None if order is None else order.to_dict()
                )

            if order is None or order.qty == 0:
                log_info(
                    self._logger,
                    "Execution skipped: empty or null order",
                    stage="policy"
                )
                return None

            # -----------------------
            # 2. Router
            # -----------------------
            with timed_block("execution.router"):
                routed = self.router.route(order, market_state)
                log_debug(
                    self._logger,
                    "Router produced routed order",
                    stage="router",
                    routed_order=routed.to_dict()
                )

            # -----------------------
            # 3. Slippage
            # -----------------------
            with timed_block("execution.slippage"):
                adjusted = self.slippage.apply(routed, market_state)
                log_debug(
                    self._logger,
                    "Slippage applied",
                    stage="slippage",
                    adjusted_price=adjusted.price,
                    slip=adjusted.extra.get("slippage", 0.0)
                )

            # -----------------------
            # 4. MatchingEngine → Fill
            # -----------------------
            with timed_block("execution.match"):
                fill = self.matcher.match(adjusted, market_state)
                log_info(
                    self._logger,
                    "Order matched",
                    stage="matching",
                    fill=fill
                )

            return fill