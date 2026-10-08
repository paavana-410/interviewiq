"""Counterfactual Audit Layer.

Five independent test modules, each with its own metric:
  1. delivery_invariance  — same content, perturbed delivery
  2. content_sensitivity  — degraded content, fixed delivery
  3. paraphrase_stability — meaning-preserving paraphrases
  4. repeated_run         — same input N times
  5. stt_sensitivity      — gold vs Whisper transcript

A Combiner aggregates all metrics against configurable thresholds
into a stable / low-confidence flag.
"""
from app.audit.variants import VariantGenerator  # noqa: F401
from app.audit.tests.delivery_invariance import DeliveryInvarianceTest  # noqa: F401
from app.audit.tests.content_sensitivity import ContentSensitivityTest  # noqa: F401
from app.audit.tests.paraphrase_stability import ParaphraseStabilityTest  # noqa: F401
from app.audit.tests.repeated_run import RepeatedRunTest  # noqa: F401
from app.audit.tests.stt_sensitivity import STTSensitivityTest  # noqa: F401
from app.audit.combiner import AuditCombiner, AuditReport  # noqa: F401
