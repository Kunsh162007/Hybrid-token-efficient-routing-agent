"""Hybrid token-efficient routing agent.

Routes each task through an escalation ladder: free local Gemma rungs first,
paid Fireworks AI rungs only when the local answer cannot be trusted.
"""

__version__ = "0.1.0"
