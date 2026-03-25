"""Custom exception types for LLM provider errors.

Each exception maps to a specific interpretation error code defined
in internal-api.yaml. The orchestrator catches these to emit the
correct terminal event.
"""


class LLMProviderError(Exception):
    """Base class for all LLM provider errors.

    Maps to error code: interpretation.internal_error
    """


class LLMUnavailableError(LLMProviderError):
    """The LLM service is unreachable or returned a service-level error.

    Maps to error code: interpretation.model_unavailable
    """


class LLMTimeoutError(LLMProviderError):
    """The LLM did not respond within the configured timeout.

    Maps to error code: interpretation.model_timeout
    """


class LLMAuthenticationError(LLMProviderError):
    """Invalid or missing API key for the LLM provider.

    Maps to error code: interpretation.internal_error
    """


class LLMRateLimitError(LLMProviderError):
    """Rate limit exceeded on the LLM provider API.

    Maps to error code: interpretation.internal_error
    """
