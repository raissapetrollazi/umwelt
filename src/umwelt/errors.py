"""Public exception hierarchy for Umwelt."""


class UmweltError(Exception):
    """Base class for expected Umwelt errors."""


class ConfigurationError(UmweltError):
    """Raised when experiment configuration is invalid."""


class DataError(UmweltError):
    """Raised when source data cannot be interpreted safely."""


class DataIntegrityError(DataError):
    """Raised when source data does not match its published checksum."""
