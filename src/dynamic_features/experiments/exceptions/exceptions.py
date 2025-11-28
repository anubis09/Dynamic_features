class ExperimentError(Exception):
    """Custom exception for experiment-related errors"""

    pass


class ConfigurationError(ExperimentError):
    """Exception raised for configuration-related errors"""

    pass
