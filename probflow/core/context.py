"""Context manager for ProbFlow model definition."""

from typing import Optional, List, Any


class ProbFlow:
    """Context manager for probabilistic model definition.

    This context manager provides a scope for defining probabilistic models,
    tracking distributions and their relationships during model construction.

    Example:
        >>> with ProbFlow() as model:
        ...     x = Normal(0, 1)
        ...     y = Normal(x, 1)
    """

    _active_context: Optional['ProbFlow'] = None

    def __init__(self):
        """Initialize a new ProbFlow context."""
        self.distributions: List[Any] = []
        self.variables: dict = {}
        self._parent_context: Optional['ProbFlow'] = None

    def __enter__(self) -> 'ProbFlow':
        """Enter the ProbFlow context.

        Returns:
            The ProbFlow context manager instance.
        """
        self._parent_context = ProbFlow._active_context
        ProbFlow._active_context = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the ProbFlow context.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.

        Returns:
            False to propagate any exceptions.
        """
        ProbFlow._active_context = self._parent_context
        return False

    def add_distribution(self, dist: Any, name: Optional[str] = None) -> None:
        """Register a distribution within this context.

        Args:
            dist: The distribution to register.
            name: Optional name for the distribution.
        """
        self.distributions.append(dist)
        if name is not None:
            self.variables[name] = dist

    def get_distribution(self, name: str) -> Any:
        """Retrieve a named distribution from this context.

        Args:
            name: Name of the distribution to retrieve.

        Returns:
            The distribution associated with the given name.

        Raises:
            KeyError: If no distribution with the given name exists.
        """
        return self.variables[name]

    @classmethod
    def get_active_context(cls) -> Optional['ProbFlow']:
        """Get the currently active ProbFlow context.

        Returns:
            The active context, or None if no context is active.
        """
        return cls._active_context

    @classmethod
    def is_active(cls) -> bool:
        """Check if a ProbFlow context is currently active.

        Returns:
            True if a context is active, False otherwise.
        """
        return cls._active_context is not None
