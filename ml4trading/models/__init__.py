"""Every method module is imported here so its ``register_model`` runs."""

from ml4trading.models import constant_portfolio  # noqa: F401
from ml4trading.models.base import Model
from ml4trading.models.registry import get_model, register_model, registered

__all__ = ["Model", "get_model", "register_model", "registered"]
