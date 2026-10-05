"""Name -> model class. A method module registers itself at import; ``models/__init__.py``
imports every method module so the registry is complete once the package is imported."""

from __future__ import annotations

from ml4trading.models.base import Model

_MODELS: dict[str, type[Model]] = {}


def register_model(name: str):
    def deco(cls: type[Model]) -> type[Model]:
        if name in _MODELS:
            raise ValueError(f"model {name!r} registered twice")
        cls.name = name
        _MODELS[name] = cls
        return cls

    return deco


def get_model(name: str) -> type[Model]:
    if name not in _MODELS:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(_MODELS)}")
    return _MODELS[name]


def registered() -> list[str]:
    return sorted(_MODELS)
