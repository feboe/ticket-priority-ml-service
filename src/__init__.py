"""Project source package.

Keep this package root lightweight so serving can import pickled training objects
without pulling in training-only dependencies such as mlflow.
"""

__all__: list[str] = []
