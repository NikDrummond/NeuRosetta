""" Base class, every single object inherits from this"""

class _Stone():
    """Core single item class"""

    # constructor
    def __init__(self, ID: int, metadata: dict) -> None:
        self.ID = ID
        self.metadata = metadata
