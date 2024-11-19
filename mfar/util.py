import numpy as np
from typing import Any


def remove_irregularities(obj: Any) -> Any:
    """Ensures that the object is JSON-serializable."""
    if isinstance(obj, str):
        return obj.replace("\n", " ").replace("\t", " ").replace("\r", " ").replace("\u001f", " ").strip()
    elif isinstance(obj, list):
        return [remove_irregularities(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: remove_irregularities(v) for k, v in obj.items()}
    elif isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, bool) or obj is None:
        return obj
    elif isinstance(obj, np.bool_):
        return obj.item()
    else:
        raise ValueError(f"Unexpected type {type(obj)}")
