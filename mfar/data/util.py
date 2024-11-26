from typing import Any, MutableMapping, TypeVar, Tuple, Iterable
from enum import Enum

from pytorch_lightning.loggers import MLFlowLogger
import numpy as np

class SpecialToken(Enum):
    query_start = "<QRY>"
    doc_start = "<DOC>"
    id_start = "<ID>"

    def __str__(self):
        return self.value

class MLFlowLoggerWrapper(MLFlowLogger):
    def log_hyperparams(self, params: dict) -> None:
        params = {k: v for k, v in params.items() if not isinstance(v, dict)}
        if any(isinstance(v, dict) for v in params.values()):
            print("Not logging the following hyperparameters as they are dicts")
            print({k: v for k, v in params.items() if isinstance(v, dict)})
        super().log_hyperparams(params)



K = TypeVar('K')
V = TypeVar('V')

class MemoryMapDict(MutableMapping[str, np.ndarray]):

    def __init__(self, path: str, keys: Iterable[str], shape: Tuple[int, ...], mode: str = "r+", dtype: np.dtype = np.float32):
        self._keys = {key: i for i, key in enumerate(keys)}
        self._path = path
        self._shape = shape
        self._dtype = dtype
        self.file = np.memmap(path, dtype=dtype, mode=mode, shape=shape)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.file[self._keys[key], :]

    def __setitem__(self, key: str, value: np.ndarray):
        self.file[self._keys[key], :] = value

    def __delitem__(self, key: str):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return self._shape[0]

    def __contains__(self, item: str) -> bool:
        return item in self._keys

    def close(self):
        self.file.flush()


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
