"""Esta função sanitiza o DataFrame, convertendo tipos não nativos do Python em tipos nativos."""
from typing import Any, Dict, List, Literal
from pydantic import BaseModel
from dataclasses import dataclass, asdict, is_dataclass
import numpy as np
import pandas as pd

def to_python_scalar(value: Any) -> Any:
    """Converte valores numpy/pandas em tipos Python nativos."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def sanitize_data(data: Any) -> Any:
    """Normaliza estruturas (dict, lista, BaseModel, dataclass) para tipos Python serializáveis."""
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="python")
    elif is_dataclass(data):
        data = asdict(data)

    if isinstance(data, dict):
        return {key: sanitize_data(value) for key, value in data.items()}
    if isinstance(data, list):
        return [sanitize_data(item) for item in data]

    return to_python_scalar(data)

def sanitize_records(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converte DataFrames para estruturas JSON válidas, substituindo NaN por None."""
    clean_df = dataframe.replace({np.nan: None})
    records = clean_df.to_dict(orient="records")
    return [sanitize_data(record) for record in records]