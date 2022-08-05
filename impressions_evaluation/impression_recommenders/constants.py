from enum import Enum


class ERankMethod(Enum):
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    DENSE = "dense"
    ORDINAL = "ordinal"
