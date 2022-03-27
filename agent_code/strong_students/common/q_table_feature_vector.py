from abc import ABC


class QTableFeatureVector(ABC):
    def to_state(self) -> int:
        pass
