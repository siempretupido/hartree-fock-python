# src/models.py
from dataclasses import dataclass #sirve para definir un objeto de forma limpia
from typing import List #sirve para incluir un nuevo tipo de variable llamada list (no es totalmente necessario en python)

@dataclass
class Atom:
    label: str
    Z: int
    x: float
    y: float
    z: float

@dataclass
class Molecule:
    atoms: List[Atom]
    charge: int
    nbasis: int
    max_nc: int

    @property
    def nelectrons(self) -> int:
        return sum(a.Z for a in self.atoms) - self.charge
