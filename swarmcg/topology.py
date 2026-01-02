from dataclasses import dataclass, field
from typing import Any


class DictLike:
    extra: dict[str, Any]

    def __getitem__(self, key: str):
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        return self.extra[key]

    def __setitem__(self, key: str, value):
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.__dataclass_fields__ or key in self.extra

    def get(self, key: str, default=None):
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        return self.extra.get(key, default)


@dataclass(slots=True)
class MoleculeType(DictLike):
    molname: str
    nrexcl: int
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "MoleculeType":
        known = {"molname", "nrexcl"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(molname=data["molname"], nrexcl=data["nrexcl"], extra=extra)


@dataclass(slots=True)
class Atom(DictLike):
    bead_id: int
    bead_type: str
    resnr: int
    residue: str
    atom: str
    cgnr: int
    charge: float
    mass: float | None
    vs_type: int | str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "Atom":
        known = {"bead_id", "bead_type", "resnr", "residue", "atom", "cgnr", "charge", "mass", "vs_type"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            bead_id=data["bead_id"],
            bead_type=data["bead_type"],
            resnr=data["resnr"],
            residue=data["residue"],
            atom=data["atom"],
            cgnr=data["cgnr"],
            charge=data["charge"],
            mass=data["mass"],
            vs_type=data.get("vs_type"),
            extra=extra,
        )


@dataclass(slots=True)
class ConstraintGroup(DictLike):
    geom_type: str
    beads: list[list[int]]
    func: int
    value: float
    value_user: float
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "ConstraintGroup":
        known = {"geom_type", "beads", "func", "value", "value_user"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            geom_type=data["geom_type"],
            beads=data["beads"],
            func=data["func"],
            value=data["value"],
            value_user=data["value_user"],
            extra=extra,
        )


@dataclass(slots=True)
class BondGroup(DictLike):
    geom_type: str
    beads: list[list[int]]
    func: int
    value: float
    value_user: float
    fct: float
    fct_user: float
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "BondGroup":
        known = {"geom_type", "beads", "func", "value", "value_user", "fct", "fct_user"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            geom_type=data["geom_type"],
            beads=data["beads"],
            func=data["func"],
            value=data["value"],
            value_user=data["value_user"],
            fct=data["fct"],
            fct_user=data["fct_user"],
            extra=extra,
        )


@dataclass(slots=True)
class AngleGroup(DictLike):
    geom_type: str
    beads: list[list[int]]
    func: int
    value: float
    value_user: float
    fct: float
    fct_user: float
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "AngleGroup":
        known = {"geom_type", "beads", "func", "value", "value_user", "fct", "fct_user"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            geom_type=data["geom_type"],
            beads=data["beads"],
            func=data["func"],
            value=data["value"],
            value_user=data["value_user"],
            fct=data["fct"],
            fct_user=data["fct_user"],
            extra=extra,
        )


@dataclass(slots=True)
class DihedralGroup(DictLike):
    geom_type: str
    beads: list[list[int]]
    func: int
    value: float
    value_user: float
    fct: float
    fct_user: float
    mult: int | None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "DihedralGroup":
        known = {"geom_type", "beads", "func", "value", "value_user", "fct", "fct_user", "mult"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            geom_type=data["geom_type"],
            beads=data["beads"],
            func=data["func"],
            value=data["value"],
            value_user=data["value_user"],
            fct=data["fct"],
            fct_user=data["fct_user"],
            mult=data.get("mult"),
            extra=extra,
        )


@dataclass(slots=True)
class VirtualSite2(DictLike):
    bead_id: int
    func: int
    vs_def_beads_ids: list[int]
    vs_params: float
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "VirtualSite2":
        known = {"bead_id", "func", "vs_def_beads_ids", "vs_params"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            bead_id=data["bead_id"],
            func=data["func"],
            vs_def_beads_ids=data["vs_def_beads_ids"],
            vs_params=data["vs_params"],
            extra=extra,
        )


@dataclass(slots=True)
class VirtualSite3(DictLike):
    bead_id: int
    func: int
    vs_def_beads_ids: list[int]
    vs_params: list[float]
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "VirtualSite3":
        known = {"bead_id", "func", "vs_def_beads_ids", "vs_params"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            bead_id=data["bead_id"],
            func=data["func"],
            vs_def_beads_ids=data["vs_def_beads_ids"],
            vs_params=data["vs_params"],
            extra=extra,
        )


@dataclass(slots=True)
class VirtualSite4(DictLike):
    bead_id: int
    func: int
    vs_def_beads_ids: list[int]
    vs_params: list[float]
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "VirtualSite4":
        known = {"bead_id", "func", "vs_def_beads_ids", "vs_params"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            bead_id=data["bead_id"],
            func=data["func"],
            vs_def_beads_ids=data["vs_def_beads_ids"],
            vs_params=data["vs_params"],
            extra=extra,
        )


@dataclass(slots=True)
class VirtualSiteN(DictLike):
    bead_id: int
    func: int
    vs_def_beads_ids: list[int]
    vs_params: list[float] | None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "VirtualSiteN":
        known = {"bead_id", "func", "vs_def_beads_ids", "vs_params"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            bead_id=data["bead_id"],
            func=data["func"],
            vs_def_beads_ids=data["vs_def_beads_ids"],
            vs_params=data.get("vs_params"),
            extra=extra,
        )


@dataclass(slots=True)
class CgTopology(DictLike):
    moleculetype: MoleculeType
    atoms: list[Atom]
    constraint: list[ConstraintGroup]
    bond: list[BondGroup]
    angle: list[AngleGroup]
    dihedral: list[DihedralGroup]
    virtual_sites2: dict[int, VirtualSite2]
    virtual_sites3: dict[int, VirtualSite3]
    virtual_sites4: dict[int, VirtualSite4]
    virtual_sitesn: dict[int, VirtualSiteN]
    exclusion: list[list[int]]
    real_beads_ids: list[int]
    vs_beads_ids: list[int]
    nb_bonds: int
    nb_angles: int
    nb_dihedrals: int
    nb_constraints: int
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, data: dict[str, Any]) -> "CgTopology":
        known = {
            "moleculetype",
            "atoms",
            "constraint",
            "bond",
            "angle",
            "dihedral",
            "virtual_sites2",
            "virtual_sites3",
            "virtual_sites4",
            "virtual_sitesn",
            "exclusion",
            "real_beads_ids",
            "vs_beads_ids",
            "nb_bonds",
            "nb_angles",
            "nb_dihedrals",
            "nb_constraints",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            moleculetype=MoleculeType.from_legacy(data["moleculetype"]),
            atoms=[Atom.from_legacy(atom) for atom in data["atoms"]],
            constraint=[ConstraintGroup.from_legacy(item) for item in data["constraint"]],
            bond=[BondGroup.from_legacy(item) for item in data["bond"]],
            angle=[AngleGroup.from_legacy(item) for item in data["angle"]],
            dihedral=[DihedralGroup.from_legacy(item) for item in data["dihedral"]],
            virtual_sites2={key: VirtualSite2.from_legacy(val) for key, val in data["virtual_sites2"].items()},
            virtual_sites3={key: VirtualSite3.from_legacy(val) for key, val in data["virtual_sites3"].items()},
            virtual_sites4={key: VirtualSite4.from_legacy(val) for key, val in data["virtual_sites4"].items()},
            virtual_sitesn={key: VirtualSiteN.from_legacy(val) for key, val in data["virtual_sitesn"].items()},
            exclusion=data["exclusion"],
            real_beads_ids=data["real_beads_ids"],
            vs_beads_ids=data["vs_beads_ids"],
            nb_bonds=data["nb_bonds"],
            nb_angles=data["nb_angles"],
            nb_dihedrals=data["nb_dihedrals"],
            nb_constraints=data["nb_constraints"],
            extra=extra,
        )
