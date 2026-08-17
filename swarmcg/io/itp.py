from collections import UserDict
from swarmcg.shared import exceptions

class CGITP(UserDict):
    """
    Object representing the content of a CG ITP file.
    Behaves like a dictionary for backward compatibility but allows for
    structured access and validation.
    """
    def __init__(self, data=None):
        super().__init__(data)
        if not self.data:
            self.data = {
                "moleculetype": {"molname": "", "nrexcl": 0},
                "atoms": [],
                "constraint": [],
                "bond": [],
                "angle": [],
                "dihedral": [],
                "virtual_sites2": {},
                "virtual_sites3": {},
                "virtual_sites4": {},
                "virtual_sitesn": {},
                "exclusion": [],
                # Metadata fields often added by the reader
                "real_beads_ids": [],
                "vs_beads_ids": [],
                "nb_bonds": 0,
                "nb_angles": 0,
                "nb_dihedrals": 0,
                "nb_constraints": 0
            }

    def validate(self) -> None:
        """Validate topology counts, bead references, and parameter shapes.

        Raises:
            MissformattedFile: If topology state is internally inconsistent.
        """
        required = {
            "moleculetype", "atoms", "constraint", "bond", "angle", "dihedral",
            "real_beads_ids", "vs_beads_ids", "nb_constraints", "nb_bonds",
            "nb_angles", "nb_dihedrals",
        }
        missing = sorted(required.difference(self.data))
        if missing:
            raise exceptions.MissformattedFile(
                f"CG ITP representation is missing required fields: {', '.join(missing)}."
            )

        atom_count = len(self.data["atoms"])
        atom_ids = [atom.get("bead_id") for atom in self.data["atoms"]]
        if atom_ids != list(range(atom_count)):
            raise exceptions.MissformattedFile(
                "CG ITP internal atom identifiers must be consecutive and start at zero."
            )

        real_ids = set(self.data["real_beads_ids"])
        virtual_ids = set(self.data["vs_beads_ids"])
        if real_ids.intersection(virtual_ids) or real_ids.union(virtual_ids) != set(range(atom_count)):
            raise exceptions.MissformattedFile(
                "Real and virtual bead identifiers must form a disjoint partition of all CG atoms."
            )

        geometry_specs = {
            "constraint": ("nb_constraints", 2),
            "bond": ("nb_bonds", 2),
            "angle": ("nb_angles", 3),
            "dihedral": ("nb_dihedrals", 4),
        }
        for geometry, (count_key, arity) in geometry_specs.items():
            groups = self.data[geometry]
            if self.data[count_key] != len(groups):
                raise exceptions.MissformattedFile(
                    f"CG ITP reports {self.data[count_key]} {geometry} groups but stores {len(groups)}."
                )
            for group_index, group in enumerate(groups, start=1):
                for bead_tuple in group.get("beads", []):
                    if len(bead_tuple) != arity or any(index < 0 or index >= atom_count for index in bead_tuple):
                        raise exceptions.MissformattedFile(
                            f"Invalid bead identifiers in {geometry} group {group_index}."
                        )
                if geometry == "dihedral" and group.get("func") in (3, 11):
                    if len(group.get("params", [])) != 6:
                        raise exceptions.MissformattedFile(
                            f"Dihedral function {group.get('func')} in group {group_index} requires six parameters."
                        )

    @property
    def atoms(self):
        return self.data.get("atoms", [])

    @property
    def moleculetype(self):
        return self.data.get("moleculetype", {})

    @property
    def constraints(self):
        return self.data.get("constraint", [])

    @property
    def bonds(self):
        return self.data.get("bond", [])

    @property
    def angles(self):
        return self.data.get("angle", [])

    @property
    def dihedrals(self):
        return self.data.get("dihedral", [])
        
    def get_geom_list(self, geom_name):
        return self.data.get(geom_name, [])
