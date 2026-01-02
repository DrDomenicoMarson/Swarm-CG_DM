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

    def validate(self):
        """Perform internal consistency checks."""
        # Example validation: check atom count matches bead ids
        # This logic was previously scattered or implicit
        pass

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
