import numpy as np
import MDAnalysis as mda

from swarmcg.simulations.vs_functions import vsn_func_2
from swarmcg.topology import Atom, CGTopology


def test_vsn_func_2_warns_on_zero_mass(caplog):
    universe = mda.Universe.empty(2, trajectory=True)
    universe.add_TopologyAttr("masses")
    universe.atoms.masses = [0.0, 1.0]
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=float)
    universe.load_new(coords)

    traj = np.empty((len(universe.trajectory), 3))
    topology = CGTopology(
        atoms=[
            Atom(0, "P1", 1, "MOL", "B1", 1, 0.0, 0.0),
            Atom(1, "P1", 1, "MOL", "B2", 2, 0.0, 1.0),
        ]
    )

    vsn_func_2(universe, traj, [0, 1], bead_id=5, topology=topology)

    assert "Virtual site ID 6" in caplog.text
    assert "IDs 1" in caplog.text
    assert traj.shape == (1, 3)
    assert np.allclose(traj[0], [1.0, 0.0, 0.0])
