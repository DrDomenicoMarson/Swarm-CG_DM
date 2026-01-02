import numpy as np
import MDAnalysis as mda

from swarmcg.simulations.vs_functions import vsn_func_2


def test_vsn_func_2_warns_on_zero_mass(capsys):
    universe = mda.Universe.empty(2, trajectory=True)
    universe.add_TopologyAttr("masses")
    universe.atoms.masses = [0.0, 1.0]
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=float)
    universe.load_new(coords)

    traj = np.empty((len(universe.trajectory), 3))
    cg_itp = {"atoms": [{"mass": 0.0}, {"mass": 1.0}], "virtual_sitesn": {}}

    vsn_func_2(universe, traj, [0, 1], bead_id=5, cg_itp=cg_itp)

    out = capsys.readouterr().out
    assert "Virtual site ID 6" in out
    assert "IDs 1" in out
    assert traj.shape == (1, 3)
    assert np.allclose(traj[0], [1.0, 0.0, 0.0])
