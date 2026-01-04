#!/bin/bash

scg_optimize \
  -aa_tpr ./data/aa_topol.tpr \
  -aa_traj ./data/aa_traj.xtc \
  -cg_map ./data/cg_map.ndx \
  -cg_itp ./data/cg_model.itp \
  -cg_gro ./data/start_conf.gro \
  -cg_top ./data/system.top \
  -cg_mdp_mini ./data/mini.mdp \
  -cg_mdp_equi ./data/equi.mdp \
  -cg_mdp_md ./data/md.mdp \
