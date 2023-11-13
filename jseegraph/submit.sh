#!/bin/bash


# ace

id=$(sbatch run.sh config/ace_p_evt_ent.yaml ace_p_evt_ent)
echo "ace_p_evt_ent -- ${id}"

id=$(sbatch run.sh config/ace_p_evt.yaml ace_p_evt)
echo "ace_p_evt -- ${id}"


id=$(sbatch run.sh config/ace_pp_evt_ent.yaml ace_pp_evt_ent)
echo "ace_pp_evt_ent -- ${id}"

id=$(sbatch run.sh config/ace_pp_evt.yaml ace_pp_evt)
echo "ace_pp_evt -- ${id}"


# ere

id=$(sbatch run.sh config/ere_p_evt_ent.yaml ere_p_evt_ent)
echo "ere_p_evt_ent -- ${id}"

id=$(sbatch run.sh config/ere_p_evt.yaml ere_p_evt)
echo "ere_p_evt -- ${id}"


id=$(sbatch run.sh config/ere_pp_evt_ent.yaml ere_pp_evt_ent)
echo "ere_pp_evt_ent -- ${id}"

id=$(sbatch run.sh config/ere_pp_evt.yaml ere_pp_evt)
echo "ere_pp_evt -- ${id}"


