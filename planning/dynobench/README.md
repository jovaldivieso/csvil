# motion primitive model configs

this directory contains dynobench model configuration files used to generate motion primitives

the YAML files define the dynamics model and its limits 
(e.g. velocity, robot shape, integration timestep)

these configs are used by:

```bash
python planning/generate_motion_primitives.py --config <model>.yaml