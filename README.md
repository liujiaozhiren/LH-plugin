

# LH-plugin

## Introduction
This repository showcases the LH-plugin currently. The LH-plugin has been adapted for use with the following models: TrajGAT, Neutraj, ST2Vec, and Traj2SimVec (with slight variations for each model). For upcoming, we will pluginize the LH-plugin to adapt it for various trajectory similarity models and enhance its usability.

## Usage Instructions

### TrajGAT
To use the TrajGAT model, follow these steps:
```
cd trajGAT
...envs/{env_name}/bin/python main.py -L 1 -s {sim_func} -c {city} --other_args {argsvalue}            # with LH-plugin
...envs/{env_name}/bin/python main.py -L 0 -s {sim_func} -c {city} --other_args {argsvalue}            # without LH-plugin
```

### Neutraj
To use the Neutraj model, follow these steps:
```
cd neutraj
...envs/{env_name}/bin/python train.py -L 1 -s {sim_func} -c {city} --other_args {argsvalue}           # with LH-plugin
...envs/{env_name}/bin/python train.py -L 0 -s {sim_func} -c {city} --other_args {argsvalue}           # without LH-plugin
```

### ST2Vec
To use the ST2Vec model, follow these steps:
```
cd ST2Vec
...envs/{env_name}/bin/python main.py -L 1 -c {config.yaml|config-DITA.yaml|config-*.yaml}             # with LH-plugin
...envs/{env_name}/bin/python main.py -L 0 -c {config.yaml|config-DITA.yaml|config-*.yaml}             # without LH-plugin
```

### Traj2SimVec
To use the Traj2SimVec model, follow these steps:
```
cd Traj2SimVec
...envs/{env_name}/bin/python main.py -L 1 -s {sim_func} -c {city} --other_args {argsvalue}            # with LH-plugin
...envs/{env_name}/bin/python main.py -L 0 -s {sim_func} -c {city} --other_args {argsvalue}            # without LH-plugin
```

other args : using -q to specific \gamma args 

## Data Preparation


To prepare your data for use with the LH-plugin, follow these instructions:

1. Place your dataset in the directory structure: `data_set/{city}/`.
2. Save your trajectory data as `data_set/{city}/trajs.pkl`.
3. Name your similarity data according to `data_set/{city}/{sim}.pkl`.

For preprocessing data for the ST2Vec model, please refer to the instructions provided in `ST2Vec/README.md`.
