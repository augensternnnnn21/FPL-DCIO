#!/bin/bash

# Create main project directory
mkdir -p dsn_fpl/{src,tests,data,configs,notebooks}

# Create source code directory structure
mkdir -p dsn_fpl/src/{core,models,utils,api}
mkdir -p dsn_fpl/src/core/{quantum,chaos,nash,temporal}
mkdir -p dsn_fpl/data/{raw,processed,embeddings}
mkdir -p dsn_fpl/configs/{model,environment}

# Create initial files
touch dsn_fpl/src/core/quantum/q_player.py
touch dsn_fpl/src/core/chaos/ripple_generator.py
touch dsn_fpl/src/core/nash/game_topology.py
touch dsn_fpl/src/core/temporal/origami_planner.py
touch dsn_fpl/requirements.txt
touch dsn_fpl/README.md

# Make setup script executable
chmod +x setup.sh 