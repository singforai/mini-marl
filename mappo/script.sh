#!/bin/bash

#base_env
command="python main.py --seed {0} --experiment_name base_rmappo --group_name base_rmappo --use_centralized_V True"
#command="python main.py --seed {0} --use_mix_advantage True --experiment_name advanced_rmappo --group_name advanced_rmappo"
#command="python main.py --seed {0} --use_common_reward False --experiment_name seperated_rmappo --group_name seperated_rmappo"

#large_env720
#command="python main.py --seed {0} --experiment_name base_rmappo720 --group_name base_rmappo720 --max_step 500"
# command="python main.py --seed {0} --experiment_name central_rmappo720 --group_name central_rmappo720 --max_step 500 --use_centralized_V True"

parallel -j 10 $command ::: 0 10 20 30 40 50 60 70 80 90