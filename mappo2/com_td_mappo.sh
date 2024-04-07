#!/bin/bash

command="python main.py --seed {0} --experiment_name td_mappo --policy_type separate --use_wandb True"

# 각각의 GPU에 할당할 인자 설정
args_gpu0="--num_gpu 0"
args_gpu1="--num_gpu 1"

# 각각의 인자를 조합하여 명령어를 백그라운드에서 실행
parallel -j 10 $command $args_gpu0 ::: 0 10 20 30 40 &
parallel -j 10 $command $args_gpu1 ::: 50 60 70 80 90 &

# 모든 작업이 완료될 때까지 대기
wait