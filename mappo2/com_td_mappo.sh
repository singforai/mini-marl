#!/bin/bash

command="python main.py --seed {0}"

# # 각각의 GPU에 할당할 인자 설정
# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"

# seeds=(42 123 333 666 777)  # 괄호 추가하여 배열로 선언
# for seed in "${seeds[@]}"    # "$seeds[@]"로 수정하여 배열 요소에 접근
# do
#     $command $args_gpu0 $seed &
# done

# seeds=(9876 2022 1004 8888 2468)  # 괄호 추가하여 배열로 선언
# for seed in "${seeds[@]}"    # "$seeds[@]"로 수정하여 배열 요소에 접근
# do
#     $command $args_gpu1 $seed &
# done

# wait
args_gpu0="--num_gpu 0"
args_gpu1="--num_gpu 1"

# 각각의 인자를 조합하여 명령어를 백그라운드에서 실행
parallel -j 10 $command $args_gpu0 ::: 42 123 333 666 777 &
parallel -j 10 $command $args_gpu1 ::: 9876 2022 1004 8888 2468 &