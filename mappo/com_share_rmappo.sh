#!/bin/bash

command="python main.py --seed {0}"


# 각각의 인자를 조합하여 명령어를 백그라운드에서 실행
parallel -j 5 $command ::: 0 10 20 30 40 &
# parallel -j 10 $command ::: 50 60 70 80 90 &

# 모든 작업이 완료될 때까지 대기
wait
