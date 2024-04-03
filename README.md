# mini-marl


base_share_rmappo
1개의 agent network와 1개의 critic network가 존재한다. agent network는 각 agent의 local_observation을 입력으로 받으며 critic은 두 에이전트의 관측의 합을 입력으로 받는다(critic이 하나임으로  use_centralized_V는 무조건적으로 사용되어 args.use_centralized_V의 값에 영향을 받지 않는다). reward는 공유되지 않은 상태여서 (use_common_reward == False) 각 agent는 다른 reward를 받는다. training_batch_size, sampling_batch_size는 1로 고정되어 있다. 

base_separate_rmappo
n개의 agent network와 n개의 critic network가 존재한다. 각 agent network는 각 agent의 local_observation을 입력으로 받으며 각 critic은 두 에이전트의 관측의 합을 입력으로 받는다(use_centralized_V가 False일 경우, 각 critic은 각 agent의 local observation만을 입력으로 받는다). reward는 공유되지 않은 상태여서 (use_common_reward == False) 각 agent는 다른 reward를 받는다. training, sampling_batch_size는 1로 고정되어 있다. 


com_share_rmappo
1개의 agent network와 1개의 critic network가 존재한다. agent network는 각 agent의 local_observation을 입력으로 받으며 critic은 두 에이전트의 관측의 합을 입력으로 받는다(critic이 하나임으로  use_centralized_V는 무조건적으로 사용되어 args.use_centralized_V의 값에 영향을 받지 않는다). reward는 공유된 상태여서 (use_common_reward == True) 각 agent는 같은 reward를 받는다. training_batch_size, sampling_batch_size는 1로 고정되어 있다. 

com_separate_rmappo
n개의 agent network와 n개의 critic network가 존재한다. 각 agent network는 각 agent의 local_observation을 입력으로 받으며 각 critic은 두 에이전트의 관측의 합을 입력으로 받는다(use_centralized_V가 False일 경우, 각 critic은 각 agent의 local observation만을 입력으로 받는다). reward는 공유된 상태여서 (use_common_reward == True) 각 agent는 같은 reward를 받는다. training, sampling_batch_size는 1로 고정되어 있다. 

com_hybrid_rmappo
