# export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
# The trajectory_sampling.time_horizon in trainval is 5.5 (default)
CACHE_TO_SAVE='$NAVSIM_EXP_ROOT/dataset_cache_traival' #set your feature cache path to save

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
agent=goalflow_agent_traj \
agent.config.trajectory_sampling.time_horizon=5 \
experiment_name=a_goalflow_trainval_cache \
cache_path=$CACHE_TO_SAVE \
train_test_split=trainval \
split=trainval