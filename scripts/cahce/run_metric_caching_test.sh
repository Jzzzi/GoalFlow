# export LD_LIBRARY_PATH="/usr/local/cuda/lib64"

SPLIT=test
SCNEE_FILTER=navtest
CACHE_TO_SAVE='$NAVSIM_EXP_ROOT/metric_cache_test' #set your metric cache path to save

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
scene_filter=$SCNEE_FILTER \
split=$SPLIT \
cache.cache_path=$CACHE_TO_SAVE \
scene_filter.frame_interval=1 \
