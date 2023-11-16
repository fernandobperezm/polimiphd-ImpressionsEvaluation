#!/bin/bash -l
# This is the script ran by cron. It uses the shebang with the option -l so it loads the profile from .bashrc or profile or how it is called in the system. This cronjob is supposed to be run everytime an instance is rebooted / turned on, therefore, it outputs all stdout and stderr to a file with the current date.
WORKING_FOLDER="/fbpm/project-impressions/impressions-evaluation"
OUTPUTS_FOLDER="outputs"

DATE=$(date '+%Y%m%d_%H%M%S')

OUT_FILE_DASK="$WORKING_FOLDER/$OUTPUTS_FOLDER/$DATE-DASK_PROCESSES.txt"
OUT_FILE_DATASETS_STATISTICS="$WORKING_FOLDER/$OUTPUTS_FOLDER/$DATE-DATASETS_STATISTICS.txt"
OUT_FILE_GRAPH="$WORKING_FOLDER/$OUTPUTS_FOLDER/$DATE-GRAPH_BASED.txt"
OUT_FILE_EVALUATION="$WORKING_FOLDER/$OUTPUTS_FOLDER/$DATE-2022_EVALUATION_REMAINING_RECOMMENDERS.txt"
# OUT_FILE_EVALUATION="$WORKING_FOLDER/$OUTPUTS_FOLDER/$DATE-2022_EVALUATION_TEST.txt"

if [ ! -d $WORKING_FOLDER/$OUTPUTS_FOLDER ]
then
    mkdir $WORKING_FOLDER/$OUTPUTS_FOLDER
fi

cd $WORKING_FOLDER || exit

poetry run python scripts/compute_evaluation_study_datasets_statistics.py \
  --create_datasets \
  --plot_datasets_popularity \
  --print_datasets_statistics \
  &> "$OUT_FILE_DATASETS_STATISTICS" &

poetry run python scripts/run_evaluation_study_graph_based_impression_aware_recommenders.py \
  --create_datasets \
  --include_impressions \
  --include_impressions_frequency \
  --print_evaluation_results \
  &> "$OUT_FILE_GRAPH" &

poetry run python scripts/setup_dask_local_cluster.py \
  --setup_dask_local_cluster \
  &> "$OUT_FILE_DASK" &

# poetry run python scripts/run_evaluation_study_impression_aware_recommenders.py \
#   --create_datasets \
#   --include_baselines \
#   --include_impressions_heuristics \
#   --include_impressions_reranking \
#   --include_impressions_profile \
#   --print_evaluation_results \
#   &> "$OUT_FILE_EVALUATION"

#poetry run python scripts/run_evaluation_study_impression_aware_recommenders.py \
#  --create_datasets \
#  --include_baselines \
#  --include_impressions_heuristics \
#  --include_impressions_reranking \
#  --include_ablation_impressions_reranking \
#  --include_signal_analysis_reranking \
#  --include_signal_analysis \
#  --include_impressions_profile \
#  --print_evaluation_results \
#  &> "$OUT_FILE_EVALUATION"

#poetry run python main.py \
#  --create_datasets \
#  --include_baselines \
#  --include_folded \
#  --include_impressions_reranking \
#  --include_impressions_profile \
#  --include_ablation_impressions_reranking \
#  --compute_statistical_tests \
#  --compute_confidence_intervals \
#  --print_evaluation_results \
#  --send_email \
#  &> "$OUT_FILE"
