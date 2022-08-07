#!/bin/bash -l
# This is the script ran by cron. It uses the shebang with the option -l so it loads the profile from .bashrc or profile
# or how it is called in the system. This cronjob is supposed to be run everytime an instance is
# rebooted / turned on, therefore, it outputs all stdout and stderr to a file with the current date.
OUTPUTS_FOLDER="outputs"
DATE=$(date '+%Y%m%d_%H%M%S')
EXPERIMENT="ALL"
WORKING_FOLDER="/fbpm/project-impressions/impressions-evaluation"
OUT_FILE="$WORKING_FOLDER/$OUTPUTS_FOLDER/$DATE-$EXPERIMENT.txt"

if [ ! -d $WORKING_FOLDER/$OUTPUTS_FOLDER ]
then
    mkdir $WORKING_FOLDER/$OUTPUTS_FOLDER
fi

cd $WORKING_FOLDER || exit
poetry run python scripts/main.py \
  --create_datasets \
  --include_baselines \
  --include_folded \
  --include_impressions_reranking \
  --include_impressions_profile \
  --include_ablation_impressions_reranking \
  --send_email \
  &> "$OUT_FILE"