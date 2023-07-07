####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
import os

from impressions_evaluation.experiments import commons

DIR_TRAINED_MODELS_IMPRESSION_AWARE = os.path.join(
    commons.DIR_TRAINED_MODELS,
    "impression_aware",
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)

commons.FOLDERS.add(DIR_TRAINED_MODELS_IMPRESSION_AWARE)
