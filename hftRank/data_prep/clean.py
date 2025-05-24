import json
import os
from pathlib import Path

import pandas as pd


with open('./settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"]
dadv_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"]
adv_group = settings["READ_FILE_SETTINGS"]["ADV_GROUP"]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
additional_columns = settings["READ_FILE_SETTINGS"]["ADDITIONAL_COLUMNS"]

seed = 43


def Clean():
    """
    This function prepares the test and train data for the LLM. It does the following:
    """

    test_data = pd.read_csv(f'../Hidden-FaiReR-TabLLM/Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv')
    train_data = pd.read_csv(
        f'../Hidden-FaiReR-TabLLM/Datasets/{experiment_name}/Training/Training_{experiment_name}.csv')
    # convert name column from Last, First to First Last
    train_data['Name'] = train_data['Name'].str.split(', ').str[::-1].str.join(' ')

    total_data = pd.read_csv(f'../Hidden-FaiReR-TabLLM/Datasets/{experiment_name}/{experiment_name}_data.csv')
    total_data['Name'] = total_data['Name'].str.split(', ').str[::-1].str.join(' ')

    if experiment_name == 'BostonMarathon':
        # search for score column in data and for each row in test_data, replace the score with the corresponding
        # score in data if the name matches
        # drop the score column in test_data and train_data
        test_data = test_data.drop(['score'], axis=1)
        train_data = train_data.drop(['score'], axis=1)
        test_data = test_data.merge(total_data[['Name', 'score']], on='Name', how='left')
        train_data = train_data.merge(total_data[['Name', 'score']], on='Name', how='left')

    test_data = test_data.dropna(subset=['doc_id', 'Name', protected_feature, score_column])
    train_data = train_data.dropna(subset=['doc_id', 'Name', protected_feature, score_column])

    """ DIRECTORY MANAGEMENT """
    data_path = Path("../Hidden-FaiReR-TabLLM/Datasets/" + experiment_name + '/')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # sort test and train data based on score column
    test_data = test_data.sort_values(by=score_column, ascending=False)
    train_data = train_data.sort_values(by=score_column, ascending=False)

    # save test and train data to csv
    test_data.to_csv(str(data_path) + f"/{experiment_name}_test_data.csv", index=False)

    # prepare test data for LLM
    demo_dict = {0: adv_group, 1: dadv_group}
    test_data[protected_feature] = test_data[protected_feature].replace(demo_dict)
    test_data.to_csv(str(data_path) + f"/{experiment_name}_test_data_for_LLM.csv", index=False)

    # prepare train data for LLM
    train_data[protected_feature] = train_data[protected_feature].replace(demo_dict)
    train_data.to_csv(str(data_path) + f"/{experiment_name}_train_data_for_LLM.csv", index=False)

    # clean train data for training (not LLM)
    train_data = train_data.drop(['doc_id'], axis=1)
    train_data.to_csv(str(data_path) + f"/{experiment_name}_train_data.csv", index=False)
    sub_clean()


def sub_clean():
    train_data = pd.read_csv( f'../Hidden-FaiReR-TabLLM/Datasets/{experiment_name}/Training/Training_{experiment_name}.csv')
    test_data = pd.read_csv(f'../Hidden-FaiReR-TabLLM/Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv')
    number_to_gender_dict = {0: adv_group, 1: dadv_group}
    train_data['Gender'] = train_data['Gender'].replace(number_to_gender_dict)
    test_data['Gender'] = test_data['Gender'].replace(number_to_gender_dict)
    test_data.to_csv(f'../Hidden-FaiReR-TabLLM/Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv', index=False)
    train_data.to_csv(f'../Hidden-FaiReR-TabLLM/Datasets/{experiment_name}/Training/Training_{experiment_name}.csv', index=False)


#########################################
# the code below was used to clean the COMPAS dataset. This was to ensure the truly disadvantaged group was reflected in the sets
def reverse_scores(scores):
    reversed_scores = [-score for score in scores]
    return reversed_scores


def format_compas():
    compas_sex = pd.read_csv(f'../Hidden-FaiReR-TabLLM/Datasets/{experiment_name}/{experiment_name}_data.csv')
    compas_sex['new_raw_score'] = reverse_scores(compas_sex['raw_score'])

    # arrange in descending order of scores
    compas_sex.sort_values(by='raw_score', ascending=True)
    compas_sex.drop(['raw_score'], axis=1, inplace=True)
    compas_sex = compas_sex.rename(columns={'new_raw_score': 'raw_score'})
    compas_sex.sort_values(by='raw_score', ascending=False, inplace=True)
    # compas_sex['new_id'] = range(1, len(compas_sex['doc_id']) + 1)
    # compas_sex = compas_sex.drop(columns='new_id', axis=1)
    compas_sex.to_csv(f'../Hidden-FaiReR-TabLLM/Datasets/{experiment_name}/{experiment_name}_data.csv', index=False)
    return
