# this script prepares the data (already split) for the LLM ranking. It contains several functions that can be used to
# prep the data for different experiments.
import ast

import chardet
import json
import os.path
import random
import shutil
import time
from .detconstsort import detconstsort as dcs, infer_with_detconstsort as iwd
import numpy as np
import pandas as pd

#from .. import adv_group
from ..data_analysis import calculate_metrics as cm
from ..data_analysis import skew as sk
from ..data_viz import plot_skew

start_time = time.time()

with open('./settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
rank_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"]
protected_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"].lower()
non_protected_group = settings["READ_FILE_SETTINGS"]["ADV_GROUP"].lower()

gender_dict = {}
gender_dict = {'female': 1 if protected_group.lower() == 'female' else 0,
               'male': 0 if protected_group.lower() == 'female' else 1}

# number_to_gender_dict = {1: protected_group, 0: non_protected_group}

random_sate = 123


def create_shots(size=50, sh=1, create_fair_data=False, create_fair_data_for_reranking=False):
    """ this function creates the shots for LLM. The data is ranked fairly used DetConstSort"""

    # do not create data for shot 0
    test_data = pd.read_csv(f"./Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv")
    train_data = pd.read_csv(f"./Datasets/{experiment_name}/Training/Training_{experiment_name}.csv")
    # use gender_dict to re-fill gender column
    # train_data[protected_feature] = train_data[protected_feature].replace(number_to_gender_dict)
    train_data_1 = train_data[train_data[protected_feature] == protected_group]
    train_data_0 = train_data[train_data[protected_feature] == non_protected_group]
    if sh == 0:
        return

    # create Scored data folder if it does not exist
    scored_save_path = f"./Datasets/{experiment_name}/Shots/size_{size}/Scored/shot_{sh}"
    if not os.path.exists(scored_save_path):
        os.makedirs(scored_save_path)

    # split data into 2 groups if using sex as a protected feature and randomize them
    if protected_feature == 'Gender':

        ave_exp = 0.8
        ave_exp_dict = {'LOAN': 0.7, 'NBAWNBA': 0.7, 'COMPASSEX': 0.7, 'BostonMarathon': 0.7}
        # if experiment_name == 'COMPASSEX':
        #     ave_exp = 0.9

        ave_exp_limit = ave_exp_dict[experiment_name]

        while ave_exp >= ave_exp_limit:
            train_df = pd.DataFrame(columns=test_data.columns)
            # generate unique random indices
            random_indices_1 = random.sample(range(len(train_data_1)), size // 2)
            random_indices_0 = random.sample(range(len(train_data_0)), size // 2)
            # select the rows with the random indices
            train_df = pd.concat([train_df, train_data_1.iloc[random_indices_1]]).reset_index(drop=True)
            train_df = pd.concat([train_df, train_data_0.iloc[random_indices_0]]).reset_index(drop=True)

            # sort by score and reset index
            # gt_df = train_df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
            gt_df = train_df.sort_values(by=score_column, ascending=False)

            # save the data before adding fairness

            gt_df.to_csv(f"{scored_save_path}/ranked_data_rank_size_{size}_shot_{sh}.csv", index=False)
            gt_df.to_csv(f"{scored_save_path}/ground_truth_rank_size_{size}_shot_{sh}.csv", index=False)
            # scored_dir = f"../Datasets/{experiment_name}/Train/Scored"
            cm.calculate_metrics_per_shot_llm(scored_save_path, rank_size=size)
            # check exposure

            result_folder = f"{scored_save_path}".replace("Datasets", "Results")
            result = pd.read_csv(f"{result_folder}/metrics.csv")
            skew_path = scored_save_path.replace("Datasets", "Results")
            if not os.path.exists(skew_path):
                os.makedirs(skew_path)
            # plot skew graph
            plot_skew(f"{skew_path}/skew.csv", size=size)
            # get the Average Exposure
            ave_exp = result["Average Exposure"].iloc[0]
    # save current random indices, random_indices_1 and random_indices_0
    with open(f"{scored_save_path}/random_indices.json", 'w') as f:
        json.dump({'random_indices_1': random_indices_1, 'random_indices_0': random_indices_0}, f)

    if create_fair_data_for_reranking:
        fair_rank_save_path = f"./Datasets/{experiment_name}/Shots/size_{size}/Fair_Reranking/"
        if not os.path.exists(fair_rank_save_path):
            os.makedirs(fair_rank_save_path)

        p = [{protected_group: round(i, 1), non_protected_group: round(1 - i, 1)} for i in
             [x / 10 for x in range(6, 10)]]
        # p = [{protected_group:0., non_protected_group:0.5}]
        # rank using DetConstSort
        fair_rank_save_path_with_shot = f"{fair_rank_save_path}/shot_{sh}"
        for pe in p:
            iwd(f"{scored_save_path}/ranked_data_rank_size_{size}_shot_{sh}.csv",
                post_process=True, p_value=pe)
            with open(f"{fair_rank_save_path}/shot_{sh}/ranked_data_rank_size_{size}_shot_{sh}.csv", "rb") as f:
                result = chardet.detect(f.read(10000))
            fair_df = pd.read_csv(f"{fair_rank_save_path}/shot_{sh}/ranked_data_rank_size_{size}_shot_{sh}.csv",
                                  encoding=result["encoding"])
            gt_fair_df = fair_df.sort_values(by='predictions', ascending=False).reset_index(drop=True)
            gt_fair_df.to_csv(f"{fair_rank_save_path}/shot_{sh}/ground_truth_rank_size_{size}_shot_{sh}.csv")
            cm.calculate_metrics_per_shot_llm(fair_rank_save_path_with_shot, rank_size=size)
            # get the Average Exposure
            result_folder = fair_rank_save_path_with_shot.replace("Datasets", "Results")
            result = pd.read_csv(f"{result_folder}/metrics.csv")
            # get the Average Exposure
            avg_exp = result["Average Exposure"].iloc[0]
            if 0.97 <= avg_exp <= 1.03:
                # if avg_exp == 1.0:
                # save pe and avg_exp to a file
                with open(f"{result_folder}/pe_avg_exp.csv", 'a') as f:
                    f.write(f"p={pe}, average_exposure={avg_exp}\n")
                # end the loop
                break

        fair_skew_path = fair_rank_save_path_with_shot.replace("Datasets", "Results")
        if not os.path.exists(fair_skew_path):
            os.makedirs(fair_skew_path)
        # plot skew graph
        plot_skew(f"{fair_skew_path}/skew.csv", size=size)


def create_test_data(size=50, number=5, equal_distribution=False):
    """ this function creates n unique test data for the shots for LLM"""
    test_data = pd.read_csv(f"./Datasets/{experiment_name}/{experiment_name}_test_data_for_LLM.csv")
    test_df = test_data.sample(n=size, random_state=42)
    # create Test folder if it does not exist
    test_folder = f"./Datasets/{experiment_name}/Tests"
    os.makedirs(test_folder, exist_ok=True)

    test_file_path = f"{test_folder}/rank_size_{size}.csv"

    # check if the file already exists to prevent overwriting
    if os.path.exists(test_file_path):
        print(f"Warning: {test_file_path} already exists. The file will not be overwritten.")
    else:
        test_df.to_csv(test_file_path, index=False)
        print(f"Test data saved to {test_file_path}")


def check_skew(df):
    position = len(df)
    skew_0 = 0
    skew_1 = 0

    # k = len(df)
    # sort by score
    df = df.sort_values(by=score_column, ascending=False)

    # get the ranked unique ids and group ids
    if 'Student ID' in df.columns:
        ranked_unique_ids = df["Student ID"].tolist()
    else:
        ranked_unique_ids = df["doc_id"].tolist()
    ranked_unique_ids = [int(id) for id in ranked_unique_ids]
    group_ids = df[protected_feature].tolist()
    group_ids = [1 if id == protected_group else 0 for id in group_ids]
    print('group_ids:', group_ids)

    # measure the skew, if the skew for protected data is greater 1.0 or skew for non-protected data is less
    # Ensure pos is within bounds using try-except
    try:
        # Calculate skew
        skew_0 = sk(np.array(ranked_unique_ids), np.array(group_ids), 0, position)
        skew_1 = sk(np.array(ranked_unique_ids), np.array(group_ids), 1, position)
        print('current length:', len(df))

        print('skew_0:', skew_0)
        print('skew_1:', skew_1)
    except Exception as e:
        # Handle the specific exception for out-of-bounds error
        if "Pos is not within the bounds of the arrays" in str(e):
            print(f"Skipping iteration {len(df)} due to error: {e}")
    return skew_0, skew_1


def map_to_target(num):
    # Map 0-9 to 1-4 cyclically
    target_values = [1, 2, 3, 4]
    return target_values[num % len(target_values)]

def remove_duplicates(df):
    """
    This function removes duplicates from the dataframe
    :param df:
    :return:
    """
    # Check for duplicates based on 'doc_id' column
    duplicates = df.duplicated(subset=['Name'], keep='first')
    # If duplicates exist, drop them
    if duplicates.any():
        df = df[~duplicates]
        print(f"Removed {duplicates.sum()} duplicate rows based on 'doc_id'.")
    return df


def create_test_data_for_reranking(size=50, number=5, equal_distribution=True):
    """Create unique test data for LLM reranking with no duplicates and controlled group distribution."""
    test_data = pd.read_csv(f"./Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv")
    full_size = len(test_data)

    if size == full_size:
        subset_data = test_data.copy()
    else:
        test_data_1 = test_data[test_data[protected_feature] == protected_group].sort_values(by=score_column, ascending=False)
        test_data_0 = test_data[test_data[protected_feature] == non_protected_group].sort_values(by=score_column, ascending=False)

        top_k = 4 if size == 10 else 10
        test_data_0_top_k = test_data_0.head(top_k)

        test_data_0 = test_data_0.iloc[top_k:].sample(frac=1, random_state=2).reset_index(drop=True)
        test_data_1 = test_data_1.iloc[:-top_k].sample(frac=1, random_state=2).reset_index(drop=True)

        check_size_0 = check_size_1 = size // 2 if equal_distribution else len(test_data_0)

    for i in range(number):
        save_dir = f"./Datasets/{experiment_name}/Tests/size_{size}/Reranking_{i}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/ranked_data_rank_size_{size}_{i}.csv"


        if size == full_size:
            new_test_df = test_data.copy()
        else:
            used_indices_0, used_indices_1 = set(), set()
            new_test_df = pd.DataFrame(columns=test_data.columns)

            # Initial top-k sample
            idx0 = random.choice([i for i in test_data_0_top_k.index if i not in used_indices_0])
            new_test_df = pd.concat([new_test_df, test_data_0_top_k.loc[[idx0]]], ignore_index=True)
            used_indices_0.add(idx0)

            while True:
                idx1 = random.choice(test_data_1.index)
                if idx1 not in used_indices_1:
                    new_test_df = pd.concat([new_test_df, test_data_1.loc[[idx1]]], ignore_index=True)
                    used_indices_1.add(idx1)
                    if valid_skew(*check_skew(new_test_df)):
                        break
                    new_test_df = new_test_df.drop(index=new_test_df.index[-1])

            for l in range(check_size_1 - 1):
                # Add from non-protected
                while True:
                    idx0 = random.choice(test_data_0.index)
                    if idx0 not in used_indices_0:
                        new_test_df = pd.concat([new_test_df, test_data_0.loc[[idx0]]], ignore_index=True)
                        used_indices_0.add(idx0)
                        if valid_skew(*check_skew(new_test_df)):
                            break
                        new_test_df = new_test_df.drop(index=new_test_df.index[-1])

                # Add from protected
                while True:
                    idx1 = random.choice(test_data_1.index)
                    if idx1 not in used_indices_1:
                        new_test_df = pd.concat([new_test_df, test_data_1.loc[[idx1]]], ignore_index=True)
                        used_indices_1.add(idx1)
                        if valid_skew(*check_skew(new_test_df)):
                            break
                        new_test_df = new_test_df.drop(index=new_test_df.index[-1])
        sample_ids = {
            "test_data_0_top_k": list(used_indices_0.intersection(test_data_0_top_k.index)),
            "test_data_0": list(used_indices_0.difference(test_data_0_top_k.index)),
            "test_data_1": list(used_indices_1)
        }
        with open(f"{save_dir}/sample_ids.json", "w") as f:
            json.dump(sample_ids, f, indent=2)

        # Final cleanup and saves
        new_test_df = new_test_df.drop_duplicates(subset=['doc_id']).sort_values(by=score_column, ascending=False)
        new_test_df['doc_id'] = range(1, len(new_test_df) + 1)
        new_test_df.to_csv(save_path, index=False)

        gt_path = f"{save_dir}/ground_truth_rank_size_{size}_{i}.csv"
        new_test_df.to_csv(gt_path, index=False)

        # Additional metrics, skew plots, p-value extraction
        cm.calculate_metrics_per_shot_llm(save_dir, rank_size=size)

        skew_path = save_dir.replace("Datasets", "Results")
        os.makedirs(skew_path, exist_ok=True)
        plot_skew(f"{skew_path}/skew.csv")

        p = None
        if size != full_size:
            shot_value = map_to_target(i)
            p_folder = save_dir.replace("Datasets", "Results").replace("Tests", "Shots").replace(f"Reranking_{i}", f"Fair_Reranking/shot_{shot_value}")
            with open(f"{p_folder}/pe_avg_exp.csv", 'r') as f:
                s = f.readlines()[0]
                p = ast.literal_eval(s[s.index("{"):s.index("}") + 1])
                with open(f"{save_dir}/p_value_{i}.txt", 'w') as g:
                    g.write(str(p))

        iwd(save_path, post_process=True, test_data=True, p_value=p)

        # create DetConstSort folder if it does not exist
        dcs_save_path = f"./Datasets/{experiment_name}/Ranked/DetConstSort"
        if not os.path.exists(dcs_save_path):
            os.makedirs(dcs_save_path)

        ranked_path = f"{dcs_save_path}/option_NAD/inf_NAD/prompt_NAD/rank_size_{size}/shot_NAD/ranked_data_rank_size_{size}_{i}.csv"
        ranked_data = pd.read_csv(ranked_path).sort_values(by='GT_score', ascending=False)
        ranked_data.to_csv(f"{ranked_path}_ground_truth.csv", index=False)

    flatten_directories(f"./Datasets/{experiment_name}/Tests/size_{size}")

    init_dir = f"./Datasets/{experiment_name}/Ranked/Initial/option_NA/inf_NA/prompt_NA/rank_size_{size}/shot_NA"
    os.makedirs(init_dir, exist_ok=True)
    for file in os.listdir(f"./Datasets/{experiment_name}/Tests/size_{size}"):
        src = os.path.join(f"./Datasets/{experiment_name}/Tests/size_{size}", file)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(init_dir, file))

def create_tests_like_shots(size=50, number=1):
    """ this function creates the shots for LLM. The data is ranked fairly used DetConstSort"""

    test_data = pd.read_csv(f"./Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv")
    # use gender_dict to re-fill gender column
    # train_data[protected_feature] = train_data[protected_feature].replace(number_to_gender_dict)
    test_data_1 = test_data[test_data[protected_feature] == protected_group]
    test_data_0 = test_data[test_data[protected_feature] == non_protected_group]
    for i in range(number):
        save_dir = f"./Datasets/{experiment_name}/Tests/size_{size}/Reranking_{i}"
        os.makedirs(save_dir, exist_ok=True)
        #save_path = f"{save_dir}/ranked_data_rank_size_{size}_{i}.csv"

        # split data into 2 groups if using sex as a protected feature and randomize them
        if protected_feature == 'Gender':

            ave_exp = 0.8
            ave_exp_dict = {'LOAN': 0.7, 'NBAWNBA': 0.7, 'COMPASSEX': 0.7, 'BostonMarathon': 0.7}
            # if experiment_name == 'COMPASSEX':
            #     ave_exp = 0.9

            ave_exp_limit = ave_exp_dict[experiment_name]

            while ave_exp >= ave_exp_limit:
                test_df = pd.DataFrame(columns=test_data.columns)
                # generate unique random indices
                random_indices_1 = random.sample(range(len(test_data_1)), size // 2)
                random_indices_0 = random.sample(range(len(test_data_0)), size // 2)
                # select the rows with the random indices
                test_df = pd.concat([test_df, test_data_1.iloc[random_indices_1]]).reset_index(drop=True)
                test_df = pd.concat([test_df, test_data_0.iloc[random_indices_0]]).reset_index(drop=True)

                # sort by score and reset index
                # gt_df = train_df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
                gt_df = test_df.sort_values(by=score_column, ascending=False)


                gt_df.to_csv(f"{save_dir}/ground_truth_rank_size_{size}_{i}.csv")
                gt_df.to_csv(f"{save_dir}/ranked_data_rank_size_{size}_{i}.csv")
                # scored_dir = f"../Datasets/{experiment_name}/Train/Scored"
                cm.calculate_metrics_per_shot_llm(save_dir, rank_size=size)
                # check exposure


                skew_path = save_dir.replace("Datasets", "Results")
                if not os.path.exists(skew_path):
                    os.makedirs(skew_path)
                # plot skew graph
                plot_skew(f"{skew_path}/skew.csv", size=size)
                # read exposure
                result_folder = f"{skew_path}".replace("Datasets", "Results")
                result = pd.read_csv(f"{result_folder}/metrics.csv")
                # get the Average Exposure
                ave_exp = result["Average Exposure"].iloc[0]
                print('ave_exp:', ave_exp)

            # # save current random indices, random_indices_1 and random_indices_0
                with open(f"{save_dir}/random_indices.json", 'w') as f:
                    json.dump({'random_indices_1': random_indices_1, 'random_indices_0': random_indices_0}, f)
    flatten_directories(f"./Datasets/{experiment_name}/Tests/size_{size}")

    init_dir = f"./Datasets/{experiment_name}/Ranked/Initial/option_NA/inf_NA/prompt_NA/rank_size_{size}/shot_NA"
    os.makedirs(init_dir, exist_ok=True)
    for file in os.listdir(f"./Datasets/{experiment_name}/Tests/size_{size}"):
        src = os.path.join(f"./Datasets/{experiment_name}/Tests/size_{size}", file)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(init_dir, file))


def valid_skew(skew_0, skew_1):
    return skew_0 >= 1 and skew_1 <= 1

def metric_tests(df):
    """
    This function calculates the average exposure and plots skew for data when called
    :return:
    """

def flatten_directories(parent_dir):
    # Iterate through all subdirectories in the parent directory
    for subdir in [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]:
        subdir_path = os.path.join(parent_dir, subdir)

        # Move all files and subdirectories to the parent directory
        for item in os.listdir(subdir_path):
            item_path = os.path.join(subdir_path, item)
            dest_path = os.path.join(parent_dir, item)

            # If destination already exists, remove it first
            if os.path.exists(dest_path):
                os.remove(dest_path)

            shutil.move(item_path, parent_dir)

        # Remove the now-empty subdirectory
        os.rmdir(subdir_path)



def get_inferred_data():
    BTN_data = pd.read_csv(f"./Datasets/{experiment_name}/Inferred/BTN/(Default=1)BTN_Inferred_{experiment_name}.csv")
    NMSOR_data = pd.read_csv(
        f"./Datasets/{experiment_name}/Inferred/NMSOR/(Default=1)NMSOR_Inferred_{experiment_name}.csv")
    GAPI_data = pd.read_csv(
        f"./Datasets/{experiment_name}/Inferred/GAPI/(Default=1)GAPI_Inferred_{experiment_name}.csv")
    test_data = pd.read_csv(f"./Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv")
    # create BTN, NMSOR and GAPI columns in test_data and fill based on names
    # test_data['BTN'] = 0
    # test_data['NMSOR'] = 0
    # test_data['GAPI'] = 0
    number_to_gender_dict = {1: protected_group, 0: non_protected_group}
    for index, row in test_data.iterrows():
        if row['Name'] in BTN_data['Name'].values:
            test_data.at[index, 'BTN'] = BTN_data[BTN_data['Name'] == row['Name']]['InferredGender'].values[0]
        if row['Name'] in NMSOR_data['Name'].values:
            test_data.at[index, 'NMSOR'] = NMSOR_data[NMSOR_data['Name'] == row['Name']]['InferredGender'].values[0]
        if row['Name'] in GAPI_data['Name'].values:
            test_data.at[index, 'GAPI'] = GAPI_data[GAPI_data['Name'] == row['Name']]['InferredGender'].values[0]
    test_data['BTN'] = test_data['BTN'].replace(number_to_gender_dict)
    test_data['NMSOR'] = test_data['NMSOR'].replace(number_to_gender_dict)
    test_data['GAPI'] = test_data['GAPI'].replace(number_to_gender_dict)
    test_data.to_csv(f"./Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv", index=False)


def Prep(size):
    number = 10
    print('size:', size)
    get_inferred_data()
    for shot in shots:
        create_shots(size, shot, create_fair_data_for_reranking=True)
    print('Done creating shots')
    # create_test_data_for_reranking(size=size, number=number, equal_distribution=True)
    create_tests_like_shots(size=size, number=number)


#################################### For reranking #################################################

# 3. Create shots for LLM reranking for fairness
# for shot in shots:
#     prep_LLM_data(20, shot, create_fair_data_for_reranking=True)
#
# # 4. Create test data for LLM reranking for fairness
# # create_non_unique_test_data_for_reranking_for_fairness(20)
# # this is the most recent one. It included the DetConstSort ranking and Initial ranking for the test data
# create_test_data_for_reranking(size=20, number=10, equal_distribution=True)
