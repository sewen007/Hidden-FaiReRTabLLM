import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from hftRank.data_analysis import kendall_tau, NDCG, NDKL, avgExp, skew

# from .data_analysis import kendall_tau, NDCG, NDKL, avgExp, skew

# print working directory

with open('./settings.json', 'r') as f:
    settings = json.load(f)

start = time.time()

sample_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]

delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"].lower()
dadv_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"]

feature_dict = {}

if dadv_group == 'female':
    adv_group = 'male'
else:
    adv_group = 'female'

feature_dict[dadv_group] = 1
feature_dict[adv_group] = 0

test_df = pd.read_csv(f"./Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv")


def calculate_metrics_per_shot_llm(shot_path, shot='shot_0', exp_name=experiment_name, rank_size='size_3', inf_app='GAPI', option='1'):
    """
    Calculate the Kendall Tau correlation coefficient between the ground truth and the inferred rankings
    :return: Kendall Tau correlation coefficient
    """
    path = shot_path + '/'
    ranked_folder = os.listdir(path)
    protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"].lower()
    # list all ranked_files in the directory (not groundtruth)
    ranked_files = [f for f in ranked_folder if
                    f.endswith('.csv') and 'ranked' in f and 'ground_truth' not in f and str(rank_size) + '_' in f]
    print(ranked_files)
    print('rank_size = ', rank_size)
    # ground_truth_file = [f for f in ranked_folder if f.endswith('.csv') and 'ground_truth' in f and str(rank_size)+'_' in f][
    #     0]

    # create a new path by changing 'Datasets' to 'Results' in shot_path
    results_path = Path(shot_path.replace('Datasets', 'Results'))

    # check if the results path exists, if not create it
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    metrics_path = results_path / "metrics.csv"

    run_number = 0
    # open the metrics file
    with open(metrics_path, 'w', newline='') as f_metrics:
        writer = csv.writer(f_metrics)
        # write the header
        writer.writerow(["Run", "Kendall Tau", "NDKL", "Average Exposure"])  # Write the header once before the loop

        # for each file, read the inferred ranking and the ground truth ranking
        for file in ranked_files:
            file_path = path + file
            print(file_path)
            ranked_df = pd.read_csv(file_path)
            ranked_df.columns = ranked_df.columns.str.lower()
            test_df.columns = test_df.columns.str.lower()
            # if 'BTN' in file_path or 'GAPI' in file_path or 'NMSOR' in file_path or 'option_3' in file_path:
            #     # rename sex to gt_sex
            #     ranked_df = ranked_df.rename(columns={'sex': 'inferred_sex'})
            #
            #
            #     # Merge on 'name' (case-insensitive handled above)
            #     ranked_df = ranked_df.merge(test_df[['name', protected_feature.lower()]], on='name', how='left')
            #
            #     # Rename protected feature column to 'sex'
            #     ranked_df = ranked_df.rename(columns={protected_feature.lower(): 'sex'})
            #
            #
            # if 'sex' in ranked_df.columns:
            #     ranked_df.rename(columns={'sex': protected_feature}, inplace=True)
            #
            # # drop duplicates based on names
            # if 'name' in ranked_df.columns:
            #     ranked_df = ranked_df.drop_duplicates(subset=['name'])
            #     if protected_feature not in ranked_df.columns:
            #         # get corresponding column from test_df using name
            #         ranked_df = ranked_df.merge(test_df[['name', protected_feature]], on='name', how='left')
            #         ranked_df = ranked_df.drop_duplicates(subset=['name'])
            # else:
            #     ranked_df = ranked_df.drop_duplicates(subset=['Name'])
            #     if protected_feature not in ranked_df.columns:
            #         # get corresponding column from test_df using name
            #         ranked_df = ranked_df.merge(test_df[['Name', protected_feature]], on='Name', how='left')
            #         ranked_df = ranked_df.drop_duplicates(subset=['Name'])
            protected_feature = protected_feature.lower()
            if protected_feature not in test_df.columns:
                if "gender" in test_df.columns:
                    protected_feature = "gender"
                elif "sex" in test_df.columns:
                    protected_feature = "sex"
                else:
                    raise KeyError(
                        f"No column matching protected feature found in test_df. Available: {list(test_df.columns)}")

            if 'btn' in file_path or 'gapi' in file_path or 'nmsor' in file_path or 'option_3' in file_path:
                # rename sex to inferred_sex
                ranked_df = ranked_df.rename(columns={'sex': 'inferred_sex'})

                # Merge with correct protected feature
                ranked_df = ranked_df.merge(test_df[['name', protected_feature]], on='name', how='left')

                # Normalize to 'sex'
                ranked_df = ranked_df.rename(columns={protected_feature: 'sex'})

            # Standardize column naming
            if 'sex' in ranked_df.columns:
                ranked_df.rename(columns={'sex': 'gender'}, inplace=True)  # or protected_feature if you want dynamic
            print('ranked_df', ranked_df)
            if 1 not in ranked_df[protected_feature].unique():
                print(ranked_df[protected_feature])
                # if 'ListNet' not in exp_name:
                ranked_df[protected_feature] = ranked_df[protected_feature].map(feature_dict)


            print('df', ranked_df)
            if score_column in ranked_df.columns:
                ground_truth_df = ranked_df.sort_values(by=score_column, ascending=False)
            elif 'score' in ranked_df.columns:
                ground_truth_df = ranked_df.sort_values(by='score', ascending=False)
            elif 'score_x' in ranked_df.columns:
                ground_truth_df = ranked_df.sort_values(by='score_x', ascending=False)
            else:  # if ZFYA column exists
                ground_truth_df = ranked_df.sort_values(by='GT_score', ascending=False)
            # run_number = file.split('_')[-1].split('.')[0]
            if 'Student ID' in ranked_df.columns:
                ranked_unique_ids = ranked_df["Student ID"].tolist()
            else:
                if 'ID' in ranked_df.columns:
                    ranked_unique_ids = ranked_df["ID"].tolist()
                else:
                    ranked_unique_ids = ranked_df["doc_id"].tolist()

            if 'ID' in ground_truth_df.columns:
                gt_unique_ids = ground_truth_df["ID"].tolist()
            else:
                gt_unique_ids = ground_truth_df["doc_id"].tolist()
            if protected_feature != '':
                group_ids = ranked_df[protected_feature].tolist()

            # Convert items in the lists to ints
            gt_unique_ids = [int(id) for id in gt_unique_ids]
            ranked_unique_ids = [int(id) for id in ranked_unique_ids]
            if protected_feature != '':
                #print('group ids', group_ids)
                group_ids = [id for id in group_ids]

            """ CALCULATE AND STORE FAIRNESS METRICS AND KENDALL TAU"""
            kT_corr = kendall_tau(gt_unique_ids, ranked_unique_ids)
            #print('unique ids', gt_unique_ids)
            if protected_feature != '':
                #print('lenght of group ids', len(group_ids))
                #print('length of gt_unique ids', len(gt_unique_ids))
                ndkl = NDKL(np.array(gt_unique_ids), np.array(group_ids))
                avg_exp = avgExp.avg_exp(np.array(ranked_unique_ids), np.array(group_ids))
                exp_ratio = avg_exp[1] / avg_exp[0]
                # print(gt_unique_ids, ranked_unique_ids)

            else:
                ndkl = ""
                avg_exp = ""
                exp_ratio = ""
            writer.writerow([run_number, kT_corr[0], ndkl, exp_ratio])

            """ CALCULATE AND STORE NDCG"""
            print("Calculating NDCG...")
            if 'ZFYA' in ranked_df.columns:
                GT_score = np.array(ranked_df["ZFYA"])
            elif 'GT_score' in ranked_df.columns:
                GT_score = np.array(ranked_df["GT_score"])
            elif 'score' in ranked_df.columns:
                GT_score = np.array(ranked_df["score"])
            elif score_column in ranked_df.columns:
                GT_score = np.array(ranked_df[score_column])
            elif 'score_x' in ranked_df.columns:
                if ranked_df['score_x'].all() == ranked_df['score_y'].all():
                    GT_score = np.array(ranked_df['score_x'])
            else:
                GT_score = np.array(ranked_df['ZFYA_x'])

            GT_score_normalized = (GT_score - np.min(GT_score)) / (np.max(GT_score) - np.min(GT_score))

            ndcg_path = results_path / "ndcg.csv"
            skew_path = results_path / "skew.csv"
            ndcg_data = []
            skew_data = []
            for i in range(1, len(ranked_df) + 1):
                ndcg = NDCG(np.array(ranked_df.iloc[:, 0]), GT_score_normalized, i)
                ndcg_data.append([i, ndcg])
                # calculate skew
            for i in range(1, len(ranked_df) + 1):
                #print('pos = ', i)
                skew_0 = skew(np.array(ranked_unique_ids), np.array(group_ids), 0, i)
                skew_1 = skew(np.array(ranked_unique_ids), np.array(group_ids), 1, i)
                skew_data.append([i, skew_0, skew_1])

            if run_number == 0:
                ndcg_header = ["Position"] + [f"NDCG_{number}" for number in range(1, int(run_number) + 1)]

                # only write on the NDCG_1 column
                with open(ndcg_path, 'w') as f_ndcg:
                    print("Writing to NDCG csv.")
                    writer_ndcg = csv.writer(f_ndcg)
                    # write the header
                    writer_ndcg.writerow(ndcg_header)

                    # write the data
                    writer_ndcg.writerows(ndcg_data)

                skew_header = ["Position", "Group_0", "Group_1"]
                with open(skew_path, 'w') as f_skew:
                    print("Writing to Skew csv.")
                    writer_skew = csv.writer(f_skew)
                    # write the header
                    writer_skew.writerow(skew_header)

                    # write the data
                    writer_skew.writerows(skew_data)
            else:
                ndcg_df = pd.read_csv(ndcg_path)
                print('ndcg_path', ndcg_path)
                print('ndcg_path', ndcg_path)
                print('ndcg df', ndcg_df)
                ndcg_df[f'NDCG_{run_number}'] = [item[1] for item in ndcg_data]
                ndcg_df.to_csv(ndcg_path, index=False)

                skew_df = pd.read_csv(skew_path)
                skew_df["Group_0"] = [item[1] for item in skew_data]
                skew_df["Group_1"] = [item[2] for item in skew_data]
                skew_df.to_csv(skew_path, index=False)
            run_number += 1


def CalculateResultMetrics(meta_exp='meta-llama/Meta-Llama-3-8B-Instruct', size=50):
    folder = Path(f"./Datasets/{experiment_name}/Ranked")
    # Get the list of experiments
    experiments = [f for f in os.listdir(folder) if os.path.isdir(folder / f)]
    print("Experiments:", experiments)
    for experiment in experiments:
        print("Calculating metrics for experiment", experiment)
        if 'llama' in experiment:
            experiment = meta_exp
        if 'deepseek' in experiment:
            experiment = 'deepseek-api/API'
        experiment_path = folder / experiment
        options = [f for f in os.listdir(experiment_path) if os.path.isdir(experiment_path / f)]
        for option in options:
            print("Calculating metrics for option", option, "in experiment", experiment)
            option_path = experiment_path / option
            inf_apps = [f for f in os.listdir(option_path) if os.path.isdir(option_path / f)]
            for inf_app in inf_apps:
                print("Calculating metrics for inf_app", inf_app, "in experiment", experiment, "and option", option)
                inf_app_path = option_path / inf_app
                prompts = [f for f in os.listdir(inf_app_path) if os.path.isdir(inf_app_path / f)]
                for prompt in prompts:
                    print("Calculating metrics for prompt", prompt, "in experiment", prompt)
                    prompt_path = inf_app_path / prompt
                    # Get the list of sizes
                    sizes = [f for f in os.listdir(prompt_path) if os.path.isdir(prompt_path / f) if str(size) in f]
                    for size in sizes:
                        print("Calculating metrics for size", size, "in experiment", experiment)
                        size_path = prompt_path / size
                        # Get the list of shots
                        shots = [f for f in os.listdir(size_path) if os.path.isdir(size_path / f)]
                        for shot in shots:
                            print("Calculating metrics for shot", shot, "in experiment", experiment, "and size", size)
                            shot_path = size_path / shot
                            print("Calculating metrics for shot", shot, "in experiment", experiment, "and size", size)
                            calculate_metrics_per_shot_llm(str(shot_path), str(shot), experiment, size, inf_app, option)


def calculate_gt_metrics():
    t_t = ['Tests', 'Train']
    for t in t_t:
        # get test files
        test_files = [f for f in os.listdir(f"./Datasets/{experiment_name}/{t}") if f.endswith('.csv') and '50' in f]
        # get size

        for file in test_files:
            size = (test_files[0].split('_')[2]).split('.')[0]
            # read file
            test_data = pd.read_csv(f"./Datasets/{experiment_name}/{t}/{file}")
            # sort by score
            test_data = test_data.sort_values(by=score_column, ascending=False)
            # apply the feature_dict to the protected_feature column
            test_data[protected_feature] = test_data[protected_feature].map(feature_dict)
            # create results folder
            results_path = Path(f"./Results/{experiment_name}/{t}/{size}")
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            metrics_path = results_path / "metrics.csv"
            ndcg_path = results_path / "ndcg.csv"

            gt_unique_ids = test_data["doc_id"].tolist()
            group_ids = test_data[protected_feature].tolist()
            kT_corr = kendall_tau(gt_unique_ids, gt_unique_ids)
            ndkl = NDKL(np.array(gt_unique_ids), np.array(group_ids))
            avg_exp = avgExp.avg_exp(np.array(gt_unique_ids), np.array(group_ids))
            if len(avg_exp) == 1:
                exp_ratio = avg_exp[0]
            else:
                exp_ratio = avg_exp[1] / avg_exp[0]

            # open the metrics file
            with open(metrics_path, 'w', newline='') as f_metrics:
                writer = csv.writer(f_metrics)
                # write the header
                writer.writerow(["Kendall Tau", "NDKL", "Average Exposure"])
                writer.writerow([kT_corr[0], ndkl, exp_ratio])

            # open the NDCG file
            with open(ndcg_path, 'w', newline='') as f_ndcg:
                writer_ndcg = csv.writer(f_ndcg)
                writer_ndcg.writerow(["Position", "NDCG"])
                GT_score = np.array(test_data.iloc[:, -1])
                GT_score_normalized = (GT_score - np.min(GT_score)) / (np.max(GT_score) - np.min(GT_score))
                ndcg_data = []
                for i in range(1, len(test_data) + 1):
                    ndcg = NDCG(np.array(test_data.iloc[:, 0]), GT_score_normalized, i)
                    ndcg_data.append([i, ndcg])
                writer_ndcg.writerows(ndcg_data)


end = time.time()

