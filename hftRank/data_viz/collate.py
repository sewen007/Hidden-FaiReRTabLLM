import json
import os
import re
import time
import math
from pathlib import Path

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patches as patches

#from hftRank import shots

with open('./settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
protected_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"].lower()

start = time.time()

prompt_dict = {'prompt_1': 'Neutral', 'prompt_2': 'FC0', 'prompt_3': 'Tabular', 'prompt_4': 'FC1',
               'prompt_5': 'Neutral 2', 'prompt_6': 'CC1', 'prompt_8': 'CC2', 'prompt_10': 'FC2',
               'prompt_12': 'WC1', 'prompt_14': 'WC2', 'prompt_16': 'FD0', 'prompt_18': 'FD1', 'prompt_20': 'FD2'}


def get_files(directory, word):
    temp = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            match = re.search(word, file)
            if match:
                # temp.append(directory + '/' + file)
                temp.append(os.path.join(dirpath, file))
    return temp


def Collate(meta_exp='meta-llama/Meta-Llama-3.1-8B-Instruct', prompt_remove=None):
    print("Collating data...")
    # result folder
    folder = Path(f"./Results/{experiment_name}/Ranked")
    print(folder)
    # select all files in the folder with 'ndcg' in the name
    experiments = [f for f in os.listdir(folder) if os.path.isdir(folder / f)]
    for experiment in experiments:
        if 'llama' in experiment:
            experiment = meta_exp
        if 'deep' in experiment:
            experiment = 'deepseek-api/API'
        experiment_path = folder / experiment
        options = [f for f in os.listdir(experiment_path) if os.path.isdir(experiment_path / f)]
        for option in options:
            option_path = experiment_path / option
            inf_apps = [f for f in os.listdir(option_path) if os.path.isdir(option_path / f)]
            inf_order = ['inf_Gender', 'inf_GAPI', 'inf_BTN', 'inf_NMSOR', 'inf_NA', 'inf_NAD']
            sorted_inf_apps = sorted(inf_apps, key=lambda x: inf_order.index(x) if x in inf_order else len(inf_order))
            for inf_app in sorted_inf_apps:
                inf_app_path = option_path / inf_app
                prompts = [f for f in os.listdir(inf_app_path) if os.path.isdir(inf_app_path / f)]
                for prompt in prompts:
                    if prompt in prompt_remove:
                        continue
                    prompt_path = inf_app_path / prompt
                    # Get the list of sizes
                    sizes = [f for f in os.listdir(prompt_path) if os.path.isdir(prompt_path / f)]
                    for size in sizes:
                        size_path = prompt_path / size
                        # get the list of shots
                        shots = [f for f in os.listdir(size_path) if os.path.isdir(size_path / f)]
                        # select only shot_0, shot_1_, shot_2
                        shots = [shot for shot in shots]
                        # shots = [shot for shot in shots if
                        #          (
                        #                  'shot_0' in shot or 'shot_1' in shot or 'shot_2' in shot or 'shot_NA' in shot) and 'shot_10' not in shot]

                        sorted_shots = sorted(shots, key=lambda x: float('inf') if 'NA' in x else int(x.split('_')[1]))
                        for shot in sorted_shots:
                            shot_path = size_path / shot
                            print("Collating metrics for shot", shot, "in experiment", experiment, "for option", option, "inf ", inf_app,  "and size", size)

                            # get ndcg file
                            ndcg_file_path = shot_path / 'ndcg.csv'
                            collate_ndcgs(ndcg_file_path, prompt, shot, size, experiment, inf_app, option)

                            # get metric file
                            metric_file_path = shot_path / 'metrics.csv'
                            collate_metrics(metric_file_path, prompt, shot, size, experiment, inf_app, option)

                            accuracy_file_path = f"./Datasets/{experiment_name}/Ranked/{experiment}/option_3/inf_NA/prompt_8/rank_size_50/shot_{shot.split('_')[1]}/accuracy.txt"
                            get_average_accuracy(accuracy_file_path, shot, experiment)
    #

def get_average_accuracy(file_path, shot, experiment):
    output_file = f'./Results/{experiment_name}/{experiment_name}_size_50_collated_accuracies_LLM.csv'
    if not os.path.exists(file_path):
        return

    # if output file does not exist, create it
    if os.path.exists(output_file):
        # Load the existing output DataFrame
        output_df = pd.read_csv(output_file, index_col=0)
    else:
        # Create an empty DataFrame (with optional columns or index if needed)
        output_df = pd.DataFrame()
    col_name = f'{shot}'
    accuracies = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Accuracy of" in line:
                try:
                    # Extract the float at the end of the line
                    accuracy = float(line.strip().split()[-1])
                    accuracies.append(accuracy)
                except ValueError:
                    continue  # Skip lines where float conversion fails
    if accuracies:
        average = sum(accuracies) / len(accuracies)
        print(f"Average accuracy: {average:.4f}")
    else:
        print("No valid accuracy values found.")

        # add new columns to the output dataframes
    if col_name not in output_df.columns:
        output_df[col_name] = np.nan

    # write column values to the output dataframes
    output_df.loc[experiment, col_name] = average

    # Save the results to a new CSV file

    output_df.to_csv(output_file, index=True)


def collate_ndcgs(ndcg_file, prompt, shot, size, experiment, inf_app, option):
    # check if the file exists
    if not os.path.exists(ndcg_file):
        return
    ndcg_data = pd.read_csv(ndcg_file)

    if 'Position' in ndcg_data.columns:
        ndcg_data = ndcg_data.drop(columns=['Position'])

    if 'GroundTruth' in experiment:
        sorted_columns = sorted(ndcg_data.columns, key=lambda x: int(x.split('_')[-1]))

        # don't get average for ListNet
        for col in sorted_columns:
            # if 'NDCG' in col:
            col_name = col + '\n' + size + '\n' + prompt + '\n' + experiment
            ndcg_data[col_name] = ndcg_data[col]
            avg_or_same_ndcg = ndcg_data[[col_name]]
            # approximate each value to 2 decimal places
            avg_or_same_ndcg = avg_or_same_ndcg.round(2)

            # Save the results to a new CSV file
            output_file = f'./Results/{experiment_name}/{experiment_name}rank_size_{size}_collated_ndcg.csv'
            if not os.path.exists(output_file):
                avg_or_same_ndcg.to_csv(output_file, index=False)
            else:  # read the file and append the new data
                collated_data = pd.read_csv(output_file)
                collated_data[col_name] = avg_or_same_ndcg
                collated_data.to_csv(output_file, index=False)

            # save the results to a new CSV file
            output_file_with_std = f'./Results/{experiment_name}/{experiment_name}_rank_size_{size}_collated_ndcg_with_std.csv'
            if not os.path.exists(output_file_with_std):
                avg_or_same_ndcg.to_csv(output_file_with_std, index=False)
            else:  # read the file and append the new data
                collated_data = pd.read_csv(output_file_with_std)
                collated_data[col_name] = avg_or_same_ndcg
                collated_data.to_csv(output_file_with_std, index=False)
    else:

        col_name = 'AverageNDCG_' + shot + '\n' + size + '\n' + prompt + '\n' + experiment + '\n' + inf_app + '\n' + option

        # Calculate the average NDCG for each position
        ndcg_data[col_name] = ndcg_data.mean(axis=1)

        # get the aggregate mean and std
        ndcg_mean_n_std = ndcg_data.apply(lambda row: f"{row.mean():.2f} ± {row.std():.2f}", axis=1)

        # Convert the Series to a DataFrame and add a column name
        ndcg_mean_n_std_df = pd.DataFrame(ndcg_mean_n_std, columns=[col_name])

        avg_or_same_ndcg = ndcg_data[[col_name]]

        # Save the results to a new CSV file
        output_file = f'./Results/{experiment_name}/{experiment_name}_rank_size_{size}_collated_ndcg.csv'
        if not os.path.exists(output_file):
            avg_or_same_ndcg.to_csv(output_file, index=False)
        else:  # read the file and append the new data
            collated_data = pd.read_csv(output_file)
            collated_data[col_name] = avg_or_same_ndcg
            collated_data.to_csv(output_file, index=False)

        # Save the results to a new CSV file
        output_file_with_std = f'./Results/{experiment_name}/{experiment_name}_rank_size_{size}_collated_ndcg_with_std.csv'
        if not os.path.exists(output_file_with_std):
            ndcg_mean_n_std_df.to_csv(output_file_with_std, index=False)
        else:  # read the file and append the new data
            collated_data = pd.read_csv(output_file_with_std)
            collated_data[col_name] = ndcg_mean_n_std_df
            collated_data.to_csv(output_file_with_std, index=False)


def collate_metrics(metric_file, prompt, shot, size, experiment, inf_app, option):
    metric_data = pd.read_csv(metric_file)

    output_file = f'./Results/{experiment_name}/{experiment_name}_{size}_collated_metrics.csv'
    output_file_with_std = f'./Results/{experiment_name}/{experiment_name}_{size}_collated_metrics_with_std.csv'
    row_names = ['Kendall Tau', 'NDKL', 'Average Exposure', 'AveExpR CI-95']
    # if output file does not exist, create it
    if not os.path.exists(output_file):
        # Initialize the DataFrame with the row names
        output_df = pd.DataFrame(index=row_names)
    else:
        output_df = pd.read_csv(output_file, index_col=0)
    if not os.path.exists(output_file_with_std):
        output_df_with_std = pd.DataFrame(index=row_names)
    else:
        output_df_with_std = pd.read_csv(output_file_with_std, index_col=0)

    col_name = f'{shot}' + '\n' + size + '\n' + prompt + '\n' + experiment + '\n' + inf_app + '\n' + option
    kT_mean = metric_data['Kendall Tau'].mean()
    kT_std = metric_data['Kendall Tau'].std()
    NDKL_mean = metric_data['NDKL'].mean()
    NDKL_std = metric_data['NDKL'].std()
    avg_Exp_mean = metric_data['Average Exposure'].mean()
    avg_Exp_std = metric_data['Average Exposure'].std()
    ci_95 = 1.96 * (avg_Exp_std / np.sqrt(len(metric_data)))  # 95% CI

    # add new columns to the output dataframes
    output_df[col_name] = np.nan
    output_df_with_std[col_name] = np.nan

    # write column values to the output dataframes
    output_df.loc['Kendall Tau', col_name] = kT_mean
    output_df.loc['NDKL', col_name] = NDKL_mean
    output_df.loc['Average Exposure', col_name] = avg_Exp_mean
    output_df.loc['AveExpR CI-95', col_name] = ci_95
    output_df_with_std[col_name] = output_df_with_std[col_name].astype('object')

    output_df_with_std.loc['Kendall Tau', col_name] = f"{kT_mean:.2f} ± {kT_std:.2f}"
    output_df_with_std.loc['NDKL', col_name] = f"{NDKL_mean:.2f} ± {NDKL_std:.2f}"
    output_df_with_std.loc['Average Exposure', col_name] = f"{avg_Exp_mean:.2f} ± {avg_Exp_std:.2f}"
    output_df_with_std.loc['AveExpR CI-95', col_name] = f"{avg_Exp_mean:.2f} ± {ci_95:.2f}"
    # Save the results to a new CSV file
    output_file_with_std = f'./Results/{experiment_name}/{experiment_name}_{size}_collated_metrics_with_std.csv'
    output_df_with_std.to_csv(output_file_with_std, index=True)

    # Save the results to a new CSV file
    output_file = f'./Results/{experiment_name}/{experiment_name}_{size}_collated_metrics.csv'
    output_df.to_csv(output_file, index=True)


# Function to extract numerical part from column names
def extract_number(col_name):
    parts = re.split(r'[_\n]', col_name)
    if 'GroundTruth' in parts:
        return 0
    return int(parts[2])


def get_color_and_label(col):
    # Define color and label mappings for each group
    if 'gpt-4' in col.lower() and 'pre-ranked' in col.lower():
        return '#F00000', 'GPT-4o-mini-pre-ranked'  # red
    elif 'llama-3-' in col.lower() and 'pre-ranked' in col.lower():
        return '#3D6D9E', 'Llama-3-8B-Instruct-pre-ranked'  # blue
    elif 'gemini-1.5-flash' in col.lower() and 'pre-ranked' in col.lower():
        return '#A6192E', 'Gemini 1.5 Flash-pre-ranked'  # red/orange
    elif 'gemini-1.5-pro' in col.lower() and 'pre-ranked' in col.lower():
        return '#008080', 'Gemini 1.5 Pro-pre-ranked'  # teal
    elif 'gpt-4' in col.lower() and 'pre-ranked' not in col.lower():
        return '#800080', 'GPT-4o-mini'  # purple
    elif 'llama-3-' in col.lower() and 'pre-ranked' not in col.lower():
        return '#00CED1', 'Llama-3-8B-Instruct'  # dark turquoise
    elif 'gemini-1.5-flash' in col.lower() and 'pre-ranked' not in col.lower():
        return '#808000', 'Gemini 1.5 Flash'  # olive
    elif 'gemini-1.5-pro' in col.lower() and 'pre-ranked' not in col.lower():
        return '#FF00FF', 'Gemini 1.5 Pro'  # magenta
    elif 'prompt_NA' in col:
        return '#FFC725', 'GroundTruth'  # yellow
    else:
        return '#000000', 'Unknown'  # default color and label if no match


# Function to extract mean from "mean ± std" format
def extract_mean(value):
    if isinstance(value, float):
        return value
    return float(value.split(' ± ')[0])



def plot_skew(metrics_file_path, size=None):
    skew_file = pd.read_csv(metrics_file_path)

    skew_file['Ideal'] = 1

    # get dataset from file path
    dataset = experiment_name
    g_dis = protected_group

    if g_dis == 'female':
        g_adv = 'Males'
        g_dis = 'Females'
    else:
        g_adv = 'Females'
        g_dis = 'Males'

    skew_file.rename(columns={'Group_0': str(g_adv), 'Group_1': str(g_dis)}, inplace=True)
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(4, 4))

    sns.set(font_scale=1.2,  # Font scale factor (adjust to your preference)
            rc={"font.style": "normal",  # Set to "normal", "italic", or "oblique"
                "font.family": "serif",  # Choose your preferred font family
                # Font size
                "font.weight": "normal"  # Set to "normal", "bold", or a numeric value
                })

    sns.lineplot(data=skew_file[[g_dis, g_adv, "Ideal"]], dashes=False, ax=ax)

    # pipe, graph_title = get_graph_name(metrics_file_path)

    ax.set_title(dataset, fontsize='x-large', fontfamily='serif')


    # Set the x and y limits to start from 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.xlabel("Ranking position", fontfamily='serif')
    plt.ylabel("Skew", fontfamily='serif')
    plt.tight_layout()
    plt.legend(frameon=False, fontsize='xx-small', loc='upper right')

    """ DIRECTORY MANAGEMENT """
    if size is not None:
        graph_path = Path(
            "./Hidden-FaiReR-TabLLM/Plots/" +
            str(dataset) + "/" + str(os.path.dirname(metrics_file_path).split(experiment_name)[1]) + "/size_" + str(size))
    else:
        graph_path = Path(
        "./Hidden-FaiReR-TabLLM/Plots/" +
        str(dataset) + "/" + str(os.path.dirname(metrics_file_path).split(experiment_name)[1]))
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    plt.savefig(os.path.join(graph_path, str(dataset) + '_skews.png'))
    plt.close()

    return



# end = time.time()
