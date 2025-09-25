import csv
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

with open('./settings.json', 'r') as f:
    settings = json.load(f)
dataset = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
protected_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"].lower()

datasets = ["NBAWNBA", "LOAN", "BostonMarathon", "LAW"]

prompt_dict = {'prompt_2': 'BASE', 'prompt_4': 'REP', 'prompt_6': 'EXP', 'prompt_8': 'FEA', 'prompt_10': 'FEA+REP',
               'prompt_12': 'FEA+EXP', 'prompt_14': 'FEA+DIS', 'prompt_16': 'FEA+DIS+REP', 'prompt_18': 'FEA+DIS+EXP',
               'prompt_NA': 'Initial', 'prompt_NAD': 'DetConstSort'}
option_dict = {'1,2':'SOTA & LLM-INF-IMPLICIT' , '1': 'SOTA-INF', '2a': 'LLM-INF-IMPLICIT-A', '2b': 'LLM-INF-IMPLICIT-B',  '3': 'LLM-INF-IMPLICIT'}
experiment_name = dataset
prompts = ['prompt_2', 'prompt_4', 'prompt_6', 'prompt_8', 'prompt_10', 'prompt_12', 'prompt_14', 'prompt_16',
           'prompt_18']

prompt_numbers = ['2', '4', '6', '8', '10', '12', '14', '16', '18']
models = ['gemini-2.0-flash-thinking-exp-01-21', 'Meta-Llama-3-8B-Instruct', 'deepseek-api']
shots = settings["GENERAL_SETTINGS"]["shots"]

# test_set = f"./Datasets/{experiment_name}/{experiment_name}_test_data.csv"
# full_size = len(pd.read_csv(test_set))

def get_llm_label(col):
    # Define label mappings for each group
    if 'gemini-2.0-flash-thinking-exp-01-21' in col:
        return 'Gemini', get_color_and_label(col)[1]
    elif 'Meta-Llama-3-8B-Instruct' in col:
        return 'Llama', get_color_and_label(col)[1]
    elif 'deepseek-api' in col:
        return 'DeepSeek', get_color_and_label(col)[1]
    else:
        return '#000000', 'Unknown'  # default color and label if no match

def get_color_and_label(col):
    # Define color and label mappings for each group
    if 'BTN' in col:
        return '#097481', "BTN"
    elif 'GAPI' in col:
        return '#FB3640',  "GAPI"
    elif 'NMSOR' in col:
        return '#4B296B',  "NMS"
    elif 'Gender' in col:
        return 'white', "GT"

    # elif 'gemini-1.5-flash' in col.lower():
    #     return '#DFF2DF', 'Gemini 1.5 Flash'  # light olive
    elif 'option_2a' in col.lower():
        return '#FFA07A', 'IMP-P'  # light salmon
    elif 'option_2b' in col.lower():
        return '#DFF2DF', 'IMP-A' # light olive
    elif 'option_3' in col.lower():
        return 'k', 'CoT'  # black
    elif 'gemini-1.5-flash' in col.lower():
        return '#FFCCFF', 'Gemini 1.5 Flash'  # lighter magenta
    elif 'prompt_NA\n' in col:
        return '#FFC725', 'Initial'  # yellow
    elif 'prompt_NAD' in col:
        return '#4682B4', 'DCS'
    elif 'llama-3' in col.lower():
        return '#00CED1', 'Llama'  # dark turquoise
    else:
        return '#000000', 'Unknown'  # default color and label if no match


def get_prompt(col):
    # in col, search for 'prompt' and return string after prompt but before /
    if 'prompt' in col:
        return prompt_dict[col.split('\n')[2]]
        # return col.split('prompt')[1].split('/')[0]
    else:
        return 'ListNet'


def get_label(column_name):
    # look for text "shot in the column name and extract the number"
    if 'shot_NAD' in column_name:
        return 'shot_NA'
    elif 'shot' in column_name:
        if 'NDCG' in column_name:
            return (column_name.split('\n')[0]).split('NDCG_')[1]
        else:
            return column_name.split('\n')[0]
    else:
        if 'NDCG' in column_name:
            return (column_name.split('\n')[0]).split('NDCG_')[1]
        else:

            return 'train_size_' + str(column_name.split('\n')[0])


# Define hatching patterns based on custom logic
def get_hatch_pattern(label):
    if 'REP' in label:
        return '//'
    elif 'EXP' in label:
        return '.'
    else:
        return ''  # No hatch


def get_hatch_color(label):
    if 'REP' in label:
        return 'black'
    elif 'EXP' in label:
        return 'black'
    else:
        return 'black'  # Default hatch color

def get_size(label: str):
    """Extract the first number after 'rank_size_' directly from the label string."""
    match = re.search(r"rank_size_(\d+)", label)
    if match:
        return int(match.group(1))
    return None

def relabel_inference_type(label, zero_only=False):
    if 'shot_0' in label or 'shot_NA' in label:  # Apply get_color_and_label only for shot_0
        if zero_only and '2b' in label:
            return 'IMP'
        else:
            return get_color_and_label(label)[1]
    else:
        match = re.search(r'shot_(\d+)', label)  # Extract number from shot_X
        return match.group(1) if match else label  # Use the number or fallback to original label
def relabel_llm(label, notion, multi_size = None):
    match_size = get_size(label)
    if 'shot_0' in label:  # Apply get_color_and_label only for shot_0
        if notion == 'EXP_FEA':
            # return str(get_llm_label(label)[0] + ', ' + get_color_and_label(label)[1])
        # uncomment next line when it is only one option and inf_app
            if multi_size:
                return str(get_llm_label(label)[0]) + f',{match_size}'
            else:
                return str(get_llm_label(label)[0]) #section 2
        else:
            if multi_size:
                return str(get_llm_label(label)[0]) + f', {match_size}'
            else:
                return get_llm_label(label)[0]
    elif 'shot_NA' in label:
        if multi_size:
            return get_color_and_label(label)[1] + f', {match_size}'
        else:
            return get_color_and_label(label)[1]
    else:
        match = re.search(r'shot_(\d+)', label)  # Extract number from shot_X
        if multi_size:
            return match.group(1) + f', {match_size}' if match else label
        else:
            return match.group(1) if match else label  # Use the number or fallback to original label

def relabel_prompts(label, multi_size = None): #so foar we don;t need the match_size. but we leave it
    match_size = get_size(label)
    if 'shot_0' in label:  # Apply get_color_and_label only for shot_0
        match = re.search(r'prompt_(\d+)', label)  # Extract number from prompt_X
        if multi_size:
            return prompt_dict['prompt_' + str(match.group(1))] + f', {match_size}' if match else label
        else:
            return prompt_dict['prompt_' + str(match.group(1))] if match else label  # Use the number or fallback to original label
    elif 'shot_NA' in label:
        return get_color_and_label(label)[1]
    else:
        match = re.search(r'shot_(\d+)', label)  # Extract number from shot_X
        if multi_size:
            return match.group(1) + f', {match_size}' if match else label
        else:
            return match.group(1) if match else label  # Use the number or fallback to original label


def plot_metrics(experiment_section, size, zero_only=False, non_zero_only=False, llm='gemini-1.5-flash', metric='Kendall Tau',
                 specific_prompts=None, notion=None, multi_llm=False, option=None, specific_inf_apps=None, multiple_options=False, special_title=None, across_sizes=None):
    pic_style = 'pdf'
    if across_sizes:
        # select muliple sizes
        data = pd.DataFrame()
        for sz in across_sizes:
            df = pd.read_csv(f'./Results/{experiment_name}/{experiment_name}_rank_size_{sz}_collated_metrics_with_std.csv')
            df = df.loc[:, ~df.columns.duplicated()]
            # append to data_frame
            data = pd.concat([data, df], axis=1)
            # drop DetConstSort
            data = data.loc[:, ~data.columns.str.contains('DetConstSort')]

    else:

        data = pd.read_csv(f'./Results/{experiment_name}/{experiment_name}_rank_size_{size}_collated_metrics_with_std.csv')
    data = data.loc[:, ~data.columns.duplicated()]


    # make first column the index
    data = data.set_index(data.columns[0])

    # filter based on option
    if option:
        data = data.loc[:,data.columns.str.contains(
                                'Initial') | data.columns.str.contains('DetConstSort')| data.columns.str.contains('|'.join(option))]

    if multi_llm:
        collated_data = data
        # Generate prompt_order dynamically
        order = ['prompt_NA\n', 'prompt_NAD\n']

        # Add additional prompts

        order += [f'prompt_{num}\n{model}' for model in models for num in prompt_numbers]
    else:

        # select only the columns with the specified llm, Initial and DetConstSort
        collated_data = data.loc[:, data.columns.str.contains(
                            'Initial') | data.columns.str.contains('DetConstSort') |
                        data.columns.str.contains(llm) ]
        order = ['prompt_NA\n', 'prompt_NAD\n', 'prompt_2\n', 'prompt_4\n', 'prompt_6\n', 'prompt_8\n', 'prompt_10\n', 'prompt_12\n', 'prompt_14\n',
                 'prompt_16\n', 'prompt_18\n']



    custom_size = 'large'
    if zero_only:
        custom_size = 'xx-large'

    if zero_only:
        # add Listnet columns and columns with shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('Initial') | collated_data.columns.str.contains(
                            'shot_0') | collated_data.columns.str.contains('DetConstSort')]
        # drop column with '2a' in it
        collated_data = collated_data.loc[:, ~collated_data.columns.str.contains('2a')]
        #  = write_folder / 'zero_only'

    if non_zero_only:
        # add Listnet columns and columns without shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('Initial') | collated_data.columns.str.contains(
                            'DetConstSort') | ~collated_data.columns.str.contains('shot_0')]

    # Reorder the columns
    ordered_columns = sorted(collated_data.columns,
                             key=lambda col: next((i for i, word in enumerate(order) if word in col), len(order)))
    collated_data = collated_data[ordered_columns]
    order = ['Initial', 'DetConstSort', 'gemini', 'Llama', 'deepseek']
    # Reorder the columns
    ordered_columns = sorted(collated_data.columns,
                             key=lambda col: next((i for i, word in enumerate(order) if word.lower() in col.lower()), len(order)))
    collated_data = collated_data[ordered_columns]
    # if across_sizes, order initial then llm by size
    if experiment_section==0:
        inf_order = []
        for sz in across_sizes:
            inf_order.append(f'Initial, rank_size_{sz}')
            inf_order.append(f'shot_0, rank_size_{sz}')

        ordered_columns = []
        for token in inf_order:
            kind, size = [t.strip() for t in token.split(",")]
            for col in collated_data.columns:
                col_str = str(col)
                if kind in col_str and size in col_str:
                    ordered_columns.append(col)

        for col in collated_data.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)

        # reindex dataframe if desired
        collated_data = collated_data.loc[:, ordered_columns]


    if specific_prompts:
        collated_data = collated_data.loc[:, collated_data.columns.str.contains('|'.join(specific_prompts))]

    if specific_inf_apps:
        # collated_data = collated_data.loc[:, collated_data.columns.str.contains('|'.join(specific_inf_apps))]
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('Initial') | collated_data.columns.str.contains(
                            'DetConstSort') | collated_data.columns.str.contains(specific_inf_apps)]
        if experiment_section==0:
            # remove DetConstSort
            collated_data = collated_data.loc[:, ~collated_data.columns.str.contains('DetConstSort')]


    #####figure size########

    if zero_only:
        if across_sizes:
            fig, ax = plt.subplots(figsize=(15, 6))
        else:
            fig, ax = plt.subplots(figsize=(12, 6)) #section 3
    else:
        fig, ax = plt.subplots(figsize=(60, 6))

    if multi_llm:
        if multiple_options:
            fig, ax = plt.subplots(figsize=(17, 6))
    if notion == 'EXP_FEA':
        if specific_inf_apps:
            print('****')
            fig, ax = plt.subplots(figsize=(8, 6))


    # store handles and labels for the legend
    handles = []
    labels = []
    added_labels = set()  # tracking labels to avoid duplicates

    bar_count = 1
    collated_data.columns = collated_data.columns.str.replace("\r", "")
    for col in collated_data.columns:

        value = float(collated_data.loc[metric, col].split(' ± ')[0])
        std = float(collated_data.loc[metric, col].split(' ± ')[1])
        if zero_only:
            custom_label = get_prompt(col)
        else:
            custom_label = get_prompt(col) + ', ' + get_label(col)

        if (metric == 'Average Exposure' or metric == 'AveExpR CI-95') and not across_sizes:
            if 'Initial' in col:
                plt.axhline(y=value, color='blue', linestyle='--')
                plt.text(x=len(collated_data.columns), y=1, s='', color='black', va='bottom',
                         ha='left')  # Label near the right edge

        # Use the custom label (wetin) for bar placement
        x_pos = custom_label
        hatch = get_hatch_pattern(x_pos)
        # bar = ax.bar(x_pos, value, yerr=std, capsize=5, color=get_color_and_label(col)[0], hatch=hatch,
        #             label=col.split('\n')[2])
        if metric == 'NDKL':
            tinny_bitty = 0.005
        else:
            tinny_bitty = 0.01

        written_number_size = 30

        color = get_color_and_label(col)[0]
        ecolor = 'grey' if color == 'k' else 'black'  # or any default color you prefer


        bar = ax.bar(col, value, yerr=std, capsize=5, color=color, hatch=hatch,
                     label=col.split('\n')[2], error_kw={'ecolor': ecolor})
        if not zero_only:  # zero_only=False
            written_number_size = 35
            if multiple_options:
                written_number_size = 40
        if experiment_section==5:
            written_number_size = 20

        ax.text(col, value + std + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                fontsize=written_number_size, fontweight='bold')

        for patch in bar:
            patch.set_edgecolor(get_hatch_color(x_pos))

        # Increment the group index for each new bar
        bar_count += 1


        color = get_color_and_label(col)[0]
        label = get_color_and_label(col)[1]

        if label not in added_labels:
            handles.append(bar)
            labels.append(label)
            added_labels.add(label)

    y = 1.9
    y_ndkl = 0.5
    y_ave = 1.2

    # if "Boston" in experiment_name:
    #     if metric == 'AveExpR CI-95':
    #         y_ave = 1.25 # section 2
    #         if specific_inf_apps is None:
    #             y_ave = 1.1 # section 3
    #         else:
    #             if across_sizes:
    #                 y_ave = 1.4
    #     elif metric == 'NDKL':
    #         y_ndkl = 0.5 #section 2
    #         if specific_inf_apps is None:
    #             y_ndkl = 0.35 #section 3
    if experiment_section ==2:
        y_ndkl = 0.38 if "COMPAS" not in experiment_name else 0.45
        if 'Boston' in experiment_name:
            y_ave = 1.25
        elif "NBA" in experiment_name or "COMPAS" in experiment_name:
            y_ave = 1.4

    elif experiment_section == 3:
        y_ave = 1.1
        y_ndkl = 0.32

    elif experiment_section == 4:
        if 'NBA' in experiment_name:
            y_ave = 1.4

    elif experiment_section == 5:
        y_ave = 1.25 if 'IMP' in notion else 1.6
        y_ndkl = 0.3 if 'IMP' in notion else 0.45
    elif experiment_section ==0:
        y_ave = 1.4

    # if 'NBA' in experiment_name and metric == 'AveExpR CI-95':
    #     y_ave = 1.6


    if metric == 'NDKL':
        plt.ylim(0.0, y_ndkl)
    elif 'Ave' in metric:
        plt.ylim(0.6, y_ave)
    else:
        plt.ylim(0, y)

    plt.yticks(fontsize=30)
    plt.xlabel(' ')

    metric_label = metric
    if metric == 'AveExpR CI-95':
        metric_label = 'AveEXpR'
    plt.ylabel(f'{metric_label}', fontsize=35, fontweight='bold')
    plt.tight_layout()
    ax.set_xticks(range(len(collated_data.columns)))

    xtickfont_size = 30
    if multiple_options:
        if zero_only is False:
            xtickfont_size = 30

    if experiment_section==5:
        xtickfont_size = 30
    if notion == 'EXP_FEA':
        if specific_inf_apps:
            xtickfont_size = 30
        else:
            xtickfont_size = 30

    rotation = 45


    # xtick rotation and alignment

    if across_sizes: #experiment_section==0:
        if metric == 'NDKL':
            # plt should be empty #uncomment if for paper
            # ax.set_xticklabels([' ' for _ in collated_data.columns],
            #                    rotation=rotation, ha='right',
            #                    fontsize=xtickfont_size)
            ax.set_xticklabels(
                [relabel_llm(col, notion=notion, multi_size=across_sizes) for col in collated_data.columns],
                rotation=rotation, ha='right',
                fontsize=xtickfont_size)

        else:
            ax.set_xticklabels([relabel_llm(col,notion=notion, multi_size=across_sizes) for col in collated_data.columns], rotation=rotation, ha='right',
                               fontsize=xtickfont_size)

    elif specific_prompts is None:
        ax.set_xticklabels([relabel_prompts(col) for col in collated_data.columns], rotation=rotation, ha='right',
                           fontsize=xtickfont_size)
    elif multi_llm:

        if metric == 'NDKL':
            # plt should be empty
            ax.set_xticklabels([relabel_llm(_, notion)  for _ in collated_data.columns], rotation=rotation, ha='center',
                               fontsize=xtickfont_size)
        else:

            ax.set_xticklabels([relabel_llm(col, notion) for col in collated_data.columns], rotation=rotation, ha='center',
                           fontsize=xtickfont_size)
    else: # use commented out part to see labels
        # ax.set_xticklabels([relabel_inference_type(col, zero_only) for col in collated_data.columns], rotation=45, ha='right',
        #                    fontsize=xtickfont_size)
        ax.set_xticklabels([' ' for _ in collated_data.columns],
                           rotation=rotation, ha='right',
                           fontsize=xtickfont_size)


    if metric == 'Average Exposure' or metric == 'AveExpR CI-95':
        plt.axhline(y=1, color='red', linestyle='--')
        # plt.text(x=len(collated_data.columns), y=1, s=oracle, color='black', va='bottom',
        #          ha='left', fontsize=25)  # Label near the right edge

    ha = 'center'
    if experiment_section==5:
        ha = 'right'
    plt.xticks(rotation=rotation, ha=ha)

    plt.title(' ')

    plot_ideal(metric, axis='y', section=experiment_section)

    if zero_only:
        for i in range(1, len(collated_data.columns), 4):  # Start from index 2 and step by 3
            plt.axvline(x=i + 0.5, color='grey', linestyle='--')  # Use x=i+0.5 to place it after the bar
    else:
        # Add vertical lines after every third bar
        for i in range(1, len(collated_data.columns), 5):  # Start from index 2 and step by 3
            plt.axvline(x=i + 0.5, color='grey', linestyle='--',
                        linewidth=1)  # Use x=i+0.5 to place it after the bar

    if experiment_section == 2:
        if "NBA" in experiment_name:
            plt.title('(W)NBA' if metric == 'NDKL' else ' ', fontsize=40)
            plt.ylabel(' ')
        elif "Boston" in experiment_name:
            plt.title('Boston Marathon' if metric == 'NDKL' else ' ', fontsize=40)
            plt.ylabel('NDKL' if metric == 'NDKL' else 'AveExpR', fontsize=35, fontweight='bold')
        elif "COMPAS" in experiment_name:
            plt.title('COMPAS' if metric == 'NDKL' else ' ', fontsize=40)
            plt.ylabel(' ')

        if metric == 'NDKL':
            # ax.set_xticklabels([' '] * len(collated_data.columns),
            #                    rotation=rotation, ha='right', fontsize=xtickfont_size)
            ax.set_xticklabels([relabel_llm(col, notion) for col in collated_data.columns], rotation=rotation,
                               ha='right',
                               fontsize=xtickfont_size)
        else:
            ax.set_xticklabels([relabel_llm(col, notion) for col in collated_data.columns], rotation=rotation,
                               ha='right',
                               fontsize=xtickfont_size)
    elif experiment_section == 3:
        # add x-ticks always
        ax.set_xticklabels([relabel_inference_type(col, zero_only) for col in collated_data.columns], rotation=0,
                           ha='center', fontsize=xtickfont_size)



    # plt.legend()
    print('notion = ', notion)
    if across_sizes:
        base_folder =  f'./Plots/{dataset}/multi_size/'
    else:
        base_folder = f'./Plots/{dataset}/size_{size}/'

    if multiple_options:
        base_folder += '/multiple_options/'

    if specific_prompts is not None:
        base_folder += f'/specific_prompts/{notion}'
    if notion is not None:
        base_folder += f'/{notion}/'
    else:
        base_folder += f'/all_notions/'

    if zero_only:
        base_folder += '/zero_only/'
    elif non_zero_only and not (multiple_options and specific_prompts is not None):
        # non_zero_only is only used when not in specific_prompts
        base_folder += '/non_zero_only/'

    save_folder = base_folder

    if specific_inf_apps is not None:
        save_folder = f'{save_folder}/{specific_inf_apps}/'

    os.makedirs(save_folder, exist_ok=True)

    metric_name = metric
    if metric == 'AveExpR CI-95':
        # metric = 'Average Exposure'
        metric_name = 'AveExpR'

    # plt.close(legend_fig)
    if across_sizes:
        save_name = f'{save_folder}_across_sizes_{metric_name}_{llm}.{pic_style}'
    elif specific_prompts is not None:
        if zero_only:
            save_name = f'{save_folder}{llm}_{metric_name}_{size}_{notion}_zero.{pic_style}'
            if specific_inf_apps:
                save_name = f'{save_folder}{llm}_{metric_name}_{size}_{notion}_zero_{specific_inf_apps}.{pic_style}'
        else:
            save_name =f'{save_folder}{llm}_{metric_name}_{size}_{notion}.{pic_style}'
            if specific_inf_apps:
                save_name = f'{save_folder}{llm}_{metric_name}_{size}_{notion}_{specific_inf_apps}.{pic_style}'
    else:
        save_name = f'{save_folder}{llm}_{metric_name}_{size}.{pic_style}'

    plt.savefig(save_name, bbox_inches='tight')

    # Create a new figure to contain just the legend
    # legend_fig, legend_ax = plt.subplots(figsize=(2, 1))
    # #legend = legend_ax.legend(handles=handles, labels=labels, ncol=1, loc='center')
    # legend_ax.axis('off')
    #
    # # Step 4: Save the legend as an image file
    # legend_fig.tight_layout()
    # legend_fig.savefig(f'legend_{llm}.{pic_style}', bbox_inches='tight')


def plot_ndcgs(experiment_section, size, zero_only=False, non_zero_only=False, prompt='prompt_1', all_prompts=False, specific_llm=None,
               specific_prompts=None, notion=None, option=None, multiple_options=None, specific_inf_apps=None):
    write_folder = f"./Plots/{dataset}/size_{size}/NDCG/{notion}"

    if multiple_options:
        write_folder = f"{write_folder}/multiple_options"

    data = pd.read_csv(f'./Results/{experiment_name}/{experiment_name}_rank_size_rank_size_{size}_collated_ndcg_with_std.csv')
    data.columns = data.columns.str.replace("\r", "")
    if option:
        data = data.loc[:, data.columns.str.contains(
                            'Initial') | data.columns.str.contains('DetConstSort') | data.columns.str.contains('|'.join(option))]


    if specific_llm is not None:
        # select only the columns with the specified llm, Initial and DetConstSort
        data = data.loc[:, data.columns.str.contains(
                                'Initial') | data.columns.str.contains('DetConstSort') | data.columns.str.contains(specific_llm)]
    if specific_inf_apps:
        # collated_data = collated_data.loc[:, collated_data.columns.str.contains('|'.join(specific_inf_apps))]
        data = data.loc[:,
                        data.columns.str.contains('Initial') | data.columns.str.contains(
                            'DetConstSort') | data.columns.str.contains(specific_inf_apps)]
        # order = ['Initial', 'DetConstSort', specific_inf_apps]

    if all_prompts:
        collated_data = data
        prompt_order = ['prompt_2\n', 'prompt_4\n', 'prompt_6\n', 'prompt_8\n', 'prompt_10\n', 'prompt_12\n',
                        'prompt_14\n',
                        'prompt_16\n', 'prompt_18\n', 'prompt_NA\n', 'prompt_NAD\n']
        # Reorder the columns
        prmt_ordered_columns = sorted(collated_data.columns,
                                      key=lambda col: next((i for i, word in enumerate(prompt_order) if word in col),
                                                           len(prompt_order)))
        collated_data = collated_data[prmt_ordered_columns]
    elif specific_prompts is not None:
        # select only the columns with the specified llm, Initial and DetConstSort
        collated_data = data.loc[:, data.columns.str.contains('Initial') | data.columns.str.contains('DetConstSort') | data.columns.str.contains('|'.join(specific_prompts))]
        prompt_order = ['prompt_2\n', 'prompt_4\n', 'prompt_6\n', 'prompt_8\n', 'prompt_10\n', 'prompt_12\n',
                        'prompt_14\n',
                        'prompt_16\n', 'prompt_18\n', 'prompt_NA\n', 'prompt_NAD\n']
        # Reorder the columns
        prmt_ordered_columns = sorted(collated_data.columns,
                                      key=lambda col: next((i for i, word in enumerate(prompt_order) if word in col),
                                                           len(prompt_order)))
        collated_data = collated_data[prmt_ordered_columns]
    else:
        # select only the columns with the specified prompt
        collated_data = data.loc[:, data.columns.str.contains('Initial') | data.columns.str.contains('DetConstSort') | data.columns.str.contains(prompt)]
    order = ['Initial', 'DetConstSort', 'gemini-2.0-pro-exp-02-05', 'Meta-Llama-3-8B-Instruct']
    # Reorder the columns
    ordered_columns = sorted(collated_data.columns,
                             key=lambda col: next((i for i, word in enumerate(order) if word in col), len(order)))
    collated_data = collated_data[ordered_columns]
    if zero_only:
        # add Initial columns and columns with shot_0
        collated_data = collated_data.loc[:,collated_data.columns.str.contains('Initial') | collated_data.columns.str.contains('DetConstSort')| collated_data.columns.str.contains('shot_0')]
        collated_data = collated_data.loc[:, ~collated_data.columns.str.contains('2a')]
        write_folder = f'{write_folder}/zero_only'

    if non_zero_only:
        # add Listnet columns and columns without shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('Initial') | collated_data.columns.str.contains(
                            'DetConstSort') | ~collated_data.columns.str.contains('shot_0')]
        write_folder = f'{write_folder}/non_zero_only'

    if specific_llm is not None:
        write_folder = f'{write_folder}/{specific_llm}'
    if specific_prompts is not None:
        write_folder = f'{write_folder}/specific_prompts/{notion}'
    if specific_inf_apps:
        write_folder = f'{write_folder}/{specific_inf_apps}'


    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    for idx, row in collated_data.iterrows():
        fig, ax = plt.subplots(figsize=(12, 6))
        if not zero_only:
            if specific_inf_apps:
                fig, ax = plt.subplots(figsize=(15, 6))
            else:
                fig, ax = plt.subplots(figsize=(60, 6))


        # store handles and labels for the legend
        handles = []
        labels = []
        added_labels = set()  # tracking labels to avoid duplicates

        # for col in collated_data.columns:
        for col in collated_data.columns:
            color = get_color_and_label(col)[0]
            ecolor = 'grey' if color == 'k' else 'black'  # or any default color you prefer
            current_prompt = get_prompt(col)
            label = get_color_and_label(col)[1]
            hatch = get_hatch_pattern(current_prompt)
            std = float(row[col].split(' ± ')[1])
            value = float(row[col].split(' ± ')[0])

            bar = ax.bar(col, value, color=color, label=label, hatch=hatch, edgecolor='black', yerr=std, capsize=5, error_kw={'ecolor': ecolor})
            # manually change name beneath the bar without changing col name

            if label not in added_labels:
                handles.append(bar)
                labels.append(label)
                added_labels.add(label)

            # plt.legend()
            text_size = 35

            if multiple_options:
                text_size = 40
            if zero_only:
                text_size = 30
            # if all_prompts:
            #     text_size = 12

            # add the integer value on top of the bar and make it vertical
            ax.text(col, value+std+0.01, f'{value:.2f}', ha='center', va='bottom', rotation='horizontal', fontsize=text_size, fontweight='bold')

        title_weight = 'bold'
        if zero_only:
            title_weight = 'normal'

        llm_name = specific_llm
        if specific_llm == 'deepseek-api':
            llm_name = 'DeepSeek-V3'

        plt.title(f'{llm_name}, {dataset}', fontsize=35, fontweight=title_weight)
        if not zero_only:
            plt.title(f'{llm_name}, {dataset}', fontsize=45, fontweight=title_weight)
        plt.title(f' ')

        #
        if not all_prompts:
            plt.xlabel('LLM shots', fontsize=30)
        else:
            if not zero_only:
                plt.xlabel('prompts,' + ' shots', fontsize=35)
            plt.xlabel('prompts', fontsize=30)
        if prompt == 'prompt_2' or prompt == 'prompt_12':
            plt.ylabel(f'NDCG@{idx + 1}', fontsize=30)
        elif specific_llm is not None:
            #plt.ylabel(f'NDCG@{idx + 1}', fontsize=30)
            plt.ylabel(f'NDCG@{idx + 1}', fontsize=35, fontweight='bold')
        else:
            plt.ylabel(' ', fontsize=30)

        plt.yticks(fontsize=30)
        plt.tight_layout()

        if zero_only:
            # plt.xlabel('prompts', fontsize=30)
            plt.xlabel(' ')
        else:
            plt.xlabel('Alternate Fair Ranking Methods as defined in Section IV', fontsize=50)
            # plt.xlabel('prompts,' + ' shots', fontsize=35)

        xtickfont_size = 30
        if not zero_only and not specific_inf_apps:
            xtickfont_size = 40
        if multiple_options:
            if zero_only is False:
                xtickfont_size = 40
        ax.set_xticks(range(len(collated_data.columns)))  # Set fixed tick positions
        if zero_only or specific_inf_apps:

            ax.set_xticklabels([relabel_inference_type(col, zero_only) for col in collated_data.columns], rotation=45, ha='right', fontsize=xtickfont_size)

        else:
            if specific_prompts is None:
                ax.set_xticklabels([relabel_prompts(col) for col in collated_data.columns], rotation=45, ha='right',
                                   fontsize=xtickfont_size)
            else:
                ax.set_xticklabels([relabel_inference_type(col, zero_only) for col in collated_data.columns], rotation=45,
                                   ha='right', fontsize=xtickfont_size)

        plt.xticks(rotation=0, ha='center', size=xtickfont_size)
        # Save the plot as a PDF file with the specified naming convention

        # Convert 'value ± std' format to two separate numeric DataFrames
        data_values = data.map(lambda x: float(x.split(' ± ')[0]) if isinstance(x, str) and ' ± ' in x else float(x))
        std_values = data.map(lambda x: float(x.split(' ± ')[1]) if isinstance(x, str) and ' ± ' in x else 0.0)

        add_to_y = 0.55 if experiment_name == 'COMPASSEX' else 0.2
        add_to_y = 0.3 if experiment_name == 'BostonMarathon' else add_to_y
        # Compute y-axis limits safely
        plt.ylim(min(data_values.min()) - 0.05, max(data_values.max() + std_values.max()) + add_to_y)
        if experiment_name == 'BostonMarathon':
            plt.ylim(0.6, 1.0)


        if zero_only:
            for i in range(1, len(collated_data.columns), 4):  # Start from index 2 and step by 3
                plt.axvline(x=i + 0.5, color='grey', linestyle='--')  # Use x=i+0.5 to place it after the bar
        else:
            # Add vertical lines after every third bar
            for i in range(1, len(collated_data.columns), 5):  # Start from index 2 and step by 3
                plt.axvline(x=i + 0.5, color='grey', linestyle='--',
                            linewidth=1)  # Use x=i+0.5 to place it after the bar
            # Add vertical lines after every third bar

        if not all_prompts and not specific_inf_apps:
            save_name = f'{write_folder}/{llm_name}_{notion}_NDCG_at_{idx + 1}_Bar_Chart.pdf'
            if zero_only:
                save_name = f'{write_folder}/{llm_name}_{notion}_NDCG_at_{idx + 1}_Bar_Chart_zero.pdf'
        else:
            if specific_llm:
                if specific_prompts:
                    if specific_inf_apps:
                        save_name = f'{write_folder}/{notion}_{specific_inf_apps}_All_NDCG_at_{idx + 1}_Bar_Chart.pdf'
                    else:
                        save_name = f'{write_folder}/{notion}_All_NDCG_at_{idx + 1}_Bar_Chart.pdf'
                else:
                    save_name = f'{write_folder}/{specific_llm}_{notion}_All_NDCG_at_{idx + 1}_Bar_Chart.pdf'
            else:
                save_name= f'{write_folder}/{notion}_All_NDCG_at_{idx + 1}_Bar_Chart.pdf'
        plt.savefig(save_name, bbox_inches='tight')
        # Close
        plt.close()


def plot_ideal(metric, axis='y', section=2):
    print(metric)
    # Define ideal values
    ideal_value = {'AveExpR CI-95': 1.0, 'NDKL': 0}
    value_to_box = ideal_value.get(metric, ideal_value[metric])

    # Get current axes
    ax = plt.gca()

    # Select axis labels
    labels = ax.get_yticklabels() if axis == 'y' else ax.get_xticklabels()

    # Find the tick label with the ideal value
    for label in labels:
        text = label.get_text()
        match = re.match(r"[-+]?\d*\.\d+|\d+", text)
        if match and float(match.group()) == value_to_box:
            # Highlight the tick label with a red-bordered box
            label.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='red'))

            if section == 0: x_point = -0.04 if metric == 'NDKL' else -0.035
            elif section == 2: x_point = -0.08
            elif section == 3: x_point = -0.042 if metric == 'NDKL' else -0.042
            elif section == 4: x_point = -0.01 if metric == 'NDKL' else -0.008
            elif section == 5: x_point = -0.04 if metric == 'NDKL' else -0.03

            # Add "BEST VALUE" directly under the tick label
            ax.text(
                x_point, value_to_box - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08,  # just under tick
                "BEST",
                color='red',
                fontsize=20,
                fontweight='bold',
                ha='center',
                va='top',
                transform=ax.get_yaxis_transform()  # relative to y-axis ticks
            )

def plot_legend(option='case'):
    # Create a separate figure and axis for the legend
    figsize = (15, 0.5)
    legend_fig, legend_ax = plt.subplots(figsize=figsize)
    lw = 2
    markersize = 8

    if option == 'llm':
        legend_items = [

            Patch(edgecolor='black', label='fairness in representation', hatch='///', facecolor='none'),
            Patch(edgecolor='black', label='fairness in exposure', hatch='.', facecolor='none'),

        ]
        # plt.text(-0.05, 0.47, 'Legend', fontsize=10, weight='bold')
        ncol = 2

    elif option == 'gemini-flash' or option == 'llama' or option == 'gemini-pro':
        if option == 'gemini-flash':
            f_color = '#DFF2DF'
        elif option == 'llama':
            f_color = '#00CED1'
        else:
            f_color = '#FFCCFF'
        legend_items = [

            Patch(edgecolor='black', label='fairness in representation', hatch='///', facecolor=f_color),
            Patch(edgecolor='black', label='fairness in exposure', hatch='..', facecolor=f_color),
            Line2D([0], [0], lw=lw, color='red', linestyle='--', label='Ideal AveExpR'),
            Line2D([0], [0], lw=lw, color='blue', linestyle='--', label='Initial AveExpR')

        ]
        # plt.text(-0.05, 0.47, 'Legend', fontsize=10, weight='bold')
        ncol = 4
    elif option == 'no_notion':
        legend_items = [Patch(edgecolor='black', label='Gemini 1.5 Flash', hatch='', facecolor='#DFF2DF'),
                        Patch(edgecolor='black', label='Gemini 1.5 Pro', hatch='', facecolor='#FFCCFF'),
                        Patch(edgecolor='black', label='Llama-3-8B-Instruct', hatch='', facecolor='#00CED1'),
                        Line2D([0], [0], lw=lw, color='red', linestyle='--', label='Ideal AveExpR'),
                        Line2D([0], [0], lw=lw, color='blue', linestyle='--', label='Initial AveExpR')
                        ]
        ncol = 4

    elif option == 'all_rep':
        legend_items = [Patch(edgecolor='black', label='Gemini 1.5 Flash', hatch=' ', facecolor='#DFF2DF'),
                        Patch(edgecolor='black', label='Gemini 1.5 Pro', hatch=' ', facecolor='#FFCCFF'),
                        Patch(edgecolor='black', label='Llama-3-8B-Instruct', hatch=' ', facecolor='#00CED1'),
                        Patch(edgecolor='black', label='fairness in exposure', hatch='..', facecolor='none'),
                        Line2D([0], [0], lw=lw, color='red', linestyle='--', label='Ideal AveExpR'),
                        Line2D([0], [0], lw=lw, color='blue', linestyle='--', label='Initial AveExpR')
                        ]
        ncol = 3


    else:
        if option == 'llm-all-no_flash':
            ncol = 6
            # Create custom legend items using matplotlib.patches.Patch
            legend_items = [

                Patch(edgecolor='black', label='fairness in representation', hatch='///', facecolor='none'),
                Patch(edgecolor='black', label='fairness in exposure', hatch='..', facecolor='none'),
                # Patch(edgecolor='black', label='Gemini 1.5 Flash', hatch='', facecolor='#DFF2DF'),
                Patch(edgecolor='black', label='Gemini 1.5 Pro', hatch='', facecolor='#FFCCFF'),
                Patch(edgecolor='black', label='Llama-3-8B-Instruct', hatch='', facecolor='#00CED1'),
                Line2D([0], [0], lw=lw, color='red', linestyle='--', label='Ideal AveExpR'),
                Line2D([0], [0], lw=lw, color='blue', linestyle='--', label='Initial AveExpR')
                # Patch(edgecolor='black', label='Initial', hatch='', facecolor='#FFC725'),
                # Patch(edgecolor='black', label='DetConstSort', hatch='', facecolor='#4682B4')

            ]

        else:

            # option == 'pareto'
            legend_items = [Line2D([0], [0], color='#F00000', lw=lw, marker='*', markersize=markersize,
                                   label='LTR $g_{dis}\\leftrightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#F00000', lw=lw, marker='o', markersize=markersize,
                                   label='LTR $g_{dis}\\rightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#F00000', lw=lw, marker='^', markersize=markersize,
                                   label='LTR $g_{dis}\\leftarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3D6D9E', lw=lw, marker='*', markersize=markersize,
                                   label='FairLTR $g_{dis}\\leftrightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3D6D9E', lw=lw, marker='o', markersize=markersize,
                                   label='FairLTR $g_{dis}\\rightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3D6D9E', lw=lw, marker='^', markersize=markersize,
                                   label='FairLTR $g_{dis}\\leftarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#FFC725', lw=lw, marker='*', markersize=markersize,
                                   label='LTR-FairRR $g_{dis}\\leftrightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#FFC725', lw=lw, marker='o', markersize=markersize,
                                   label='LTR-FairRR $g_{dis}\\rightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#FFC725', lw=lw, marker='^', markersize=markersize,
                                   label='LTR-FairRR $g_{dis}\\leftarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3EC372', lw=lw, marker='*', markersize=markersize,
                                   label='Hidden-FairRR $g_{dis}\\leftrightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3EC372', lw=lw, marker='o', markersize=markersize,
                                   label='Hidden-FairRR $g_{dis}\\rightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3EC372', lw=lw, marker='^', markersize=markersize,
                                   label='Hidden-FairRR $g_{dis}\\leftarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='k', lw=lw, marker='*', markersize=markersize,
                                   label='Oblivious-FairRR $g_{dis}\\leftrightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='k', lw=lw, marker='o', markersize=markersize,
                                   label='Oblivious-FairRR $g_{dis}\\rightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='k', lw=lw, marker='^', markersize=markersize,
                                   label='Oblivious-FairRR $g_{dis}\\leftarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='darkorange', lw=lw, marker='+', markersize=markersize,
                                   label='Oblivious',
                                   markerfacecolor='darkorange', linestyle=' '),
                            Line2D([0], [0], color='#6600CC', lw=lw, marker='+', markersize=markersize,
                                   label='Hidden',
                                   markerfacecolor='#6600CC', linestyle=' ')

                            ]
        #ncol = 7

    plt.axis('off')

    # Add the legend to the separate legend axis
    legend_ax.legend(handles=legend_items, loc='center', ncol=ncol, edgecolor='k')
    plt.tight_layout()

    plt.savefig('legend_' + str(option) + '.pdf')

    return

def plot_accuracy_from_file(file_path, output_path='accuracy_plot.png', shot=None, experiment=experiment_name, llm=None):
    accuracies = []
    # Updated regex to strictly match valid decimal numbers
    pattern = re.compile(r'is:\s*([0-9]+(?:\.[0-9]+)?)$')

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                accuracies.append(float(match.group(1)))
    color_dict = {'gemini-2.0-flash-thinking-exp-01-21': '#009600','deepseek-api/API': '#00B0B3', 'Meta-Llama-3-8B-Instruct': '#FFCCFF'}
    llm_dict = {'gemini-2.0-flash-thinking-exp-01-21': 'Gemini 2.0 Flash', 'deepseek-api/API': 'DeepSeek', 'Meta-Llama-3-8B-Instruct': 'Llama-3'}
    exp_dict = {'BostonMarathon': 'BM', 'COMPASEX': 'COMPAS', 'NBAWNBA':'(W)NBA'}

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, marker='o', linestyle='-', color=color_dict[llm])
    print('accuracies = ', accuracies)
    plt.title(f'Accuracy, {llm_dict[llm]}, {shot}-shot, {exp_dict[experiment_name]}', fontsize=30)
    plt.xlabel('Run', fontsize=30)
    plt.ylabel('Inference Accuracy', fontsize=30)
    #increase tick size
    plt.xticks(fontsize=20, weight = 'bold')
    plt.yticks(fontsize=20, weight='bold')
    # set range of y axis
    plt.ylim(0, 1.1)
    plt.grid(True)
    #tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close()



## shared settings
llms = ['gemini-2.0-flash-thinking-exp-01-21', 'deepseek-api', 'Meta-Llama-3-8B-Instruct' ]

is_zero = [True, False]

cot_option = ['option_3']
exp_option = ['option_1']
impp_option = ['option_2a']
impb_option = ['option_2b']

#this is a dict of the notion and the corresponding options
options_s_option = [
    {'options': {'COT_FEA': cot_option}},
    {'options': {'EXP_FEA': exp_option}},
    {'options': {'IMPA_FEA': impb_option}},
    {'options': {'IMPP_FEA': impp_option}}
]
all_options=['option_1', 'option_2a', 'option_2b', 'option_3']
specific_prompts=['prompt_8', 'prompt_NA', 'prompt_NAD']
special_titles = {'COT_FEA': 'COT-FairRank', 'EXP_FEA': 'EXP-FairRank', 'IMPA_FEA': 'IMP-FairRank-A',
                                  'IMPP_FEA': 'IMP-FairRank-B'}
################################################################
# experiment section 1 - inference tables (found in paper)
################################################################

###############################################################################################
# experiment section 2 - Evaluating Fair Re-Ranking Capabilities of LLMs (no shot, no inference)
###############################################################################################
# notion = 'EXP_FEA'
specific_inf_apps='inf_Gender'
def Plot(size, experiment_section = 2):
    ###############################################################################################
    # experiment section B - Evaluating Fair Re-Ranking Capabilities of LLMs (no shot, no inference)
    ###############################################################################################
    exp = experiment_section
    if exp==2:
        zero = True
        opt = exp_option
        notion = 'EXP_FEA'

        plot_metrics(exp, size, zero_only=zero, non_zero_only=False, llm='All LLMs', metric='AveExpR CI-95',
                     specific_prompts=specific_prompts, multi_llm=True, notion=notion, option=opt, special_title=special_titles[notion], specific_inf_apps=specific_inf_apps)
        plot_metrics(exp, size, zero_only=zero, non_zero_only=False, llm='All LLMs', metric='NDKL',
                 specific_prompts=specific_prompts, multi_llm=True, notion=notion, option=opt, special_title=special_titles[notion],specific_inf_apps=specific_inf_apps)
    elif exp==3 or exp==4:
        ################################################################################################
        # experiment section C - valuating Fair Re-Ranking Strategies (in uncertain demographic attributes) of LLMs (no shots)
        # experiment section D - Evaluating the Effect of Shots on Fair Re-Ranking Strategies (in uncertain demographic attributes) of LLMs
        ###############################################################################################
        zero=True if experiment_section==3 else False
        for llm in llms:
            exp = experiment_section
            plot_metrics(exp,size, zero_only=zero, non_zero_only=False, llm=llm, metric='AveExpR CI-95'
                         ,specific_prompts=specific_prompts)
            plot_metrics(exp, size, zero_only=zero, non_zero_only=False, llm=llm, metric='NDKL',
                         specific_prompts=specific_prompts)
            plot_ndcgs(exp, size, zero_only=zero, non_zero_only=False, all_prompts=False, specific_llm=llm,
                       specific_prompts=specific_prompts)
    elif exp==5:
        ###############################################################################################
        # experiment section E - Assessing Backbone Language Models for Fair Re-Ranking Applications
        ###############################################################################################
        print('waiting to add section 5')
        zero=False
        for item in options_s_option:
            for key, value in list(item['options'].items())[0:1]:
                opt = value
                notion = key
                plot_metrics(exp, size, zero_only=zero, non_zero_only=False, llm='All LLMs', metric='AveExpR CI-95',
                             specific_prompts=specific_prompts, multi_llm=True, notion=notion, option=opt, multiple_options=True)
                plot_metrics(exp, size, zero_only=zero, non_zero_only=False, llm='All LLMs', metric='NDKL',
                         specific_prompts=specific_prompts, multi_llm=True, notion=notion, option=opt, multiple_options=True)
        ##############################################################################################################
    elif experiment_section==0: #prelimiary -  to pick sizes
        zero=True
        for llm in llms[1:2]:
            plot_metrics(exp, size, zero_only=zero, non_zero_only=False, llm=llm, metric='AveExpR CI-95'
                         ,specific_prompts=specific_prompts, specific_inf_apps=specific_inf_apps, across_sizes=[20, 50, 100])
            plot_metrics(exp, size, zero_only=zero, non_zero_only=False, llm=llm, metric='NDKL'
                         , specific_prompts=specific_prompts, specific_inf_apps=specific_inf_apps,
                         across_sizes=[20, 50, 100])


#notion = 'FEA'
# for question on Evaluating Fair Re-ranking Strategies With In-Context Fairness Exemplars (without inference, i.e Ground Truth)
# notion = 'EXP_FEA'
# specific_inf_apps='inf_Gender'
# specific_prompts=['prompt_8', 'prompt_NA', 'prompt_NAD']
#
#
# llms = ['gemini-2.0-flash-thinking-exp-01-21', 'deepseek-api', 'Meta-Llama-3-8B-Instruct' ]
#
# is_zero = [True, False]
# option = exp_option
#
# def Plot(size):
#     ### FOR SINGLE OPTION and all LLMS###
#     for zero in is_zero[:1]:
#
#         for item in options_s_option:
#             for key, value in item['options'].items():
#                 opt = value
#                 notion = key
#                 special_titles = {'COT_FEA': 'COT-FairRank', 'EXP_FEA': 'EXP-FairRank', 'IMPA_FEA': 'IMP-FairRank-A',
#                                   'IMPP_FEA': 'IMP-FairRank-B'}
#
#                 plot_metrics(size, zero_only=zero, non_zero_only=False, llm='All LLMs', metric='AveExpR CI-95',
#                              specific_prompts=specific_prompts, multi_llm=True, notion=notion, option=opt, special_title=special_titles[notion], multiple_options=True, specific_inf_apps=specific_inf_apps)
#                 plot_metrics(size, zero_only=zero, non_zero_only=False, llm='All LLMs', metric='NDKL',
#                          specific_prompts=specific_prompts, multi_llm=True, notion=notion, option=opt, special_title=special_titles[notion], multiple_options=True, specific_inf_apps=specific_inf_apps)
#
#

        # for llm in llms[0:1]:
        #     ### FOR SINGLE OPTION and single LLMS###
        #     # plot_metrics(size, zero_only=zero, non_zero_only=False, llm=llm, metric='AveExpR CI-95', option=option,
        #     #              notion=notion,specific_prompts=specific_prompts, specific_inf_apps=specific_inf_apps)
        #     # plot_metrics(size, zero_only=zero, non_zero_only=False, llm=llm, metric='NDKL', option=option,
        #     #              notion=notion, specific_prompts=specific_prompts, specific_inf_apps=specific_inf_apps)
        #     plot_ndcgs(size, zero_only=zero, non_zero_only=False, all_prompts=False, specific_llm=llm,
        #                specific_prompts=specific_prompts, notion=notion, option=option, specific_inf_apps=specific_inf_apps)


            ##################################################################################################################
            ### FOR MULTIPLE OPTIONS  and single LLMS###
            ###################################################################################################################
            # plot_metrics(size, zero_only=True, non_zero_only=False, llm=llm, metric='NDKL',
            #              specific_prompts=specific_prompts, multi_llm=False, notion=notion, multiple_options=multiple_options)
            # plot_metrics(size, zero_only=True, non_zero_only=False, llm=llm, metric='AveExpR CI-95',
            #              specific_prompts=specific_prompts, multi_llm=False, notion=notion, multiple_options=multiple_options)
            # plot_metrics(size, zero_only=False, non_zero_only=False, llm=llm, metric='NDKL',
            #               specific_prompts=specific_prompts, multi_llm=False, notion=notion, multiple_options=multiple_options)
            # plot_metrics(size, zero_only=False, non_zero_only=False, llm=llm, metric='AveExpR CI-95',
            #               specific_prompts=specific_prompts, multi_llm=False, notion=notion, multiple_options=multiple_options)
            # plot_ndcgs(size, zero_only=True, non_zero_only=False, all_prompts=False, specific_llm=llm,
            #            specific_prompts=specific_prompts, notion=notion, multiple_options=multiple_options)
            # plot_ndcgs(size, zero_only=False, non_zero_only=False, all_prompts=False, specific_llm=llm,
            #            specific_prompts=specific_prompts, notion=notion, multiple_options=multiple_options)

#####################################################################################################################





    # plot_metrics(size, zero_only=True, non_zero_only=False, llm='Meta-Llama-3-8B-Instruct', metric='NDKL')
    # plot_metrics(size, zero_only=True, non_zero_only=False, llm='gemini-1.5-flash', metric='NDKL')
    # plot_ndcgs(size, zero_only=True, non_zero_only=False, prompt='', all_prompts=True, specific_llm='gemini-1.5-flash')
    # plot_ndcgs(size, zero_only=True, non_zero_only=False, prompt='', all_prompts=True, specific_llm='gemini-1.5-pro')
    # plot_ndcgs(size, zero_only=True, non_zero_only=False, prompt='', all_prompts=True, specific_llm='Meta-Llama-3-8B-Instruct')
    # plot_metrics(size, zero_only=False, non_zero_only=False, llm='gemini-1.5-flash', metric='AveExpR CI-95')
    # plot_metrics(size, zero_only=False, non_zero_only=False, llm='gemini-1.5-flash', metric='NDKL')
    # plot_metrics(size, zero_only=False, non_zero_only=False, llm='gemini-1.5-pro', metric='AveExpR CI-95')
    # plot_metrics(size, zero_only=False, non_zero_only=False, llm='gemini-1.5-pro', metric='NDKL')
    # plot_metrics(size, zero_only=False, non_zero_only=False, llm='Meta-Llama-3-8B-Instruct', metric='AveExpR CI-95')
    # plot_metrics(size, zero_only=False, non_zero_only=False, llm='Meta-Llama-3-8B-Instruct', metric='NDKL')
# # SIMPLE PROMPTS

# for llm in llms:
#
#     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='AveExpR CI-95',
#                  specific_prompts=['prompt_2\n', 'prompt_4\n', 'prompt_6\n', 'prompt_NA', 'prompt_NAD'])
#     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='NDKL',
#                  specific_prompts=['prompt_2\n', 'prompt_4\n', 'prompt_6\n', 'prompt_NA', 'prompt_NAD'])
#     # # FEATURE PROMPTS
#
#     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='AveExpR CI-95',
#                  specific_prompts=['prompt_8\n', 'prompt_10\n', 'prompt_12\n', 'prompt_NA'])
#     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='NDKL',
#                  specific_prompts=['prompt_8\n', 'prompt_10\n', 'prompt_12\n', 'prompt_NA'])
#
#     # # PROTECTED PROMPTS
#     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='AveExpR CI-95',
#                  specific_prompts=['prompt_14\n', 'prompt_16\n', 'prompt_18\n', 'prompt_NA'])
#     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='NDKL',
#                  specific_prompts=['prompt_14\n', 'prompt_16\n', 'prompt_18\n', 'prompt_NA'])
