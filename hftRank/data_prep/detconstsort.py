# adopted from https://github.com/evijit/SIGIR_FairRanking_UncertainInference
import csv
import json
import math
import os
import re
from collections import defaultdict as ddict
import operator
import pandas as pd

# from learning_to_rank.listwise import ListNet as ln


with open('./settings.json', 'r') as f:
    settings = json.load(f)
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]


def get_files(directory):
    temp = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            temp.append(os.path.join(dirpath, file))

    return temp


def detconstsort(a, k_max, p):
    """ This function ranks the documents using the DetConstSort algorithm
    :param a: dictionary of attributes
    :param k_max: maximum number of documents to rank
    :param p: dictionary of protected attribute distribution"""
    scores = []
    for a_i in a.keys():
        for i_d, score in a[a_i].items():
            scores.append((a_i, i_d, score))
    attributes = a.keys()
    attribute_scores = {}

    # create and initialize counter for each attribute value
    counts_ai = {}
    minCounts_ai = {}
    totalCounts_ai = {}
    for a_i in a.keys():
        counts_ai[a_i] = 0
        minCounts_ai[a_i] = 0
        totalCounts_ai[a_i] = len(a[a_i])

    re_ranked_attr_list = {}
    re_ranked_score_list = {}
    maxIndices = {}

    lastEmpty = 0
    k = 0

    for i, a_i in enumerate(attributes):
        counts_ai[a_i] = 0
        minCounts_ai[a_i] = 0
        totalCounts_ai[a_i] = sum([1 for s in scores if s[0] == a_i])
        attribute_scores[a_i] = [(s[2], s[1]) for s in scores if
                                 s[0] == a_i]

    # print(attribute_scores)

    while lastEmpty <= k_max:

        if lastEmpty == len(scores):
            break

        k += 1
        tempMinAttrCount = ddict(int)
        changedMins = {}
        for a_i in attributes:
            tempMinAttrCount[a_i] = math.floor(k * p[a_i])
            if minCounts_ai[a_i] < tempMinAttrCount[a_i] and minCounts_ai[a_i] < totalCounts_ai[a_i]:
                changedMins[a_i] = attribute_scores[a_i][counts_ai[a_i]]

        if len(changedMins) != 0:
            ordChangedMins = sorted(changedMins.items(), key=lambda x: x[1][0], reverse=True)
            for a_i in ordChangedMins:
                re_ranked_attr_list[lastEmpty] = a_i[0]
                lastEmpty = int(lastEmpty)
                # print('here', attribute_scores[a_i[0]][counts_ai[a_i[0]]])
                re_ranked_score_list[lastEmpty] = attribute_scores[a_i[0]][counts_ai[a_i[0]]]
                maxIndices[lastEmpty] = k
                start = lastEmpty
                while start > 0 and maxIndices[start - 1] >= start and re_ranked_score_list[start - 1][0] < \
                        re_ranked_score_list[start][0]:
                    swap(re_ranked_score_list, start - 1, start)
                    swap(maxIndices, start - 1, start)
                    swap(re_ranked_attr_list, start - 1, start)
                    start -= 1
                counts_ai[a_i[0]] += 1
                lastEmpty += 1
            minCounts_ai = dict(tempMinAttrCount)

    re_ranked_attr_list = [re_ranked_attr_list[i] for i in sorted(re_ranked_attr_list)]
    re_ranked_score_list = [re_ranked_score_list[i] for i in sorted(re_ranked_score_list)]

    return re_ranked_attr_list, re_ranked_score_list


def swap(temp_list, pos_i, pos_j):
    temp = temp_list[pos_i]
    temp_list[pos_i] = temp_list[pos_j]
    temp_list[pos_j] = temp


def wrapper(url):
    """
    This is the wrapper code to convert detlr output to input 1 for detconstsort_rank
    :param url: url pointing to deltr output
    :return:
    """
    a = {}
    df = pd.read_csv(url)
    dff = df.groupby('Gender')
    for row in df['Gender']:
        # check if prediction column is present
        if 'predictions' in df.columns:
            a[row] = dict(zip(dff.get_group(row).doc_id, dff.get_group(row).predictions))
        else:  # use score column
            a[row] = dict(zip(dff.get_group(row).doc_id, dff.get_group(row)[score_column]))
    return a


def getdist(df):
    # Given the ranked dataframe, return the true protected attr dist as a dictionary
    d = {}
    for index, row in df.iterrows():
        if row["Gender"] not in d:
            d[row['Gender']] = 1
        else:
            d[row['Gender']] += 1
    for attr in d:
        d[attr] = d[attr] / len(df)
    return d


def find_unaware_ranked(file):
    match = re.search('gamma=0.0', file)
    if match:
        return True
    else:
        return False


def writeRanked(writefile, dict):
    # field names
    fields = ['doc_id', 'Name', 'Gender', 'predictions', 'GT_score', 'Gender']

    # writing to csv file
    with open(writefile, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        for player in dict.keys():
            csvwriter.writerow(dict.get(player))
    print("SUCCESS! Saved to: " + writefile)


def infer_with_detconstsort(file, rank_test=False, post_process=False, test_data=False, p_value=None):
    # read the data
    data = pd.read_csv(file)

    # order the ranking data by score
    gt_data = data.sort_values(by=score_column, ascending=False)

    if rank_test:
        write_path = './Datasets/' + experiment_name + '/Ranked/DCS/prompt_NAD/rank_size_20/shot_NAD/'
    elif post_process:
        if test_data:
            # check size from basename
            match = re.search(r'size_(\d+)', os.path.basename(file))
            write_path = './Datasets/' + experiment_name + '/Ranked/DetConstSort/prompt_NAD/rank_size_' + match.group(
                1) + '/shot_NAD/'
            if not os.path.exists(write_path):
                os.makedirs(write_path)
        else:
            # search for number between shot_ and . in file name
            match = re.search(r'shot_(\d+)', os.path.basename(file))
            match_size = re.search(r'size_(\d+)', os.path.basename(file))
            write_path = f'./Datasets/{experiment_name}/Shots/size_{match_size.group(1)}/Fair_Reranking/shot_' + match.group(1) + '/'
            if not os.path.exists(write_path):
                os.makedirs(write_path)
    else:
        write_path = './Datasets/' + experiment_name + '/Shots/Fair/'

    ranked_dict = {}

    write_file = write_path + os.path.basename(file)

    a = wrapper(file)

    # # p = {'male': 0.1, 'female': 0.9}
    # if experiment_name == 'NBAWNBA':
    #     p = {'female': 0.9, 'male': 0.1}
    # else:
    #     p = getdist(data)
    k_max = len(data.index)
    if p_value:
        p = p_value
    else:
        p = getdist(data)

    result = detconstsort(a, k_max, p)
    result_genders = result[0]

    # get the scores and doc_id
    result_scores = result[1]
    print('result_scores', result_scores)

    for i in range(k_max):
        if result_scores[i][1] not in ranked_dict.keys():
            # include ground truth score for utility calculation later
            gt_score = gt_data.loc[gt_data['doc_id'] == result_scores[i][1], score_column].iloc[0]
            # return groundtruth gender for fairness calculation later
            gt_gender = gt_data.loc[gt_data['doc_id'] == result_scores[i][1], 'Gender'].iloc[0]
            gt_name = gt_data.loc[gt_data['doc_id'] == result_scores[i][1], 'Name'].iloc[0]
            # get other attributes from the ground truth data.
            ranked_dict[result_scores[i][1]] = [result_scores[i][1], gt_name, gt_gender, result_scores[i][0], gt_score,
                                                result_genders[i]]


        else:
            print("There are duplicates in the ranking, something went wrong.")
            return

    writeRanked(write_file, ranked_dict)


# Test_folder = '../Datasets/' + experiment_name + '/Tests/LTR_ranked/'
# # select all files in the test folder
# all_files = get_files(Test_folder)
# for file in all_files:
#     # path = os.path.join(Test_folder, file)
#     infer_with_detconstsort(file, rank_test=True)

# DetConstSort()
