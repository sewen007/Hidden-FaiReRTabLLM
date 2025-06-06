# from data_analysis.calculate_metrics import kendall_tau
import io
import random

import google.api_core.exceptions
import torch
import google.generativeai as genai
from pyexpat.errors import messages
from transformers import AutoTokenizer, AutoModelForCausalLM
import pathlib

from vertexai.preview import tokenization

from hf_login import CheckLogin
from .rank_gpt import create_permutation_instruction, receive_permutation
import json
import os
import re
import time
from pathlib import Path
import pandas as pd

# os.environ['TRANSFORMERS_CACHE'] = '/scratch/shared/models/'
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = 'AIzaSyAqfx1lqmB8SMSFHs3U0bzXAxUPsQvSlno'

start = time.time()
# torch.cuda.empty_cache()
# variables for GPT
api_key = os.getenv("open_ai_key")

with open('./settings.json', 'r') as f:
    settings = json.load(f)
inferred_gender = settings["READ_FILE_SETTINGS"]["INF"]
sample_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
item_type = settings["READ_FILE_SETTINGS"]["ITEM"].lower()
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"]
protected_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"].lower()
non_protected_group = settings["READ_FILE_SETTINGS"]["ADV_GROUP"].lower()

test_set = f"./Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv"

delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))

if protected_feature == "Gender":
    prot = 'sex'


def RankWithLLM_Gemini(model_name, shot_number=1, size=50, prompt_id=2, post_process=False, option='1', inf_app=None):
    """
    This function ranks the data using a language model
    :param inf_app:
    :param post_process:
    :param option: if option is 1, add protected attribute
    :param prompt_id: prompt 0 is the neutral prompt
    :param model_name: LLM model name
    :param shot_number: number of examples. Each example has the size of the rank
    :param size: size of the rank

    :return:
    """

    print('option = ', option)
    print('model_name = ', model_name)
    print('prompt_id = ', prompt_id)
    # specify the test folder. if pre_ranked experiment, use ListNet ranked data
    post_process = post_process

    if inf_app is None:
        inf_name = 'NA'
    else: inf_name = inf_app

    test_folder = f"./Datasets/{experiment_name}/Tests/size_{size}"
    results_dir = Path(
        f'./Datasets/{experiment_name}/Ranked/{model_name}/option_{option}/inf_{inf_name}/prompt_{prompt_id}/rank_size_{size}/shot_{shot_number}')
    # check test folder for all files with rank_size
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    accuracy_file_path = Path(results_dir / 'accuracy.txt')
    if accuracy_file_path.exists():
        accuracy_file_path.unlink()
    invalid_count_path = Path(results_dir / 'invalid_count.txt')
    if invalid_count_path.exists():
        invalid_count_path.unlink()
    # messages_path = Path(results_dir / 'messages.txt')
    # if messages_path.exists():
    #     messages_path.unlink()

    test_files = [f for f in os.listdir(test_folder) if
                  f'ranked_data_rank_size_{size}' in f and os.path.isfile(os.path.join(test_folder, f))]
    for file in test_files:
        # sort by score to get the ground truth
        # gt_df = pd.read_csv(os.path.join(test_folder, file))
        # gt_df = gt_df.sort_values(by=[score_column], ascending=False)
        gt_df, ranked_df = rank_with_gemini_pre(model_name, file, shot_number, size, prompt_id, test_folder,
                                     model_type='gemini', post_process=post_process, option=option, inf_app=inf_app)
        # save the ground truth
        gt_df.to_csv(os.path.join(results_dir, f'{os.path.basename(file)}_ground_truth.csv'), index=False)

        if not gt_df['doc_id'].equals(ranked_df['doc_id']):
            ranked_df.to_csv(os.path.join(results_dir, f'ranked_data_{os.path.basename(file)}_{inf_app}.csv'),
                             index=False)

        if '2' not in option:
            # save the accuracy of inference
            if option=='3':
                inf_name = model_name

            with open(accuracy_file_path, 'a') as f:
                f.write(
                    f"Accuracy of {model_name} with size {size} and number of shots {shot_number} is: {calc_accuracy(gt_df, inf_name)}\n")

def calc_accuracy(gt_df,inf_name=None):
        gt_column = gt_df[protected_feature]
        inferred_column = gt_df[inf_name]
        # Calculate the number of correct predictions
        correct_predictions = (gt_column == inferred_column).sum()
        # Calculate the total number of predictions
        total_predictions = len(gt_column)
        # Calculate the accuracy
        accuracy = correct_predictions / total_predictions
        return accuracy



def RankWithLLM_Llama_OR_DeepSeek(model_name, size=50, post_process=False, option='1', inf_app=None):
    print('option = ', option)
    if inf_app is None:
        inf_name = 'NA'
    else: inf_name = inf_app
    CheckLogin()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available, exiting")
        exit()
    # load model
    if 'deepseek' not in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 torch_dtype=torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # read the test data
    test_folder = f"./Datasets/{experiment_name}/Tests/size_{size}"
    # check test folder for all files with rank_size
    test_files = [f for f in os.listdir(test_folder) if
                  f'ranked_data_rank_size_{size}' in f and os.path.isfile(os.path.join(test_folder, f))]
    #
    for prmpt_id in range(14, 15, 2):
        for shot in shots:
            results_dir = Path(
                f'./Datasets/{experiment_name}/Ranked/{model_name}/option_{option}/inf_{inf_name}/prompt_{prmpt_id}/rank_size_{size}/shot_{shot}')
            # check test folder for all files with rank_size
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            for file in test_files:
                # sort by score to get the ground truth
                gt_df = pd.read_csv(os.path.join(test_folder, file))
                gt_df = gt_df.sort_values(by=[score_column], ascending=False)
                # save the ground truth
                gt_df.to_csv(os.path.join(results_dir, f'{os.path.basename(file)}_ground_truth.csv'), index=False)
                df = pd.read_csv(os.path.join(test_folder, file))

                item = create_items(df, shot, prmpt_id, post_process, option, inf_app, inferred_by_cot=None)

                # (1) Create permutation generation instruction
                messages = create_permutation_instruction(item=item, rank_start=0, rank_end=size,
                                                          item_type=item_type,
                                                          prompt_id=prmpt_id,
                                                          model_type=model_name, option=option)

                # TODO sample should not be hardcoded. Code based on size
                sample = (
                    '[20] > [4] > [14] > [3] > [10] > [9] > [11] > [5] > [17] > [1] > [6] > [16] > [15] > [19] > [7] '
                    '> [12]'
                    '> [2] > [8] > [13] > [18]')
                # count the number of tokens
                token_number = get_tokens_and_count(str(messages), tokenizer)
                sample_token_number = get_tokens_and_count(sample, tokenizer)

                total_tokens = token_number + sample_token_number

                if total_tokens > 128000:
                    print('total tokens = ', total_tokens)
                    print('tokens exceed 128000')
                    return
                # results_dir = Path(
                #     f'./Datasets/{experiment_name}/Ranked/{model_name}/prompt_{prmpt_id}/rank_size_{size}/shot_{shot}')

                # Create the directory if it doesn't exist
                results_dir.mkdir(parents=True, exist_ok=True)

                # save the messages to txt
                with open(results_dir / 'messages.txt', 'w') as f:
                    for message in messages:
                        f.write(f"{message}")
                        # f.write(f"{message['role']}: {message['content']}\n")

                template_prompt_pred = tokenizer.apply_chat_template(messages, tokenize=False,
                                                                     add_generation_prompt=False)
                print('template after applying chat template = ', template_prompt_pred)
                template_prompt_pred += '<|start_header_id|>assistant<|end_header_id|>\n\n'
                # print('template after adding assistant role = ', template_prompt_pred)
                inputs_template_pred = tokenizer(template_prompt_pred, add_special_tokens=False, return_tensors='pt')
                # print('inputs_template_pred = ', inputs_template_pred)
                inputs_template_pred = inputs_template_pred.to(device)
                #
                # # generate an output using the template prompt and print only the model generated tokens
                outputs_template_pred = model.generate(**inputs_template_pred, pad_token_id=tokenizer.eos_token_id,
                                                       return_dict_in_generate=True, max_length=10000000)
                generated_tokens_template_pred = outputs_template_pred.sequences[:,
                                                 inputs_template_pred.input_ids.shape[1]:]  # for decoder only models
                generated_text_template_pred = tokenizer.decode(generated_tokens_template_pred[0],
                                                                skip_special_tokens=True)
                print('generated_result = ', generated_text_template_pred)

                # Use permutation to re-rank the passage
                new_item = receive_permutation(item, generated_text_template_pred, rank_start=0, rank_end=size)

                if '3' in option:
                    # append sex to list
                    new_list = create_items(df, shot, prmpt_id, post_process, option, inf_name,
                                            inferred_by_cot=str(generated_text_template_pred))
                    # take new item add string: Q: Re-rank the list to incorporate fairness in representation with respect to sex and send to gemini
                    new_message = str(messages) + '\nA: ' + str(
                        generated_text_template_pred) + "\nNext, we insert the inferred sexes back into the list. \n\nINPUT LIST:" + str(
                        convert_hits_to_list(new_list[
                                                 'hits'])) + "\n" + f"Q: Re-rank the list to incorporate fairness in representation with respect to sex {fair_instruction(prompt_id)}" "\n\nOUTPUT FORMAT\n" + (
                                      f"The output format should be [] > [], e.g., [1] > [2]. Only return the ranking "
                                      f"results, do not say any word or explain. ")
                    with open(results_dir / 'messages-cot.txt', 'w') as f:
                        for message in new_message:
                            f.write(f"{message}")
                    template_prompt_pred = tokenizer.apply_chat_template(new_message, tokenize=False,
                                                                         add_generation_prompt=False)
                    template_prompt_pred += '<|start_header_id|>assistant<|end_header_id|>\n\n'
                    # print('template after adding assistant role = ', template_prompt_pred)
                    inputs_template_pred = tokenizer(template_prompt_pred, add_special_tokens=False,
                                                     return_tensors='pt')
                    # print('inputs_template_pred = ', inputs_template_pred)
                    inputs_template_pred = inputs_template_pred.to(device)
                    #
                    # # generate an output using the template prompt and print only the model generated tokens
                    outputs_template_pred = model.generate(**inputs_template_pred, pad_token_id=tokenizer.eos_token_id,
                                                           return_dict_in_generate=True, max_length=10000000)
                    generated_tokens_template_pred = outputs_template_pred.sequences[:,
                                                     inputs_template_pred.input_ids.shape[
                                                         1]:]  # for decoder only models
                    generated_text_template_pred = tokenizer.decode(generated_tokens_template_pred[0],
                                                                    skip_special_tokens=True)
                    print('generated_result cot = ', generated_text_template_pred)
                    new_item = receive_permutation(item, generated_text_template_pred, rank_start=0, rank_end=size)
                    # Use permutation to re-rank the passage

                # Extract information and store in a list of dictionaries
                gt_df, merged_df = extract_and_save_permutation(df, new_item, model_name, prmpt_id, shot,
                                                                size)

                merged_df.to_csv(os.path.join(results_dir, f'ranked_data_{os.path.basename(file)}_option_{option}_{inf_app}.csv'), index=False)


def rank_with_gemini_pre(model_name, file, number_of_shots=0, size=20, prompt_id=2,
                     test_folder=f'./Datasets/{experiment_name}/Tests', model_type='gpt', post_process=False,
                     option='1', inf_app=None):
    if inf_app is None:
        inf_name = 'NA'
    else: inf_name = inf_app
    # read the test data
    df = pd.read_csv(os.path.join(test_folder, file))

    # serialize the rows of the data of the DataFrame, and creates shots
    item = create_items(df, number_of_shots, prompt_id, post_process, option, inf_name, inferred_by_cot=None)

    # add pre-prompt, fairness instruction, and post-prompt
    messages = create_permutation_instruction(item=item, rank_start=0, rank_end=size, item_type=item_type,
                                              prompt_id=prompt_id, model_type=model_type, option=option, number_of_shots=number_of_shots)
    # messages = messages.replace("['", "").replace("']", "")
    print('messages = ', messages)

    # create directory for results based on n and model name
    results_dir = Path(
        f'./Datasets/{experiment_name}/Ranked/{model_name}/option_{option}/inf_{inf_name}/prompt_{prompt_id}/rank_size_{size}/shot_{number_of_shots}')

    # create the directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # save the prompt to a txt file
    with open(results_dir / 'messages.txt', 'w') as f:
        for message in messages:
            # Check if the message has 'content' before writing it
            if 'content' in message:
                # f.write(f"{message['role']}: {message['content']}\n")
                f.write(f"{message}")
            else:  # save the message as is, (no rol)
                f.write(f"{message}")

    # get LLM predicted permutation
    permutation = None
    #for attempt in range(3):
    try:
        permutation = rank_with_gemini(key=GEMINI_API_KEY, messages=messages, model_name=model_name, size=size,
                                    number_of_shots=number_of_shots, prompt_id=prompt_id, inf_name=inf_name, option=option)

        print('permutation = ', permutation)
        #print('size = ', size)
        if option == '3':
            inferred_sexes = extract_genders(permutation)
            if (
                    inferred_sexes
                    and set(inferred_sexes) != {"unknown"}
                    and len(inferred_sexes) == 50
            ):
                pass # valid gender list received
        else:
            pass  # no validation needed, success

    except google.api_core.exceptions.ResourceExhausted:
        delay = 20 + random.uniform(0, 3)
        time.sleep(delay)
    #inferred_sexes = extract_genders(permutation)
    # use permutation to re-rank the passage
    new_item = receive_permutation(item, permutation, rank_start=0, rank_end=size)
    #print('new_item = ', new_item)

    if '3' in option:
        #append sex to list
        new_list = create_items(df, number_of_shots, prompt_id, post_process, option, inf_name, inferred_by_cot=str(permutation))
        # take new item add string: Q: Re-rank the list to incorporate fairness in representation with respect to sex and send to gemini
        new_message = str(messages) +'\nA: '+ str(permutation) + "\nNext, we insert the inferred sexes back into the list. \n\nINPUT LIST:" + str(convert_hits_to_list(new_list['hits'])) +"\n" + f"Q: Re-rank the list to incorporate fairness in representation with respect to sex {fair_instruction(prompt_id)}" "\n\nOUTPUT FORMAT\n" + (
            f"The output format should be [] > [], e.g., [1] > [2]. Only return the ranking "
            f"results, do not say any word or explain. ")
        # get LLM predicted permutation
        #for attempt in range(3):
        try:
            perm = rank_with_gemini(key=GEMINI_API_KEY, messages=new_message, model_name=model_name, size=size,
                                    number_of_shots=number_of_shots, prompt_id=prompt_id, inf_name=inf_name,option=option)
            # print('permutation = ', permutation)
        except google.api_core.exceptions.ResourceExhausted:
            delay = 20 + random.uniform(0, 3)
            time.sleep(delay)
        new_item = receive_permutation(item, perm, rank_start=0, rank_end=size)
        print('option 3 new_item = ', new_item)

        # save the prompt to a txt file
        with open(results_dir / 'messages-cot.txt', 'w') as f:
            for message in new_message:
                # Check if the message has 'content' before writing it
                if 'content' in message:
                    # f.write(f"{message['role']}: {message['content']}\n")
                    f.write(f"{message}")
                else:  # save the message as is, (no rol)
                    f.write(f"{message}")


    # extract information and store

    gt_df, merged_df = extract_and_save_permutation(df, new_item, model_name, prompt_id, number_of_shots, size)
    # add extracted gender to the gt_df

    if option == '3':
        print('length of inferred sexes', len(inferred_sexes))
        print('inferred', inferred_sexes)
        print('length of gt_df', len(gt_df))
        # create a new column in gt_df and insert extracted gender
        if len(gt_df) == len(inferred_sexes):
            gt_df[model_name] = inferred_sexes
        else:
            #return empty column
            gt_df[model_name] = ['unknown'] * len(gt_df)

    return gt_df, merged_df
    # return df, df


def convert_hits_to_list(hits):
    output = []
    for i, hit in enumerate(hits, start=1):
        content = hit['content']
        # Replace "sex: X" with "inferred sex: X"
        modified = re.sub(r'sex: (male|female)', r'inferred sex: \1', content)
        output.append(f'[{i}] {modified}')
    return ' '.join(output)

def rank_with_gemini(
    key="<api_key>",
    messages=None,
    model_name="gemini-1.5-flash",
    size=50,
    number_of_shots=0,
    prompt_id=2,
    inf_name=None,
    option='1',
):
    # Get Gemini tokenizer
    #tokenizer = tokenization.get_tokenizer_for_model(model_name)
    inf_app = inf_name
    if inf_app is None:
        inf_app = 'NA'
    results_dir = Path(
        f'./Datasets/{experiment_name}/Ranked/{model_name}/option_{option}/inf_{inf_app}/prompt_{prompt_id}/rank_size_{size}/shot_{number_of_shots}')

    results_dir.mkdir(parents=True, exist_ok=True)

    # Count tokens
    #token_number = tokenizer.count_tokens(str(messages))
    #print('Number of tokens =', token_number.total_tokens)

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(str(messages))
        if response.candidates and response.candidates[0].content.parts:
            print("Response is:", response.text)
        else:
            print(f"[DEBUG] No text content returned. finish_reason={response.candidates[0].finish_reason}")

        if check_output_format(response.text, option=option)[0]:
            print('✅ Verified output format!')
            invalid_count_path = Path(results_dir / 'invalid_count.txt')

            with open(invalid_count_path, 'a') as f:
                f.write(
                    f"Valid output format, number of value numbers: {check_output_format(response.text)[1]}\n"
                )

            return response.text
        else:
            print("❌ Invalid output format.")
            with open(results_dir / 'invalid_count.txt', 'a') as f:
                f.write(
                    f"Invalid output format for {model_name} with size {size} and number of shots {number_of_shots}, number of value numbers: {check_output_format(response.text)[1]}\n"
                )

            # Retry
            time.sleep(5)
            return rank_with_gemini(key, messages, model_name, size, number_of_shots, prompt_id, inf_name, option)

    except google.api_core.exceptions.InvalidArgument as e:
        print('⚠️ Invalid argument:', e)
        time.sleep(20)
        return rank_with_gemini(key, messages, model_name, size, number_of_shots, prompt_id, inf_name, option)

    except google.api_core.exceptions.ResourceExhausted as e:
        print('⚠️ Resource exhausted:', e)
        time.sleep(20)
        return rank_with_gemini(key, messages, model_name, size, number_of_shots, prompt_id, inf_name, option)


# def rank_with_gemini(key="<api_key>", messages=None, model_name="gemini-1.5-flash",  size=50,
#     number_of_shots=0,
#     prompt_id=2):
#     # get gemini tokenizer
#     tokenizer = tokenization.get_tokenizer_for_model(model_name)
#
#     # count the number of tokens
#     token_number = tokenizer.count_tokens(str(messages))
#     print('number of tokens = ', token_number.total_tokens)
#     model = genai.GenerativeModel(model_name)
#     try:
#         # key = os.getenv('GEMINI_API_KEY')
#         genai.configure(api_key='AIzaSyCf_c4f-JaaP-l60n0nECYd1TmpCLBhdmE')
#         response = model.generate_content(str(messages))
#         # key = os.getenv('GEMINI_API_KEY')
#         # genai.configure(api_key=key)
#
#         print('response:', response.text)
#         return response.text
#     except google.api_core.exceptions.ResourceExhausted:
#         print('Resource exhausted. Trying again')
#         time.sleep(20)
#         return rank_with_gemini(key, messages, model_name)


def extract_and_save_permutation(df, new_item, model_name, prompt_id, number_of_shots=0, size=50):
    # Extract information and store in a list of dictionaries
    extracted_ranked_data = [extract_info(item['content']) for item in new_item['hits']]
    print('extracted_ranked_data = ', extracted_ranked_data)
    ranked_df = pd.DataFrame(extracted_ranked_data)
    # Convert list of tuples to DataFrame

    # order gt_df by score_column
    gt_df = df.sort_values(by=[score_column], ascending=False)

    if 'doc_id' in gt_df.columns:
        ranked_df.rename(columns={'ID': 'doc_id'}, inplace=True)
        ranked_df['doc_id'] = ranked_df['doc_id'].astype(str)
        gt_df['doc_id'] = gt_df['doc_id'].astype(str)

        merged_df = pd.merge(ranked_df, gt_df[['doc_id', score_column]], on='doc_id', how='left')

    else:
        merged_df = pd.merge(ranked_df, gt_df[['doc_id', score_column]], on='doc_id', how='left')

    return gt_df, merged_df


def extract_info(content):
    # Dictionary to store extracted values
    extracted_values = {}

    # Pattern to match key-value pairs in the content
    key_value_pattern = r'(\w+): ([\w.-]+)'

    # Extract all key-value pairs
    matches = re.findall(key_value_pattern, content)

    # Handle each key-value pair
    for key, value in matches:
        cleaned_value = value.strip()
        if key == "name":
            # Capture everything before the next comma
            name_match = re.search(r'name: (.*?)(?=,|$)', content)
            if name_match:
                cleaned_value = name_match.group(1).strip()
        extracted_values[key] = cleaned_value

    return extracted_values

def normalize_gender(raw):
    raw = raw.lower().strip()
    if raw in ['male', 'm']:
        return 'male'
    if raw in ['female', 'f']:
        return 'female'
    return 'unknown'


def extract_genders(text):
    if not text:
        return ['unknown']

    text = text.lower()

    # If LLM says something like "all unknown", catch early
    if "all entries" in text and "unknown" in text:
        return ['unknown']

    # Match gender patterns: M, F, male, female, possibly numbered or comma-separated
    pattern = r'\b(male|female|m|f|man|woman|boy|girl|unknown)\b'
    matches = re.findall(pattern, text)

    # # Normalize
    normalized = [normalize_gender(g) for g in matches]

    return normalized if normalized  else ['unknown']



def create_items(df, number_of_shots=0, prompt_id=1, post_process=False, option='1', inf_app=None, inferred_by_cot=None):
    """creates items with format item = {
    'query': 'Re-rank the following list to incorporate fairness. ',
    'hits': [
        {'content': 'Student ID: 18739 (Gender: female, score:...)}'},
        {'content': 'Student ID: 8742 (Gender: male, score:...)'},
        {'content': 'Student ID: 17967 (Gender: male, score:...)'},
        {'content': 'Student ID: 13964 (Gender: male, score:...)'}
    ]
}"""
    # for sexes gotten from cot
    gender_list = extract_genders(inferred_by_cot)
    print('number_of_shots = ', number_of_shots)
    n_rank_size = len(df)
    s_shots = number_of_shots

    # get the query
    query = prepare_fair_rerank_template(s_shots, n_rank_size, prompt_id, option, inf_app)

    # Create the hit. This is actually where the data to be ranked is serialized
    hits = []
    score_labels = {
        'LAW': ('Student ID', 'score'),
        'LOAN': ('ID', 'Applicant Income'),
        'NBAWNBA': ('ID', 'Career Points')
    }

    id_label, score_label = score_labels.get(experiment_name, ('ID', 'score'))

    # for _, row in df.iterrows():
    #     inferred_part = f", inferred sex: {row[inf_app]}" if option == '1' else ""
    #     content = f"{id_label}: {row['doc_id']} (name: {row['Name']}{inferred_part}, {score_label}: {row[score_column]})"
    #     hits.append({'content': content})  # serialize the rows of the DataFrame

    for idx, (_, row) in enumerate(df.iterrows()):
        sex = gender_list[idx] if idx < len(gender_list) else 'unknown'  # Fallback if mismatch
        inferred_part = f", inferred sex: {row[inf_app]}" if option == '1' else ""
        sex_part = f", sex: {sex}" if inferred_by_cot is not None else ""

        content = f"{id_label}: {row['doc_id']} (name: {row['Name']}{inferred_part}{sex_part}, {score_label}: {row[score_column]})"
        hits.append({'content': content})

    # Creating the final JSON object
    result = {
        'query': query,
        'hits': hits
    }
    return result


def fair_instruction(id_):
    if id_ == 2:
        return "."
    elif id_ == 4:
        return "in representation."
    elif id_ == 6:
        return "in exposure."
    elif id_ == 8:
        return f"with respect to {prot}."
    elif id_ == 10:
        return f"in representation with respect to {prot}."
    elif id_ == 12:
        return f"in exposure with respect to {prot}."
    elif id_ == 14:
        return f"with respect to {prot}, where {protected_group} is the disadvantaged group."
    elif id_ == 16:
        return (f"in representation respect to {prot}, "
                f"where {protected_group} is the disadvantaged group.")
    elif id_ == 18:
        return f"in exposure with respect to {prot}, where {protected_group} is the disadvantaged group."


def check_output_format(text, option='1'):
    if option == '3':
        return True, 0  # Skip format check for option 3

    # Find all numbers inside square brackets
    matches = re.findall(r'\[(\d+)\]', text)
    numbers = list(map(int, matches))

    has_all = set(numbers) == set(range(1, 51))
    correct_length = len(numbers) == 50

    if has_all and correct_length:
        return True, 50
    else:
        return False, len(numbers)



def prepare_fair_rerank_template(shott, size=50, prompt_id=2, option='1', inf_app=None):
    """
    "For example, .... Re-rank the following list to incorporate fairness."
    :param option:
    :param shott:
    :param size:
    :param prompt_id:
    :param inf_app:
    :return:
    """
    instruct_template = ""

    base_template = ""
    # re_rank_template = "You are given a list that is already ranked based on scores"
    re_rank_template = "\nTASK\n" + "Re-rank the following list to incorporate fairness "
    if experiment_name != ' ':  # experiment_name == 'LAW':
        instruct_template = ""

        if prompt_id == 2:
            base_template = re_rank_template
        elif prompt_id in {4, 6, 8, 10, 12, 14, 16, 18}:
            if option == '3' and shott > 0:
                base_template = ""
            else:
                base_template = re_rank_template + f"{fair_instruction(prompt_id)}"

        else:
            base_template = generate_base(prompt_id)
            instruct_template = " Rank the following applicants:"
        if not(option == '3' and shott > 0):
            base_template += "\n\nDATA\n\nINPUT LIST: "

    shot_templates = []

    shot_template = instruct_template
    if shott == 0:
        shot_template += base_template
        # print(shot_template)
    else:
        for i in range(1, shott + 1):
            # example_template = (pick_conjunction(i - 1) + f" example, given this list of {item_type}s: ")

            example_template = "\n\nEXAMPLE " + str(i) + "\n\nINPUT LIST: "
            if option =='3':
                example_template = ""

            shot_sample = pd.read_csv(
                f"./Datasets/{experiment_name}/Shots/size_{size}/Fair_Reranking/shot_{i}/ground_truth_rank_size_{size}_shot_{i}.csv")

            # reset the index
            shot_sample.reset_index(drop=True, inplace=True)

            # Create examples list from the shot_sample
            examples = [row_converter(row, post_process=True, option=option, inf_app=inf_app, is_shot=True, cot_shot=1) for index, row in
                        shot_sample.iterrows()]

            formatted_examples = [f"[{i + 1}] {item}" for i, item in enumerate(examples)]

            if option != '3':
                example_template += ' '.join(formatted_examples) + "\n\n" + "OUTPUT LIST: "
            else:
                example_template += ' '.join(formatted_examples) + "\n\nQ: Infer the sexes of the runners in the order of the list? Return only one sex per runner. \nA: Here are the inferred sexes for each runner based on their names: "
                example_template += ', '.join(shot_sample['Gender'].str.capitalize()) + "\n\nNext, we insert the inferred sexes into the list. \n\nINPUT LIST: "
                example_with_sexes= [row_converter(row, post_process=True, option=option, inf_app=inf_app, is_shot=True, cot_shot=2) for index, row in
                        shot_sample.iterrows()]
                formatted_examples_2 = [f"[{i + 1}] {item}" for i, item in enumerate(example_with_sexes)]
                example_template +=' '.join(formatted_examples_2) + "\n" + f"Q: Re-rank the list to incorporate fairness {fair_instruction(prompt_id)}"
                example_template += "\n\nOUTPUT FORMAT\n" + (
                    f"The output format should be [] > [], e.g., [1] > [2]. Only return the ranking "
                    f"results, do not say any word or explain. \n\nA:The answer is ")


            # Get the row numbers of shuffled_sample based on the order in shot_sample

            fair_ranked_sample = pd.read_csv(
                f"./Datasets/{experiment_name}/Shots/size_{size}/Fair_Reranking/shot_{i}/ranked_data_rank_size_{size}_shot_{i}.csv")

            # create a mapping of items to their original indices in shot_example
            items_to_index = shot_sample['doc_id'].reset_index().set_index('doc_id')['index']

            # use the mapping to get the indices of items in fair_examples based on shot_example
            reordered_indices = fair_ranked_sample['doc_id'].map(items_to_index)

            formatted_output = " > ".join(map(lambda x: f"[{x}]", reordered_indices.values + 1))

            example_template += formatted_output
            shot_template = shot_template + example_template + "\n\n"

        shot_template += base_template.replace("['", "").replace("']", "")
    shot_templates.append(shot_template)

    return shot_templates[0].replace("['", "").replace("']", "") + ""


def pick_conjunction(i):
    conjunction_options = [" For", " Another", " Yet another", " And another"]
    if i == 0:
        return conjunction_options[0]
    elif i == 1:
        return conjunction_options[1]
    elif i == 2:
        return conjunction_options[2]
    else:
        return conjunction_options[3]


def row_converter(row, post_process=False, option='1', inf_app=None, is_shot=False, cot_shot=None):
    if experiment_name == 'LAW':
        if post_process:
            if option == '1':  # add protected attribute
                if is_shot:
                    return "Student ID: " + str(row['doc_id']) + " (" + "name: " + str(
                        row['Name']) + ", sex: " + str(row['Gender']) + ", score: " + str(row[score_column]) + ")"
                else:
                    return "Student ID: " + str(row['doc_id']) + " (" + "name: " + str(
                    row['Name']) + ", inferred sex: " + str(
                    row[inf_app]) + ", score: " + str(row[score_column]) + ")"
            elif option == '2a':
                if is_shot:
                    return "Student ID: " + str(row['doc_id']) + " (" + "name: " + str(row['Name']) + ", sex: " + str(row['Gender']) +  ", score: " + str(
                        row[score_column]) + ")"
                else:
                    return "Student ID: " + str(row['doc_id']) + " (" + "name: " + str(row['Name']) + ", score: " + str(
                        row[score_column]) + ")"
            elif option == '2b':
                return "Student ID: " + str(row['doc_id']) + " (" + "name: " + str(row['Name']) + ", score: " + str(
                        row[score_column]) + ")"
            elif option == '3':
                if cot_shot==1:
                    return "Student ID: " + str(row['doc_id']) + " (" + "name: " + str(
                        row['Name']) + ", score: " + str(row[score_column]) + ")"
                elif cot_shot==2:
                    return "Student ID: " + str(row['doc_id']) + " (" + "name: " + str(
                        row['Name']) + ", inferred sex: " + str(row['Gender']) + ", score: " + str(row[score_column]) + ")"

        # else:
        #     return "Student ID: " + str(row['doc_id']) + " (" + "sex: " + str(row['Gender']) + ", UGPA: " + str(
        #         row['UGPA']) + (
        #         ",LSAT: ") + str(row['LSAT']) + ")"

    else:
        return create_content(row, option=option, inf_app=inf_app, is_shot=is_shot, cot_shot=cot_shot)


# def create_content(row, option=1):
#     # content_parts = [f"Unique ID: {row['doc_id']}"]
#     content_parts = []
#     for column in row.index:
#         if option == 1:
#             content_parts = [f"ID: {row['doc_id']} ", f"(name: {row['Name']},sex: {row['Gender']}, "]
#         else:
#             content_parts = [f"ID: {row['doc_id']} ", f"(sex: {row['Gender']}, "]
#
#         if experiment_name == 'NBAWNBA':
#             content_parts.append(f"Career Points: {row['GT_score']}")
#         else:
#             content_parts.append(f"score: {row['GT_score']}")
#     content_parts += "),"  # add a closing bracket
#     content = "".join(content_parts)
#     return content
def create_content(row, option='1', inf_app=None, is_shot=False, cot_shot=None):
    content_parts = [f"ID: {row['doc_id']} "]

    if option == '1':
        if is_shot:
            content_parts.append(f"(name: {row['Name']}, sex: {row['Gender']}, ")
        else:
            content_parts.append(f"(name: {row['Name']}, inferred sex: {row[inf_app]}, ")
    elif option == '2a':
        if is_shot:
            content_parts.append(f"(name: {row['Name']}, sex: {row['Gender']}, ")
        else:
            content_parts.append(f"(name: {row['Name']}, ")
    elif option == '2b':
        # is_shot should have no sex
        content_parts.append(f"(name: {row['Name']}, ")
    elif option == '3':
        if cot_shot == 1:
            content_parts.append(f"(name: {row['Name']}, ")
        elif cot_shot == 2:
            content_parts.append(f"(name: {row['Name']}, inferred sex: {row['Gender']}, ")

    score_label = "Career Points" if experiment_name == "NBAWNBA" else "score"
    content_parts.append(f"{score_label}: {row['GT_score']})")

    return "".join(content_parts)


def generate_base(prompt_id):  # hardcoded for now. Considering having multiple queries
    """
    this should only give us the base template
    :param prompt_id:
    :return:
    """
    if experiment_name == 'LAW':

        return ("You are the admissions ranking system for a Law school; their goal is to rank a list of applicants to "
                "predict their success in the program. The school wants to rank the applicants using their UGPA score "
                "and LSAT scores and gender. Rank the following applicants in a predicted order of success in the "
                "program:")
    else:
        return " "


def get_tokens_and_count(string, tokenizer):
    tokens = tokenizer.encode(string)
    return len(tokens)


end = time.time()

print("time taken = ", end - start)
