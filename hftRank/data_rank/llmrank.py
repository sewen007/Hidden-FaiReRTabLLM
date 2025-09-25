# from data_analysis.calculate_metrics import kendall_tau
import ast
import io
import random

import google.api_core.exceptions
import torch
import google.generativeai as genai
from openai import OpenAI
from pyexpat.errors import messages
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
import pathlib

from transformers.models.auto.image_processing_auto import model_type
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

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY ="AIzaSyCwP2tWCBN5wuurnPKNnCp9V8FPfw7vZFE"
#DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = "sk-18be54ed718345ecb741f50d33a7fb21"
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
current_size = settings["GENERAL_SETTINGS"]["current_rank_size"]
test_set = f"./Datasets/{experiment_name}/Testing/Testing_{experiment_name}.csv"

delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))

if protected_feature == "Gender":
    prot = 'sex'


def RankWithLLM_Gemini_or_Deepseek(model_name, shot_number=1, size=50, prompt_id=2, post_process=False, option='1', inf_app=None):
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
    model_save = model_name
    if 'api.deepseek' in model_name:
        model_save = 'deepseek-api/API'
    test_folder = f"./Datasets/{experiment_name}/Tests/size_{size}"
    results_dir = Path(
        f'./Datasets/{experiment_name}/Ranked/{model_save}/option_{option}/inf_{inf_name}/prompt_{prompt_id}/rank_size_{size}/shot_{shot_number}')
    # check test folder for all files with rank_size
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    accuracy_file_path = Path(results_dir / 'accuracy.txt')
    if accuracy_file_path.exists():
        accuracy_file_path.unlink()
    invalid_count_path = Path(results_dir / 'invalid_count.txt')
    if invalid_count_path.exists():
        invalid_count_path.unlink()
    inference_path = Path(results_dir / 'inference.txt')
    if inference_path.exists():
        inference_path.unlink()
    # messages_path = Path(results_dir / 'messages.txt')
    # if messages_path.exists():
    #     messages_path.unlink()

    test_files = [f for f in os.listdir(test_folder) if
                  f'ranked_data_rank_size_{size}' in f and os.path.isfile(os.path.join(test_folder, f))]
    model_type = None
    if 'gemini' in model_name:
        model_type = 'gemini'
    elif 'deepseek' in model_name:
        model_type = 'deepseek'
    for file in test_files:
        # sort by score to get the ground truth
        # gt_df = pd.read_csv(os.path.join(test_folder, file))
        # gt_df = gt_df.sort_values(by=[score_column], ascending=False)
        gt_df, ranked_df = rank_with_gemini_pre(model_name, file, shot_number, size, prompt_id, test_folder,
                                     model_type=model_type, post_process=post_process, option=option, inf_app=inf_app, save_inf_path=inference_path)
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

def safe_rank_with_gemini(key, messages, model_name, size, number_of_shots, prompt_id, inf_name, option, max_retries=5):
    """
    Wrapper to call rank_with_gemini safely with retry/backoff.
    """
    for attempt in range(max_retries):
        try:
            return rank_with_gemini(
                key=key,
                messages=messages,
                model_name=model_name,
                size=size,
                number_of_shots=number_of_shots,
                prompt_id=prompt_id,
                inf_name=inf_name,
                option=option
            )
        except google.api_core.exceptions.ResourceExhausted as e:
            # Suggested delay from API if available, else exponential backoff
            retry_delay = getattr(e, 'retry_delay', None)
            wait_time = retry_delay.seconds if retry_delay else (2 ** attempt) + random.uniform(0, 3)
            print(f"[429 Rate Limit] Waiting {wait_time:.1f}s before retry (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[ERROR] Unexpected exception: {e}")
            raise
    raise RuntimeError("Max retries exceeded for Gemini API call")

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



def RankWithLLM_Llama(model_name, size=50, post_process=False, option='1', inf_app=None):
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
    elif 'Llama-4-Scout-17B-16E-Instruct' in model_name:
        processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
        model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to GPU (if available)

    # read the test data
    test_folder = f"./Datasets/{experiment_name}/Tests/size_{size}"
    # check test folder for all files with rank_size
    test_files = [f for f in os.listdir(test_folder) if
                  f'ranked_data_rank_size_{size}' in f and os.path.isfile(os.path.join(test_folder, f))]
    #
    for prmpt_id in range(8, 9, 2):
        print('prompt_id = ', prmpt_id)
        print('model_name = ', model_name)
        for shot in shots:
            results_dir = Path(
                f'./Datasets/{experiment_name}/Ranked/{model_name}/option_{option}/inf_{inf_name}/prompt_{prmpt_id}/rank_size_{size}/shot_{shot}')
            # check test folder for all files with rank_size
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            accuracy_file_path = Path(results_dir / 'accuracy.txt')
            if accuracy_file_path.exists():
                accuracy_file_path.unlink()
            invalid_count_path = Path(results_dir / 'invalid_count.txt')
            if invalid_count_path.exists():
                invalid_count_path.unlink()
            inference_path = Path(results_dir / 'inference.txt')
            if inference_path.exists():
                inference_path.unlink()
            # if not os.path.exists(results_dir):
            #     os.makedirs(results_dir)

            for file in test_files:
                # sort by score to get the ground truth
                gt_df = pd.read_csv(os.path.join(test_folder, file))
                gt_df = gt_df.sort_values(by=[score_column], ascending=False)
                # save the ground truth
                gt_df.to_csv(os.path.join(results_dir, f'{os.path.basename(file)}_ground_truth.csv'), index=False)
                df = pd.read_csv(os.path.join(test_folder, file))

                item = create_items(df, shot, prmpt_id, post_process, option, inf_app, inferred_by_cot=None, save_inf_path=inference_path)

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

                generated_text_template_pred = generate_until_valid(model, tokenizer, inputs_template_pred)
                print('generated_result = ', generated_text_template_pred)
                if option == '3':
                    # save permutation to file
                    with open(inference_path, 'a') as f:
                        f.write(f"{generated_text_template_pred}\n")
                    inferred_sexes = extract_genders(generated_text_template_pred)
                    if (
                            inferred_sexes
                            and set(inferred_sexes) != {"unknown"}
                            and len(inferred_sexes) == current_size
                    ):
                        pass  # valid gender list received
                    print('length of inferred sexes', len(inferred_sexes))
                    print('inferred', inferred_sexes)
                    print('length of gt_df', len(gt_df))
                    # create a new column in gt_df and insert extracted gender
                    if len(gt_df) == len(inferred_sexes):
                        gt_df[model_name] = inferred_sexes
                    else:
                        # return empty column
                        gt_df[model_name] = ['unknown'] * len(gt_df)
                else:
                    pass  # no validation needed, success



                # Use permutation to re-rank the passage
                new_item = receive_permutation(item, generated_text_template_pred, rank_start=0, rank_end=size)

                if '3' in option:
                    # append sex to list
                    new_list = create_items(df, shot, prmpt_id, post_process, option, inf_name,
                                            inferred_by_cot=str(generated_text_template_pred), save_inf_path=inference_path)

                    new_message = messages + [{'role': 'user',
                 'content': '\nA: ' + str(
                        generated_text_template_pred) + "\nNext, we insert the inferred sexes back into the list. \n\nINPUT LIST:" + str(
                        convert_hits_to_list(new_list[
                                                 'hits'])) + "\n" + f"Q: Re-rank the list to incorporate fairness in representation with respect to sex {fair_instruction(prmpt_id)}" "\n\nOUTPUT FORMAT\n" + (
                                      f"The output format should be [] > [] > [] > ... , e.g., [1] > [2] > [20] > ... Only return the ranking "
                                      f"results, do not say any word or explain. ")}]
                    with open(results_dir / 'messages-cot.txt', 'w') as f:
                        for message in new_message:
                            f.write(f"{message}")
                    template_prompt_pred = tokenizer.apply_chat_template(new_message, tokenize=False,
                                                                         add_generation_prompt=False)
                    template_prompt_pred += '<|start_header_id|>assistant<|end_header_id|>\n\n'
                    # print('template after adding assistant role = ', template_prompt_pred)
                    inputs_template_pred = tokenizer(template_prompt_pred, add_special_tokens=False,
                                                     return_tensors='pt')

                    inputs_template_pred = inputs_template_pred.to(device)
                    #

                    generated_text_template_pred = generate_until_valid(model, tokenizer, inputs_template_pred)
                    #
                    new_item = receive_permutation(item, generated_text_template_pred, rank_start=0, rank_end=size)
                    # Use permutation to re-rank the passage
                    inf_cal = model_name

                with open(accuracy_file_path, 'a') as f:
                    f.write(
                        f"Accuracy of {model_name} with size {size} and number of shots {shot} is: {calc_accuracy(gt_df, inf_cal)}\n")

                # Extract information and store in a list of dictionaries
                gt_df, merged_df = extract_and_save_permutation(df, new_item, model_name, prmpt_id, shot,
                                                                size)

                merged_df.to_csv(os.path.join(results_dir, f'ranked_data_{os.path.basename(file)}_option_{option}_{inf_app}.csv'), index=False)

def generate_until_valid(model, tokenizer, inputs_template_pred, max_attempts=5):
    attempt = 0
    success = False
    generated_text_template_pred = None

    while not success and attempt < max_attempts:
        try:
            outputs_template_pred = model.generate(
                **inputs_template_pred,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                max_length=10000000
            )

            generated_tokens_template_pred = outputs_template_pred.sequences[:, inputs_template_pred["input_ids"].shape[1]:]
            generated_text_template_pred = tokenizer.decode(generated_tokens_template_pred[0], skip_special_tokens=True)

            if check_output_format(generated_text_template_pred):
                print("✅ Output format is valid.")
                success = True
            else:
                print(f"❌ Attempt {attempt + 1}: Output format invalid. Retrying...")

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}: Error during generation or validation: {e}")

        attempt += 1

    if success:
        print("👉 Generated valid output:\n")
    else:
        print("❗ Max attempts reached. Returning last generated output despite format issues.")

    print(generated_text_template_pred)
    return generated_text_template_pred

def rank_with_gemini_pre(model_name, file, number_of_shots=0, size=20, prompt_id=2,
                     test_folder=f'./Datasets/{experiment_name}/Tests', model_type='gpt', post_process=False,
                     option='1', inf_app=None, save_inf_path=None):
    if inf_app is None:
        inf_name = 'NA'
    else: inf_name = inf_app
    # read the test data
    df = pd.read_csv(os.path.join(test_folder, file))

    # serialize the rows of the data of the DataFrame, and creates shots
    item = create_items(df, number_of_shots, prompt_id, post_process, option, inf_name, inferred_by_cot=None, save_inf_path=save_inf_path)

    # add pre-prompt, fairness instruction, and post-prompt
    messages = create_permutation_instruction(item=item, rank_start=0, rank_end=size, item_type=item_type,
                                              prompt_id=prompt_id, model_type=model_type, option=option, number_of_shots=number_of_shots)
    # messages = messages.replace("['", "").replace("']", "")
    print('messages = ', messages)

    # create directory for results based on n and model name
    model_save = model_name
    if 'api.deepseek' in model_name:
        model_save = 'deepseek-api/API'

    results_dir = Path(
        f'./Datasets/{experiment_name}/Ranked/{model_save}/option_{option}/inf_{inf_name}/prompt_{prompt_id}/rank_size_{size}/shot_{number_of_shots}')

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
            # save permutation to file
            with open(save_inf_path, 'a') as f:
                f.write(f"{permutation}\n")
            inferred_sexes = extract_genders(permutation)
            if (
                    inferred_sexes
                    and set(inferred_sexes) != {"unknown"}
                    and len(inferred_sexes) == current_size
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
        new_list = create_items(df, number_of_shots, prompt_id, post_process, option, inf_name, inferred_by_cot=str(permutation), save_inf_path=save_inf_path)
        # take new item add string: Q: Re-rank the list to incorporate fairness in representation with respect to sex and send to gemini
        if 'deepseek' in model_name:
            new_message = messages + [{'role': 'user',
                                         'content': '\nA: ' + str(
                                             permutation) + "\nNext, we insert the inferred sexes back into the list. \n\nINPUT LIST:" + str(
                                             convert_hits_to_list(new_list[
                                                                      'hits'])) + "\n" + f"Q: Re-rank the list to incorporate fairness in representation with respect to sex {fair_instruction(prompt_id)}" "\n\nOUTPUT FORMAT\n" + (
                                                        f"The output format should be [] > [] > [] > ... , e.g., [1] > [2] > [20] > ... Only return the ranking "
                                                        f"results, do not say any word or explain. ")}]
        else:

            new_message = str(messages) +'\nA: '+ str(permutation) + "\nNext, we insert the inferred sexes back into the list. \n\nINPUT LIST:" + str(convert_hits_to_list(new_list['hits'])) +"\n" + f"Q: Re-rank the list to incorporate fairness in representation with respect to sex {fair_instruction(prompt_id)}" "\n\nOUTPUT FORMAT\n" + (
                f"The output format should be [] > [] > [] > ... , e.g., [1] > [2] > [20] > ... Only return the ranking "
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

def rank_with_gemini(key="<api_key>", messages=None, model_name="gemini-1.5-flash", size=50, number_of_shots=0, prompt_id=2,
    inf_name=None, option='1'):
    from pathlib import Path
    import time
    import google.generativeai as genai
    import os
    from openai import OpenAI  # for deepseek
    import google.api_core.exceptions

    if inf_name is None:
        inf_app = 'NA'
    else:
        inf_app = inf_name

    model_save = model_name
    if 'api.deepseek' in model_name:
        model_save = 'deepseek-api/API'

    results_dir = Path(
        f'./Datasets/{experiment_name}/Ranked/{model_save}/option_{option}/inf_{inf_app}/prompt_{prompt_id}/rank_size_{size}/shot_{number_of_shots}')
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        if 'gemini' in model_name:
            key = GEMINI_API_KEY
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(str(messages))

            # ✅ Check that there is at least one candidate before accessing text
            if hasattr(response, "candidates") and response.candidates:
                try:
                    response_text = response.text  # this is safe now
                    print('response =', response_text)
                    response = response_text
                except Exception as e:
                    print("⚠️ Failed to access response.text:", str(e))
                    time.sleep(5)
                    return rank_with_gemini(key, messages, model_name, size, number_of_shots, prompt_id, inf_name,
                                            option)
            else:
                print("❌ Gemini response has no candidates.")
                print("Input messages were:", messages)
                time.sleep(5)
                return rank_with_gemini(key, messages, model_name, size, number_of_shots, prompt_id, inf_name, option)

        elif 'deepseek' in model_name:
            key = DEEPSEEK_API_KEY
            # print('key = ', key)
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            response = response.choices[0].message.content
            print('response = ', response)

        if check_output_format(response, option=option)[0]:
            print('✅ Verified output format!')
            invalid_count_path = Path(results_dir / 'invalid_count.txt')

            with open(invalid_count_path, 'a') as f:
                f.write(
                    f"Valid output format, number of value numbers: {check_output_format(response)[1]}\n"
                )

            return response
        else:
            print("❌ Invalid output format.")
            with open(results_dir / 'invalid_count.txt', 'a') as f:
                f.write(
                    f"Invalid output format for {model_name} with size {size} and number of shots {number_of_shots}, number of value numbers: {check_output_format(response)[1]}\n"
                )

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

def normalize_gender(g):
    g = g.lower()
    if g in ['m', 'male', 'man', 'boy']:
        return 'male'
    elif g in ['f', 'female', 'woman', 'girl']:
        return 'female'
    elif g == 'unknown':
        return 'unknown'
    return 'unknown'


def extract_genders(text, max_count=50):
    if not text:
        return ['unknown']

    text = text.lower().strip()

    # Catch cases like "all unknown"
    if "all entries" in text and "unknown" in text:
        return ['unknown']

    # Match all gender-related words, but only up to max_count
    pattern = r'\b(male|female|m|f|man|woman|boy|girl|unknown)\b'
    matches = re.findall(pattern, text)

    # Normalize and trim to desired length
    normalized = [normalize_gender(g) for g in matches]
    return normalized[:max_count] if normalized else ['unknown']

def create_items(df, number_of_shots=0, prompt_id=1, post_process=False, option='1', inf_app=None, inferred_by_cot=None, save_inf_path=None):
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
    # save inferred sexs to file
    # inference_path = Path(save_inf_path)
    # with open(inference_path, 'a') as f:
    #     f.write(f"{gender_list}\n")
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

    has_all = set(numbers) == set(range(1, current_size+1))
    correct_length = len(numbers) == current_size

    if has_all and correct_length:
        return True, current_size
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
                    f"The output format should be [] > [] > [] > ... , e.g., [1] > [2] > [20] > ... Only return the ranking "
                    f"results, do not say any word or explain. \n\nA:The answer is ")



            fair_ranked_sample = pd.read_csv(f"./Datasets/{experiment_name}/Shots/size_{size}/Fair_Reranking/shot_{i}/ranked_data_rank_size_{size}_shot_{i}.csv")

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
