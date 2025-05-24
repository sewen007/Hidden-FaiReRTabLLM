# functions contained in the file are adapted from sunnweiwei's code. https://github.com/sunnweiwei/RankGPT
import copy
import time
import json
import os
import re

from openai import OpenAI

with open('./settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
dp_api_key = os.getenv("DEEP_SEEK_API_KEY")


class OpenaiClient:
    def __init__(self, keys=None, start_id=None, proxy=None):
        from openai import OpenAI
        import openai
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        self.api_key = self.key[self.key_id % len(self.key)]
        self.client = OpenAI(api_key=self.api_key)

    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        print('completion', completion)
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.completions.create(
                    *args, **kwargs
                )
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion


def get_prefix_and_suffix_prompt(query, hits, rank_end, num, Item, model_type, prompt_id, option, number_of_shots=0):
    # TASK is contained in query
    rerank_suffix = ""
    if '3' in option:
        if number_of_shots != 0:
            rerank_suffix = {'role': 'system',
                 'content': f"Given these examples, solve the following task:  \n"}

    sex_infered_statement = ", sex inferred using the name," if option == '1' else ""
    # post_prompt is now included in this function
    from .llmrank import fair_instruction as fi
    rank = 0
    rank_start = 0
    combined_content = ""  # Initialize an empty string to store all content

    # Process hits
    for hit in hits[rank_start: rank_end]:
        rank += 1
        content = hit.get('content', '').strip()  # Ensure content is a valid string
        if content:
            combined_content += f"[{rank}] {content} "  # Append each hit's content to the combined_content string
    if option != '3':
        combined_content += "\n\nOUTPUT FORMAT\n" + (
            f"The output format should be [] > [] > [] > ... , e.g., [1] > [2] > [20] > ... Only return the ranking "
            f"results, do not say any word or explain.")
        # if model_type == "meta-llama/Meta-Llama-3-8B-Instruct":
        #     combined_content += "\n\nOUTPUT FORMAT\n" + (
        #         f"The output format should be [] > [] > [] > ... , e.g., [1] > [2] > [20] > ... Only return the ranking "
        #         f"results, do not say any word or explain.")
        # else:
        #     combined_content += "\n\nOUTPUT FORMAT\n" + (
        #         f"The output format should be [] > [], e.g., [1] > [2]. Only return the ranking "
        #         f"results, do not say any word or explain.")
        instruct_prompt = f"{fi(prompt_id)}{query} {combined_content.strip()}"
    else: # option == '3'
        instruct_prompt = f"{fi(prompt_id)}\n\nINPUT LIST: {combined_content.strip()}""\n\n" + "Q: Infer the sexes of the runners in the order of the list? Return only one sex per runner. "
    if option == '3' and number_of_shots > 0:
        add_query = extract_chunks_after_answer(query)
    else:
        add_query = ""

    if experiment_name == 'LAW':
        base_system = {'role': 'system',
                 'content': "You are a fair ranking system. "}
        base_user = f"You are the ranking officer for a Law school and your goal is to rank {Item}s by "\
                            "predicted success in the program, while ensuring fairness. "\
            f"I will provide a list of {num} {Item}s, each described by a sequential index (e.g., [1]), an ID," \
                 f"a name{sex_infered_statement} and a score. The list is already ranked by success scores, and you are to re-rank to incorporate fairness "

    elif experiment_name == 'BostonMarathon':
        base_system = {
            'role': 'system',
            'content': "You are a fair ranking system. "
        }

        base_user = f"You are the ranking officer for a marathon and your goal is to rank {Item}s by "\
                       f"predicted success in the marathon, while ensuring fairness. "\
            f"I will provide a list of {num} {Item}s, each described by a sequential index (e.g., [1]), " \
                    f"an ID, a name{sex_infered_statement} and a score representing performance. " \
                    f"The list is already ranked by official time scores, from fastest to slowest, and you are to re-rank " \
                    f"to incorporate fairness"


    elif experiment_name == 'COMPASSEX':
        base_system =  {'role': 'system',
                 'content': "You are a fair ranking system. " }
        base_user =  f"You are the parole officer for criminal offenders and your goal is to rank {Item}s by "\
                            f"predicted likelihood of re-offending, while ensuring fairness. "\
            f"I will provide a list of {num} {Item}s, each described by a sequential"\
                            f" index (e.g., [1]), an ID, a name{sex_infered_statement} and a score. The list is already "\
                            f"ranked by"\
                            f" recidivism scores, and you are to re-rank to incorporate fairness"
    elif experiment_name == 'NBAWNBA':
        base_system =  {'role': 'system',
                 'content': "You are a fair ranking system. "}
        base_user = f"You are the ranking officer for a national basketball showcase. Your goal is to rank "\
                            f"{Item}s based on "\
                            f"predicted performance, while ensuring fairness. " \
            f"I will provide a list of {num} {Item}s, each described by a sequential"\
                            f" index (e.g., [1]), an ID, a name{sex_infered_statement} and a score. The list is already "\
                            f"ranked by"\
                            f" career points, and you are to re-rank to incorporate fairness "
    elif experiment_name == 'LOAN':
        base_system = {'role': 'system',
                 'content': "You are a fair ranking system. "}
        base_user = f"You are the loan officer for a bank and your goal is to rank loan applicants by "\
                            f"predicted likelihood of default, while ensuring fairness. "\
            f"I will provide a list of {num} {Item}s, each described by a sequential"\
                            f" index (e.g., [1]), a name{sex_infered_statement} and the applicant's income. The list is "\
                            f"already ranked by"\
                            f" applicant income, and you are to re-rank to incorporate fairness {instruct_prompt}{add_query}"
    result = [ base_system,
        {'role': 'user', 'content': f"{base_user} {instruct_prompt}{add_query}"}
    ]

    # If in few-shot mode, add another example
    if option == '3' and number_of_shots > 0:
        # create empty list to store the results
        result = []
        i=1
        for subquery in add_query:
            example_prefix = {'role': 'system',
                              'content': f"EXAMPLE {i} \n"}
            subquery = ".\n\nDATA\nINPUT LIST: " + subquery
            result.append(example_prefix)
            result.append(base_system)
            result.append(
                {'role': 'user', 'content': f"{base_user} {subquery}"}
            )
            i+=1
        result.append(rerank_suffix)
        result.append(base_system)
        result.append({'role': 'user', 'content': f"{base_user} {instruct_prompt}"})

        # result = [
        #     base_system,
        #     {'role': 'user', 'content': f"{base_user} {instruct_prompt}{add_query}"},
        #     base_system,
        #     {'role': 'user', 'content': f"{base_user} {instruct_prompt}"}
        # ]
        # result += "\nA:"

    return result

def extract_chunks_after_answer(text):
    chunks = []
    start = 0

    # Find all occurrences of "the answer is"
    for match in re.finditer(r"The answer is", text):
        idx = match.start()

        # Find the next \n\n after this point
        next_break = text.find("\n\n", idx)
        if next_break == -1:
            next_break = len(text)

        # Slice from current start to after the next \n\n
        chunk = text[start:next_break + 2]  # include the \n\n
        chunks.append(chunk)
        start = next_break + 2  # continue from after the current double newline

    return chunks


def create_permutation_instruction(item=None, rank_start=0, rank_end=50, item_type='applicant', prompt_id=1,
                                   model_type=None, option='1', number_of_shots=0):
    # pre-prompt, fairness instruction, shots and post-prompt
    if not isinstance(item, dict):
        print('Invalid item, create items function not working')
        exit()
    query = item.get('query', '')  # Ensure query is a valid string
    hits = item.get('hits', [])  # Ensure hits is a valid list
    num = len(hits[rank_start: rank_end])

    messages = get_prefix_and_suffix_prompt(query, hits, rank_end, num, item_type, model_type,
                                 prompt_id, option, number_of_shots)  # Get prefix prompt messages
    if model_type == 'gemini':
        if messages and messages[0] == {'role': 'system', 'content': 'You are a fair ranking system'}:
            messages = messages[1:]
        messages = [message for message in messages]

    # Ensure prefix prompt messages are valid
    for message in messages:
        if 'content' not in message or not isinstance(message['content'], str):
            print('No content available')

    # Final check to ensure all messages have valid content
    valid_messages = [message for message in messages if
                      'content' in message and isinstance(message['content'], str) and message['content'].strip()]
    # Extract the content values and concatenate them into a single string
    add_s = ""
    if number_of_shots>1:
        add_s = "s"
    if model_type == 'gemini':
        concatenated_content = ' '.join([message['content'].replace("'", '"')
                                         for i, message in enumerate(valid_messages)])
        if option == '3' and number_of_shots >0:
            concatenated_content = f"You are a reranking agent. I will give {number_of_shots} example{add_s} on how to follow a least to most method to reranking \n" + concatenated_content

        return concatenated_content
    else:
        if option == '3' and number_of_shots >0:
            pre_valid_messages = {'role': 'user',
                 'content': f"You are a reranking agent. I will give {number_of_shots} example{add_s} on how to follow a least to most method to reranking \n"}
            valid_messages = [pre_valid_messages] + valid_messages
        return valid_messages


def get_post_prompt(query, num, Item):
    return (f"The output format should be [] > [] > [] > ... , e.g., [1] > [2] > [20] > ... Only return the ranking "
            f"results, do not say any word or explain.")


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    for idx, entry in enumerate(cut_range, start=1):  # Start IDs from 1
        # Extract the current content
        content = entry['content']
        # Remove the extra '(' before 'sex' and prepend the ID
        content = content.replace("(sex:", "sex:")  # Remove the extra '('
        # Prepend the ID to the content
        entry['content'] = f"(ID: {idx}, {content[1:]}"
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
        if 'CareerPoints' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['CareerPoints'] = cut_range[j]['CareerPoints']
    return item
