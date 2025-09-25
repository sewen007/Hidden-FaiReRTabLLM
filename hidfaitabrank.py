from hftRank import *


options = ['1', '2a','2b','3']
inf_apps = ['GAPI', 'NMSOR', 'BTN', 'Gender']
meta_apps = ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct']

rank_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
current_size = settings["GENERAL_SETTINGS"]["current_rank_size"]


# # # # # #
# 1. Prepare data. Data is set for experiments
# # # # # #
# Clean()
# Prep(size=current_size)
# # # # # #


# # # # # # #
# 2. For Gemini and Deepseek
# # # # # #
# for size in rank_sizes:
#
#     api_apps = ['https://api.deepseek.com', 'gemini-2.0-flash-thinking-exp-01-21']
#     for api_app in api_apps[0:1]:
#         for option in options[:1]:
#             for shot in shots[2:]:
#                 for prmpt_id in range(8, 9, 2):
#                     print('prompt_id = ', prmpt_id)
#                     # print('shot = ', shot)
#
#                     #inf_apps = [ 'BTN', 'Gender']
#                     if option=='1':
#                         for inf_app in inf_apps[3:4]:
#                             print('inf_app = ', inf_app)
#                             RankWithLLM_Gemini_or_Deepseek(api_app, shot_number=shot, size=size, prompt_id=prmpt_id, post_process=True, option=option, inf_app=inf_app)
#
#                     else:
#                         RankWithLLM_Gemini_or_Deepseek(api_app, shot_number=shot, size=size, prompt_id=prmpt_id, post_process=True, option=option, inf_app=None)

# # # # # #
#3.  For Meta-Llama #GPUS needed

# # # # # # # #
# for meta_exp in meta_apps[1:]:
#     print('meta_exp = ', meta_exp)
#     for option in options:
#         print('option = ', option)
#         if option=='1':
#             for inf_app in inf_apps:
#                 print('inf_app = ', inf_app)
#                 RankWithLLM_Llama(meta_exp, size=50, post_process=True, option=option, inf_app=inf_app)
#         else:
#             RankWithLLM_Llama(meta_exp, size=50, post_process=True, option=option, inf_app=None)



# # # # #
# 4. Process and Plot results
# # # #
# for meta_exp in meta_apps[:1]:
#     for sizee in rank_sizes:
#         current_size = sizee
#
#         # CalculateResultMetrics(meta_exp, size=current_size)
#         # # # # run this only once or you get multiple columns. Delete collated file before rerun
#         Collate(meta_exp, prompt_remove=['prompt_1'])
        #
    # # # #
    # Plot results
experiment_sections = [2] #2 = Section B, 3 = Section C,...
for sizee in rank_sizes:
    for experiment_section in experiment_sections:
        current_size = sizee
        Plot(size=current_size, experiment_section = experiment_section)

# This eas run only once to append race to compas dataset
#append_race()