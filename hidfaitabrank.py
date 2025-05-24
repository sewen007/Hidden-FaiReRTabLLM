from hftRank import *


options = ['1', '2a','2b','3']
inf_apps = ['GAPI', 'NMSOR', 'BTN', 'Gender']
meta_apps = ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct']
# # # # # #
# 1. Prepare data. Data is set for experiments
# # # # # #
# Clean()
# Prep(size=50)
# # # # # #


# # # # # # #
# 2. For Gemini and Deepseek
# # # # # # #
# api_apps = ['https://api.deepseek.com', 'gemini-2.0-flash-thinking-exp-01-21']
# for api_app in api_apps[1:]:
#     for option in options[2:]:
#         for shot in shots:
#             for prmpt_id in range(8, 9, 2):
#                 print('prompt_id = ', prmpt_id)
#                 # print('shot = ', shot)
#
#                 #inf_apps = [ 'BTN', 'Gender']
#                 if option=='1':
#                     for inf_app in inf_apps[2:]:
#                         print('inf_app = ', inf_app)
#                         RankWithLLM_Gemini_or_Deepseek(api_app, shot_number=shot, size=50, prompt_id=prmpt_id, post_process=True, option=option, inf_app=inf_app)
#
#                 else:
#                     RankWithLLM_Gemini_or_Deepseek(api_app, shot_number=shot, size=50, prompt_id=prmpt_id, post_process=True, option=option, inf_app=None)

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
# # # # #
# for meta_exp in meta_apps[:1]:
#
#     CalculateResultMetrics(meta_exp, size=50)
#     # # # #
#     Collate(meta_exp, prompt_remove=['prompt_1'])
    #
#     # # # #
#     # Plot results
# Plot(size=50)