from hftRank import *

# Clean()
# exit()

# one time run - bm prep - not needed to reproduce experiments
#Temp_clean_bm_nba()
gemini_app = 'gemini-2.0-flash-thinking-exp-01-21'
#deep_seek_app = 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
deep_seek_app = 'deepseek-ai/DeepSeek-V2-Lite'
meta_exp = 'meta-llama/Meta-Llama-3-8B-Instruct'
# Prep(size=50)
# exit()
# #option='3'
options = ['1','2a','2b','3']
# #options = ["1"]
# # #
for option in options:
    for shot in shots:
        for prmpt_id in range(14, 15, 2):
            print('prompt_id = ', prmpt_id)
            # print('shot = ', shot)
            inf_apps = ['GAPI', 'NMSOR', 'BTN', 'Gender']
            #inf_apps = ['Gender']
            if option=='1':
                for inf_app in inf_apps:
                    RankWithLLM_Llama_OR_DeepSeek(deep_seek_app, size=50, post_process=True, option=option, inf_app=inf_app)
                    #RankWithLLM_Llama_OR_DeepSeek(meta_exp, size=50, post_process=True, option=option,
                      #                            inf_app=inf_app)
                    #RankWithLLM_Gemini(gemini_app, shot_number=shot, size=50, prompt_id=prmpt_id, post_process=True, option=option,
                    #                  inf_app=inf_app)
            else:
                #RankWithLLM_Gemini(gemini_app, shot_number=shot, size=50, prompt_id=prmpt_id, post_process=True, option=option, inf_app=None)
                RankWithLLM_Llama_OR_DeepSeek(deep_seek_app, size=50, post_process=True, option=option, inf_app=None)
                #RankWithLLM_Llama_OR_DeepSeek(meta_exp, size=50, post_process=True, option=option, inf_app=None)
#
# meta_exp = 'meta-llama/Meta-Llama-3-8B-Instruct'
# # # # # # # # # #
# #CalculateResultMetrics(meta_exp, size=50)
# # # # #
# Collate(meta_exp, prompt_remove=['prompt_1'])
# #
# Plot(size=50, meta_app=meta_exp)