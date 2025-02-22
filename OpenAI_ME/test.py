import pandas as pd
import time
from run_openai_ME import run_me_caller


def conv_openAI_ME_data(ME_responses):
    """Converts ME responses to bool values and makes new csv"""
    conv_ME_output = [1 if ME_response["results"][0]["flagged"] else 0 for ME_response in ME_responses]

    return conv_ME_output

# read in data
dataset = pd.read_csv("/Users/daviddai/PycharmProjects/AI_Auditing/RawData/movie_TV_raw_data.csv")
save_path = "/Users/daviddai/PycharmProjects/AI_Auditing/OpenAI_ME/movie_TV_ME_Feb.csv"

# openAI call
print("Starting ME call...")
start = time.time()
OpenAI_ME_responses = run_me_caller(dataset, "content")
dataset["OpenAI_ME_responses"] = OpenAI_ME_responses[0]
dataset["OpenAI_ME_bool_Feb"] = conv_openAI_ME_data(OpenAI_ME_responses[0])
dataset['OpenAI_data'] = OpenAI_ME_responses[1]
dataset.to_csv(save_path, index=False)
print("Elapsed time:", time.time() - start)
