import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer
from nltk import PorterStemmer
from sys import exit
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch.nn.functional as F
pd.options.mode.chained_assignment = None

MAX_LEN = None
enable_pos = None
enable_att = None
length_factor = None
tokenizer = None

def init(setting_dict):
    '''
    Init template, max length and tokenizer.
    '''

    global MAX_LEN, tokenizer
    global enable_pos, enable_att, length_factor

    MAX_LEN = setting_dict["max_len"]
    enable_pos = setting_dict["enable_pos"]
    enable_att = setting_dict["enable_att"]
    length_factor = setting_dict["length_factor"]

    tokenizer = T5Tokenizer.from_pretrained("t5-" + setting_dict["model"], model_max_length=MAX_LEN)


def get_PRF(num_c, num_e, num_s):
    F1 = 0.0
    P = float(num_c) / float(num_e) if num_e!=0 else 0.0
    R = float(num_c) / float(num_s) if num_s!=0 else 0.0
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1

def print_PRF(P, R, F1, N):
    print(f"\nN={N}")
    print(f"P={P}")
    print(f"R={R}")
    print(f"F1={F1}\n")
    return 0



def keyphrases_selection(setting_dict, doc_list, labels_stemed, labels, dataloader, std_layer_product, att_weight, model, device):
    """
    Main function for keyphrase selection and evaluation.
    """
    
    # Initialize settings
    init(setting_dict)
    model.eval()

    cos_similarity_list = {}
    candidate_list = []
    cos_score_list = []
    doc_id_list = []
    pos_list = []
    att_score_list = []
    whole_att_score_list = []
    length_list = []
    pred_labels = {}
    
    num_c_5 = num_c_10 = num_c_15 = 0  
    num_e_5 = num_e_10 = num_e_15 = 0 
    num_s = 0  

    # Process each batch in the dataloader
    for id, [input_ids, input_mask, dic] in enumerate(tqdm(dataloader, desc="Evaluating:")):
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        score = np.zeros(input_ids.shape[0])
        
        kwargs = {
            "cand_pos_s": dic["t5_pos_s"],
            "cand_pos_e": dic["t5_pos_e"],
            "temp_start_idx": dic["temp_start_idx"],
            "cand_start_idx": dic["start_idx"],
            "initial_std": 0.3,
            "std_layer_product": std_layer_product,
            "topk": 15,
            "sim_word_idx": dic["sim_word_idx"],
        }

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=input_mask, output_attentions=True, **kwargs)
            logits = output.logits  # (B, L, V)
            log_probs = F.log_softmax(logits, dim=-1)

            score = torch.zeros(input_ids.size(0))

            # Group candidates by start index
            group_by_start = defaultdict(list)
            for j in range(input_ids.size(0)):
                group_by_start[dic["start_idx"][j]].append(j)

            # Calculate scores for each candidate group
            for start, indices in group_by_start.items():
                for j in indices:
                    cand_len = dic["candidate_len"][j]
                    
                    pos_indices = torch.arange(start-2, start + cand_len)
                
                    # Target token IDs that should appear at the next position
                    token_ids = input_ids[j, pos_indices + 1]
                    
                    # Extract log probabilities for candidate positions
                    log_probs_j = log_probs[j, pos_indices, :]  # (cand_len, V)
                    token_scores = log_probs_j.gather(1, token_ids.unsqueeze(1)).squeeze(1)  # (cand_len,)
            
                    # Calculate normalized score by candidate length
                    score_j = token_scores.sum() / ((cand_len+2) ** length_factor)
                    score[j] = score_j
               
            # Collect results for current batch
            doc_id_list.extend(dic["idx"])
            candidate_list.extend(dic["candidate"])
            cos_score_list.extend(score)
            pos_list.extend(dic["pos"])
            att_score_list.extend(dic["att_score"])
            whole_att_score_list.extend(dic["whole_att_score"])
            length_list.extend(dic["candidate_len"])

    # Organize results into DataFrame
    cos_similarity_list["doc_id"] = doc_id_list
    cos_similarity_list["candidate"] = candidate_list
    cos_similarity_list["score"] = cos_score_list
    cos_similarity_list["pos"] = pos_list
    cos_similarity_list["att_score"] = att_score_list
    cos_similarity_list["candidate_len"] = length_list
    cos_similarity_list["whole_att_score"] = whole_att_score_list
    cosine_similarity_rank = pd.DataFrame(cos_similarity_list)

    # Process results for each document
    for i in range(len(doc_list)):
        temp_label = []
        tt_label = []
        doc_len = len(doc_list[i].split())
        
        doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id'] == i]
        
        # Apply position-based scoring if enabled
        if enable_pos == True:
            position_factor = 1.2e8
            doc_results["pos"] = doc_results["pos"] / doc_len + position_factor / (doc_len ** 3)
            doc_results["score"] = doc_results["pos"] * doc_results["score"]

        # Apply attention-based scoring if enabled
        if enable_att == True: 
            doc_results["att_score"] = (doc_results["att_score"] * (doc_results["whole_att_score"])) 
            doc_results["length"] = doc_results["candidate_len"].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
            att_scores_float = doc_results["att_score"].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
            att_scores_clipped = att_scores_float.clip(upper=2.5)
            doc_results["att_score"] = att_scores_clipped
            
            # Adjust scores based on attention weights
            doc_results["score"] = doc_results["score"] / (1 + att_weight * np.log1p(doc_results["att_score"]))   
           
        # Rank keyphrases by score
        ranked_keyphrases = doc_results.sort_values(by='score', ascending=False)
        top_k = ranked_keyphrases.reset_index(drop=True)
        top_k_can = top_k.loc[:, ['candidate']].values.tolist()
  
        # Remove duplicate candidates
        candidates_set = set()
        candidates_dedup = []
        for temp in top_k_can:
            temp = temp[0].lower()
            if temp in candidates_set:
                continue
            else:
                candidates_set.add(temp)
                candidates_dedup.append(temp)

        # Evaluate top-15 predictions against ground truth
        j = 0
        Matched = candidates_dedup[:15]
        porter = PorterStemmer()
        
        for id, temp in enumerate(candidates_dedup[0:15]):
            tokens = temp.split()
            tt = ' '.join(porter.stem(t) for t in tokens) 
            
            if (tt in labels_stemed[i] or temp in labels[i]):
                Matched[id] = [temp]
                temp_label.append(temp)
                tt_label.append(tt)
                
                if (j < 5):
                    num_c_5 += 1
                    num_c_10 += 1
                    num_c_15 += 1
                elif (j < 10 and j >= 5):
                    num_c_10 += 1
                    num_c_15 += 1
                elif (j < 15 and j >= 10):
                    num_c_15 += 1
            j += 1

        if (len(top_k[0:5]) == 5):
            num_e_5 += 5
        else:
            num_e_5 += len(top_k[0:5])

        if (len(top_k[0:10]) == 10):
            num_e_10 += 10
        else:
            num_e_10 += len(top_k[0:10])

        if (len(top_k[0:15]) == 15):
            num_e_15 += 15
        else:
            num_e_15 += len(top_k[0:15])

        num_s += len(labels[i])

   
 
    # Calculate and print evaluation metrics
    # Precision, Recall, F1-score for top-5, top-10, and top-15
    p, r, f = get_PRF(num_c_5, num_e_5, num_s)
    print_PRF(p, r, f, 5)

    p, r, f = get_PRF(num_c_10, num_e_10, num_s)
    print_PRF(p, r, f, 10)

    p, r, f = get_PRF(num_c_15, num_e_15, num_s)
    print_PRF(p, r, f, 15)
    


