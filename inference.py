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
enable_freq = None
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
    


def calculate_score(setting_dict, cosine_similarity_rank,doc_list,labels,labels_stemed,enable_pos,enable_att,enable_length,length_factor,weight,position_factor2,length_factor2,method):
    init(setting_dict)
    num_c_5 = num_c_10 = num_c_15 = 0
    num_e_5 = num_e_10 = num_e_15 = 0
    num_s = 0
    new_cand_score = []
    for i in range(len(doc_list)):
            
        doc_len = len(doc_list[i].split())
        
        doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id']==i]
        if enable_pos == True:
            #doc_results.loc[:,"pos"] = torch.Tensor(doc_results["pos"].values.astype(float)) / doc_len + position_factor / (doc_len ** 3)
            doc_results["pos"] = doc_results["pos"] / doc_len + position_factor2 / (doc_len ** 3)
            doc_results["score"] = doc_results["pos"] * doc_results["score"]
        #* doc_results["score"].values.astype(float)
        if enable_att == True: 
            att_scores = doc_results["att_score"]
            att_scores = torch.tensor(att_scores.tolist(), dtype=torch.float64)
            
            if method == 0:
                doc_results["att_score"] = doc_results["att_score"]
            elif method == 1:
                doc_results["att_score"] = doc_results["whole_att_score"]
            else:
                # doc_results["att_score"] = doc_results["att_score"]*1.7
                doc_results["att_score"] = (doc_results["att_score"] * (doc_results["whole_att_score"])) 
            doc_results["length"] = doc_results["candidate_len"].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
            length = np.minimum(doc_results["length"], 4)
            doc_results["att_score"] = (length ** length_factor) * doc_results["att_score"]

            att_scores_float = doc_results["att_score"].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
            att_scores_clipped = att_scores_float.clip(upper=2.5)
            doc_results["att_score"] = att_scores_clipped
            doc_results["score"] = doc_results["score"] / (1 + weight * np.log1p(doc_results["att_score"]))   
            # doc_results["score"] = (1-weight)*doc_results["score"] + (weight) * doc_results["att_score"]
        if enable_length == True:   
            cand_len = doc_results["candidate_len"].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)            
            doc_results["score"] = doc_results["score"] / ((cand_len+2) ** length_factor2) 
        ranked_keyphrases = doc_results.sort_values(by='score', ascending=False)
        doc_results2 = doc_results.sort_values(by="score", ascending=False).drop_duplicates(subset=["candidate"], keep="first")
        labels_col = []
        porter = PorterStemmer()
        for _, row in doc_results2.iterrows():
            candidate = row["candidate"]
            
            # candidate의 stemming 처리
            tokens = candidate.split()
            candidate_stemmed = ' '.join(porter.stem(t) for t in tokens)
            
            # 매칭 여부 확인
            if candidate_stemmed in labels_stemed[i] or candidate in labels[i]:
                labels_col.append(1)
            else:
                labels_col.append(0)

        doc_results2["label"] = labels_col
        new_cand_score.append(doc_results2)

        
        top_k = ranked_keyphrases.reset_index(drop = True)
        top_k_can = top_k.loc[:, ['candidate']].values.tolist()
        #print(top_k)
        #exit()
        candidates_set = set()
        candidates_dedup = []
        for temp in top_k_can:
            temp = temp[0].lower()
            if temp in candidates_set:
                continue
            else:
                candidates_set.add(temp)
                candidates_dedup.append(temp)

        # log.logger.debug("Sorted_Candidate: {} \n".format(top_k_can))
        # log.logger.debug("Candidates_Dedup: {} \n".format(candidates_dedup))

        j = 0
        Matched = candidates_dedup[:15]
        porter = PorterStemmer()
        for id, temp in enumerate(candidates_dedup[0:15]):
            tokens = temp.split()
            tt = ' '.join(porter.stem(t) for t in tokens)
            if (tt in labels_stemed[i] or temp in labels[i]):
                Matched[id] = [temp]
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
        

    p5, r5, f5 = get_PRF(num_c_5, num_e_5, num_s)
    p10, r10, f10 = get_PRF(num_c_10, num_e_10, num_s)
    p15, r15, f15 = get_PRF(num_c_15, num_e_15, num_s)
    print(f5,f10,f15)
        
    new_cand_scores = pd.concat(new_cand_score, ignore_index=True) 
    return {
        'f5': round(f5 * 100, 2),
        'f10': round(f10 * 100, 2),
        'f15': round(f15 * 100, 2),
    }, new_cand_scores



def calculate_score2_with_results(setting_dict, cosine_similarity_rank, doc_list, labels, labels_stemed, 
                                 enable_pos, enable_att, enable_freq, length_factor, weight, position_fac, method):
    """
    기존 calculate_score2 함수를 수정하여 결과를 반환하도록 함
    """
    init(setting_dict)
    num_c_5 = num_c_10 = num_c_15 = 0
    num_e_5 = num_e_10 = num_e_15 = 0
    num_s = 0
    
    for i in range(len(doc_list)):
        doc_len = len(doc_list[i].split())
        
        doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id']==i]
        if enable_pos == True:
            doc_results["pos"] = doc_results["pos"] / doc_len + position_fac / (doc_len ** 3)
            doc_results["score"] = doc_results["pos"] * doc_results["score"]
            
        if enable_att == True: 
            att_scores = doc_results["att_score"]
            att_scores = torch.tensor(att_scores.tolist(), dtype=torch.float64)
    
            if method == 0:
                doc_results["att_score"] = doc_results["att_score"]
            elif method == 1:
                doc_results["att_score"] = doc_results["whole_att_score"]
            else:
                doc_results["att_score"] = (doc_results["att_score"] * doc_results["whole_att_score"]) 
            doc_results["length"] = doc_results["length"].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
            length = np.minimum(doc_results["length"], 4)
            doc_results["att_score"] = (length ** length_factor) * doc_results["att_score"]

            att_scores_float = doc_results["att_score"].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
            att_scores_clipped = att_scores_float.clip(upper=2.5)
            doc_results["att_score"] = att_scores_clipped
            doc_results["score"] = doc_results["score"] / (1 + weight * np.log1p(doc_results["att_score"]))   
      
        if enable_freq == True:                          
            freq = doc_results["candidate"].value_counts().to_dict()
            doc_results["score"] = doc_results.apply(
                lambda row: row["score"] * (0.5 - 0.02 * min(3,freq[row["candidate"]] - 1)), axis=1)
                
        ranked_keyphrases = doc_results.sort_values(by='score', ascending=False)
        top_k = ranked_keyphrases.reset_index(drop = True)
        top_k_can = top_k.loc[:, ['candidate']].values.tolist()
        
        candidates_set = set()
        candidates_dedup = []
        for temp in top_k_can:
            temp = temp[0].lower()
            if temp in candidates_set:
                continue
            else:
                candidates_set.add(temp)
                candidates_dedup.append(temp)

        j = 0
        Matched = candidates_dedup[:15]
        porter = PorterStemmer()
        for id, temp in enumerate(candidates_dedup[0:15]):
            tokens = temp.split()
            tt = ' '.join(porter.stem(t) for t in tokens)
            if (tt in labels_stemed[i] or temp in labels[i]):
                Matched[id] = [temp]
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

    # F-score 계산
    p5, r5, f5 = get_PRF(num_c_5, num_e_5, num_s)
    p10, r10, f10 = get_PRF(num_c_10, num_e_10, num_s)
    p15, r15, f15 = get_PRF(num_c_15, num_e_15, num_s)
    
    return {
        'f5': round(f5 * 100, 2),
        'f10': round(f10 * 100, 2),
        'f15': round(f15 * 100, 2),
        'p5': round(p5 * 100, 2),
        'p10': round(p10 * 100, 2),
        'p15': round(p15 * 100, 2),
        'r5': round(r5 * 100, 2),
        'r10': round(r10 * 100, 2),
        'r15': round(r15 * 100, 2)
    }
