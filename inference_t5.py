import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer
from nltk import PorterStemmer
from sys import exit
import torch.nn.functional as F
pd.options.mode.chained_assignment = None

MAX_LEN = None
enable_pos = None
enable_freq = None
enable_att = None
temp_en1 = None
temp_en3 = None
temp_de1 = None
temp_de2 = None
temp_de3 = None
length_factor = None
position_factor = None
tokenizer = None

def init(setting_dict):
    '''
    Init template, max length and tokenizer.
    '''

    global MAX_LEN, temp_en1, temp_de1, tokenizer
    global enable_pos, enable_att, length_factor

    MAX_LEN = setting_dict["max_len"]
    temp_en1 = "Book:"
    temp_de1 = "This book mainly talks about "
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


def keyphrases_selection(setting_dict, doc_list, labels_stemed, labels, dataloader,std_scaling_factor, att_weight, model, device):

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
    pred_labels={}
    num_c_5 = num_c_10 = num_c_15 = 0
    num_e_5 = num_e_10 = num_e_15 = 0
    num_s = 0
    underscore_token_id = 3
    template_len = tokenizer(temp_de1, return_tensors="pt")["input_ids"].shape[1] - 3 # single space

    
    
    for id, [en_input_ids,  en_input_mask, de_input_ids, dic] in enumerate(tqdm(dataloader,desc="Evaluating:")):

        en_input_ids = en_input_ids.to(device)
        en_input_mask = en_input_mask.to(device)
        de_input_ids = de_input_ids.to(device)
        real_lens = en_input_mask.sum(dim=1)

        underscore_mask = (en_input_ids == underscore_token_id).unsqueeze(1).unsqueeze(2)
        kwargs = {
            "t5_cand_pos_s": dic["t5_pos_s"],
            "t5_cand_pos_e": dic["t5_pos_e"],
            "initial_std": 0.3,
            "std_scaling_factor": std_scaling_factor,
            "underscore_mask": underscore_mask,
            "topk": 15,
            "sim_word_idx": dic["sim_word_idx"],
        }
        score = np.zeros(en_input_ids.shape[0])

        with torch.no_grad():
            output = model(
                input_ids=en_input_ids,
                attention_mask=en_input_mask,
                decoder_input_ids=de_input_ids,
                output_hidden_states=True,
                **kwargs
            )[0]  # (B, L, V): Decoder output logits for each position

            log_probs = F.log_softmax(output, dim=-1)  # (B, L, V): Log-probabilities over vocabulary

            B = de_input_ids.shape[0] 
            max_i = de_input_ids.shape[1] - 3  # End index for candidate span 
            indices = torch.arange(template_len, max_i).to(de_input_ids.device)  # Range of candidate positions

            next_token_indices = de_input_ids[:, template_len+1 : max_i+1]  # Ground-truth next tokens to predict (B, T)

            selected_log_probs = log_probs[:, template_len:max_i, :]  # Log-probabilities for candidate positions (B, T, V)
            gathered = selected_log_probs.gather(2, next_token_indices.unsqueeze(-1)).squeeze(-1)  # Select the predicted log-prob for each true token (B, T)

            mask = torch.arange(template_len, max_i).unsqueeze(0).to(de_input_ids.device)  # Position mask (1, T)
            de_input_lens = torch.tensor(dic["de_input_len"]).unsqueeze(1).to(de_input_ids.device)  # True target length per example (B, 1)
            valid_mask = (mask < de_input_lens).float()  # Mask positions that are beyond the actual candidate length (B, T)

            masked_scores = gathered * valid_mask  # Apply mask to ignore padding tokens (B, T)
            sum_log_probs = masked_scores.sum(dim=1)  # Sum log-probs across valid candidate tokens (B,)

            denom = (de_input_lens.squeeze(1) - template_len).float().pow(length_factor)  # Length normalization
            final_score = sum_log_probs / denom  # Length-normalized log-prob score for each example
            score = final_score.cpu().numpy()  

               
            doc_id_list.extend(dic["idx"])
            candidate_list.extend(dic["candidate"])
            cos_score_list.extend(score)
            pos_list.extend(dic["pos"])
            att_score_list.extend(dic["att_score"])
            whole_att_score_list.extend(dic["whole_att_score"])
            length_list.extend(dic["length"])
    
    cos_similarity_list["doc_id"] = doc_id_list
    cos_similarity_list["candidate"] = candidate_list
    cos_similarity_list["score"] = cos_score_list
    cos_similarity_list["pos"] = pos_list
    cos_similarity_list["att_score"] = att_score_list
    cos_similarity_list["length"] = length_list
    cos_similarity_list["whole_att_score"] = whole_att_score_list
    
    cosine_similarity_rank = pd.DataFrame(cos_similarity_list)

    for i in range(len(doc_list)):
        temp_label=[]
        tt_label=[]
        doc_len = len(doc_list[i].split())
        
        doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id']==i]
        if enable_pos == True:
            position_factor =  1.2e8
            doc_results["pos"] = doc_results["pos"] / doc_len + position_factor / (doc_len ** 3)
            doc_results["score"] = doc_results["pos"] * doc_results["score"]
        #* doc_results["score"].values.astype(float)
        if enable_att == True: 
            att_scores = doc_results["att_score"]
            att_scores = torch.tensor(att_scores.tolist(), dtype=torch.float64)
            doc_results["att_score"] = (doc_results["att_score"] * doc_results["whole_att_score"]) 
            att_scores_float = doc_results["att_score"].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
            att_scores_clipped = att_scores_float.clip(upper=2.5)
            doc_results["att_score"] = att_scores_clipped
            
            doc_results["score"] = doc_results["score"] / (1 + att_weight * np.log1p(doc_results["att_score"]))   

                
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

        pred_labels[i] = {
            "temp": temp_label,
            "tt": tt_label
        }
    p, r, f = get_PRF(num_c_5, num_e_5, num_s)
    print_PRF(p, r, f, 5)

    p, r, f = get_PRF(num_c_10, num_e_10, num_s)
    print_PRF(p, r, f, 10)

    p, r, f = get_PRF(num_c_15, num_e_15, num_s)
    print_PRF(p, r, f, 15)
    return cosine_similarity_rank

def calculate_score2(setting_dict, cosine_similarity_rank,doc_list,labels,labels_stemed,enable_pos,enable_att,enable_freq,length_factor,weight,position_fac,method):
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
            doc_results["pos"] = doc_results["pos"] / doc_len + position_fac / (doc_len ** 3)
            doc_results["score"] = doc_results["pos"] * doc_results["score"]
        #* doc_results["score"].values.astype(float)
        if enable_att == True: 
            # block_att_col = f"block_att_score_{att_head_idx}"
            # whole_att_col = f"whole_att_score_{att_head_idx}"
            # doc_results["att_score"] = doc_results[block_att_col]
            # doc_results["whole_att_score"] = doc_results[whole_att_col]
            
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
            # doc_results["score"] = (1-weight)*doc_results["score"] + (weight) * doc_results["att_score"]
      
        if enable_freq == True:                          # 후보단어 빈도수 스코어링 추가
            # calculate candidate frequency
            freq = doc_results["candidate"].value_counts().to_dict()
            doc_results["score"] = doc_results.apply(
                lambda row: row["score"] * (0.5 - 0.02 * min(3,freq[row["candidate"]] - 1)), axis=1)
                
        # ranked_keyphrases = doc_results.sort_values(by='score', ascending=False)
        ranked_keyphrases = doc_results.sort_values(by="score", ascending=False) #drop_duplicates(subset=["candidate"], keep="first")  
        doc_results2 = doc_results.sort_values(by="score", ascending=False)   #drop_duplicates(subset=["candidate"], keep="first")
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

    p, r, f = get_PRF(num_c_5, num_e_5, num_s)
    print_PRF(p, r, f, 5)

    p, r, f = get_PRF(num_c_10, num_e_10, num_s)
    print_PRF(p, r, f, 10)

    p, r, f = get_PRF(num_c_15, num_e_15, num_s)
    print_PRF(p, r, f, 15)
    new_cand_scores = pd.concat(new_cand_score, ignore_index=True)
    return new_cand_scores