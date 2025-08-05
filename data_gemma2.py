#coding=utf-8
import re
import codecs
import json
import os
import pandas as pd
import ast
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
from transformers import T5TokenizerFast, T5EncoderModel
from transformers import GemmaTokenizer, GemmaTokenizerFast
from att_data_gemma2 import get_self_att_score, get_sim_cand_idx
MAX_LEN = None

# update the CoreNLP path
StanfordCoreNLP_path = 'stanford-corenlp-full-2018-02-27'
stopword_dict = set(stopwords.words('english'))
en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)
tokenizer = None

GRAMMAR = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """

    cans_count = dict()
    
    np_parser = nltk.RegexpParser(GRAMMAR)  # Noun phrase parser
    keyphrase_candidate = []
    
    
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            
            if len(np.split()) == 1:
                if np not in cans_count.keys():
                    cans_count[np] = 0
                cans_count[np] += 1
                
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1
        
   
    
    return keyphrase_candidate


class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, en_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}

        self.tokens = []
        self.tokens_tagged = []
        self.tokens = en_model.word_tokenize(text)
        self.tokens_tagged = en_model.pos_tag(text)
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")
        self.keyphrase_candidate = extract_candidates(self.tokens_tagged, en_model)
        
class DocBatchDataset(Dataset):
    def __init__(self, all_docs_pairs):
        self.all_docs_pairs = all_docs_pairs  # List of List of doc_pairs

    def __len__(self):
        return len(self.all_docs_pairs)  # 문서 수

    def __getitem__(self, idx):
        return self.all_docs_pairs[idx]         
     
class KPE_Dataset(Dataset):
    def __init__(self, docs_pairs):

        self.docs_pairs = docs_pairs
        self.total_examples = len(self.docs_pairs)

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):

        doc_pair = self.docs_pairs[idx]
        en_input_ids = doc_pair[0][0]
        en_input_mask = doc_pair[1][0]
        dic = doc_pair[2]

        return [en_input_ids, en_input_mask, dic]
    
def clean_text(text="",database="Inspec"):

    #Specially for Duc2001 Database
    if(database=="Duc2001" or database=="Semeval2017"):
        pattern2 = re.compile(r'[\s,]' + '[\n]{1}')
        while (True):
            if (pattern2.search(text) is not None):
                position = pattern2.search(text)
                start = position.start()
                end = position.end()
                # start = int(position[0])
                text_new = text[:start] + "\n" + text[start + 2:]
                text = text_new
            else:
                break

    pattern2 = re.compile(r'[a-zA-Z0-9,\s]' + '[\n]{1}')
    while (True):
        if (pattern2.search(text) is not None):
            position = pattern2.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + " " + text[start + 2:]
            text = text_new
        else:
            break

    pattern3 = re.compile(r'\s{2,}')
    while (True):
        if (pattern3.search(text) is not None):
            position = pattern3.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + "" + text[start + 2:]
            text = text_new
        else:
            break

    pattern1 = re.compile(r'[<>[\]{}]')
    text = pattern1.sub(' ', text)
    text = text.replace("\t", " ")
    text = text.replace(' p ','\n')
    text = text.replace(' /p \n','\n')
    lines = text.splitlines()
    # delete blank line
    text_new=""
    for line in lines:
        if(line!='\n'):
            text_new+=line+'\n'

    return text_new



def extract_candidate_position(doc, candidates):
    tokenized = tokenizer_fast(doc, return_offsets_mapping=True, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    offsets = tokenized["offset_mapping"][0].tolist()
    tokens = tokenizer_fast.convert_ids_to_tokens(input_ids.tolist())

    words = [item[0] for item in candidates]
    word_to_token_indices = {}
    candidate_idx = []

    used_char_positions = set()
    last_used_token_end_idx = -1
    for word in words:
        word_lower = word.lower().strip()
        doc_lower = doc.lower()

        matches = list(re.finditer(re.escape(word_lower), doc_lower))
        matched = False

        for match in matches:
            start_char = match.start()
            end_char = match.end()

            if any(pos in used_char_positions for pos in range(start_char, end_char)):
                continue

            for pos in range(start_char, end_char):
                used_char_positions.add(pos)

            token_start_idx = None
            token_end_idx = None

            for idx, (token_start, token_end) in enumerate(offsets):
                if token_start <= start_char < token_end:
                    token_start_idx = idx
                if token_start < end_char <= token_end:
                    token_end_idx = idx
                    break
            if token_start_idx is None or token_end_idx is None:
                for idx, (token_start, token_end) in enumerate(offsets):
                    if token_start <= start_char and token_start_idx is None:
                        token_start_idx = idx
                    if token_end >= end_char:
                        token_end_idx = idx
                        break

            if token_start_idx is not None and token_end_idx is not None:
                if token_start_idx <= last_used_token_end_idx:
                    continue 

                if word not in word_to_token_indices:
                    word_to_token_indices[word] = []
                word_to_token_indices[word].append([token_start_idx, token_end_idx])
                candidate_idx.append([word, (token_start_idx, token_end_idx)])
                last_used_token_end_idx = token_end_idx 
                matched = True
                break

        if not matched:
            print(f"[WARNING] '{word}' not matched in doc.")

    return word_to_token_indices, candidate_idx

def get_long_data(file_path="data/nus/nus_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                fulltxt = jsonl['fulltext']
                doc = ' '.join([abstract, fulltxt])
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="nus")
                doc = doc.replace('\n', ' ')
                data[jsonl['name']] = doc
                labels[jsonl['name']] = keywords
            except:
                raise ValueError
    return data,labels


def get_duc2001_data(file_path="data/DUC2001"):
    pattern = re.compile(r'<TEXT>(.*?)</TEXT>', re.S)
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(file_path):
      
        for fname in sorted(filenames):
            if (fname == "annotations.txt"):
                # left, right = fname.split('.')
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                lines = text.splitlines()
                for line in lines:
                    left, right = line.split("@")
                    d = right.split(";")[:-1]
                    l = left
                    labels[l] = d
                f.close()
            else:
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                text = re.findall(pattern, text)[0]

                text = text.lower()
                text = clean_text(text,database="Duc2001")
                data[fname]=text.strip("\n")
                # data[fname] = text
    return data,labels

def get_semeval2017_data(data_path="data/SemEval2017/docsutf8",labels_path="data/SemEval2017/keys"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(data_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            # f = open(infile, 'rb')
            # text = f.read().decode('utf8')
            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", '')
            text = clean_text(text,database="Semeval2017")
            data[left] = text.lower()
            # f.close()
    for dirname, dirnames, filenames in os.walk(labels_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            f = open(infile, 'rb')
            text = f.read().decode('utf8')
            text = text.strip()
            ls=text.splitlines()
            labels[left] = ls
            f.close()
    return data,labels

def get_wikihow_data(data_path="data/wikihow",labels_path="data/wikihow"):

    data={}
    labels={}
    wiki= pd.read_csv(data_path)
    for idx in range(len(wiki)) :
        text = wiki['text'].iloc[idx]
        text = clean_text(text)
        data[idx] = text.lower()
   
        lab = wiki['label'].iloc[idx]
        lab = ast.literal_eval(lab)
        labels[idx] = lab
    return data,labels

def remove (text):
    text_len = len(text.split())
    remove_chars = '[’!"#$%&\'()*+,./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, '', text)
    re_text_len = len(text.split())
    if text_len != re_text_len:
        return True
    else:
        return False


    
def generate_doc_pairs(doc, candidates, idx):
    count = 0
    doc_pairs = []
    doc_prompt = f"Text: {doc}"
    # prompt = task_prompt + " " + doc_prompt
    prompt = doc_prompt
    for id, can_and_pos in enumerate(candidates):
        candidate = can_and_pos[0]
        target_temp = "Answer: This text mainly talks about "
        target = f"Answer: This text mainly talks about {candidate}."
        full_input = prompt + " " + target 
        input_ids = tokenizer(full_input, max_length=782, padding="max_length", truncation=True, return_tensors="pt")['input_ids']
        input_mask = tokenizer(full_input, max_length=782, padding="max_length", truncation=True, return_tensors="pt")['attention_mask']
        
        target_temp_ids = tokenizer(target_temp, return_tensors="pt")['input_ids'][0]
        target_temp_len = target_temp_ids.size(0)
        target_ids = tokenizer(target, return_tensors="pt")['input_ids'][0]
        target_len = target_ids.size(0)  
        non_pad_len = (input_ids != tokenizer.pad_token_id).sum().item() 
        context_len = non_pad_len - target_len            
        start_idx = context_len + target_temp_len -1          
        candidate_len= target_len - target_temp_len

        dic = {"start_idx":start_idx, "candidate":candidate, "idx":idx, "candidate_len":candidate_len, "temp_start_idx": context_len + 1,
               "pos":can_and_pos[1][0],"t5_pos_s":can_and_pos[2][0], "t5_pos_e":can_and_pos[2][1],"att_score": can_and_pos[3],"sim_word_idx": can_and_pos[5],"whole_att_score": can_and_pos[6],
                } 
        
        doc_pairs.append([input_ids, input_mask, dic])

    return doc_pairs, count

  

def init(setting_dict):
    '''
    Init template, max length and tokenizer.
    '''

    global MAX_LEN, tokenizer, tokenizer_fast
    MAX_LEN = setting_dict["max_len"]

    tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2-9b")
    tokenizer_fast = GemmaTokenizerFast.from_pretrained("google/gemma-2-9b")
    

def data_process(setting_dict, dataset_dir, dataset_name,window, att_layer, model, device):
    '''
    Core API in data.py which returns the dataset
    '''
    global text_obj
    init(setting_dict)
    

    if dataset_name =="SemEval2017":
        data, referneces = get_semeval2017_data(dataset_dir + "/docsutf8", dataset_dir + "/keys")
    elif dataset_name == "DUC2001":
        data, referneces = get_duc2001_data(dataset_dir)
    elif dataset_name == "nus" :
        data, referneces = get_long_data(dataset_dir + "/nus_test.json")
    elif dataset_name == 'wikihow':
        data, referneces = get_wikihow_data(dataset_dir + "/wikihow.csv")
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}. Please check --dataset_name argument.")
    
    docs_pairs = []
    doc_list = []
    labels = []
    labels_stemed = []
    docs_candidates = []
    docs_att_candidates = []
    candidates_embedding = []
    t_n = 0
    candidate_num = 0
    porter = nltk.PorterStemmer()
    tokenizer_t5 = T5TokenizerFast.from_pretrained("t5-base")
    model_t5 = T5EncoderModel.from_pretrained("t5-base").to(device)

    for idx, (key, doc) in enumerate(data.items()):
        # if idx >= 10:  # Stop after processing 5 documents
        #     break
        labels.append([ref.replace(" \n", "") for ref in referneces[key]])
        labels_s = []
        for l in referneces[key]:
            tokens = l.split()
            labels_s.append(' '.join(porter.stem(t) for t in tokens))

        doc = ' '.join(doc.split()[:MAX_LEN])  
        labels_stemed.append(labels_s)
        doc_list.append(doc)
        
        # Statistic on empty docs
        empty_doc = 0
        try:
            text_obj = InputTextObj(en_model, doc)
        except:
            empty_doc += 1
            print("doc: ", doc)

        # Generate candidates (lower)
        cans = text_obj.keyphrase_candidate
        candidates = []
        for can, pos in cans:
            candidates.append([can.lower(), pos])
        candidate_num += len(candidates)
        candidates=[candidate for candidate in candidates if not remove(candidate[0])]
        att_candidates= get_self_att_score(doc, candidates, window, model, tokenizer, tokenizer_fast, device, att_layer)
        docs_att_candidates.append(att_candidates)
        # extract candidate position 
        doc_prompt = f"Text: {doc}"
        word_to_token_indices, candidates_idx = extract_candidate_position(doc_prompt,candidates)
        if len(candidates) != len(candidates_idx):
            print('document idx that does not match the number of candidate and t5 tokenization candidate')
            print(idx)

        i=0
        for j in range(len(candidates)):
            if i >= len(candidates_idx):
                i = len(candidates_idx) - 1
                
            if candidates[j][0] == candidates_idx[i][0]:
                candidates[j].append(candidates_idx[i][1])
                i+=1
            else:
                candidates[j].append(candidates_idx[i][1])
        
        k=0
       
        cand_sim_group = get_sim_cand_idx(doc_prompt, candidates, tokenizer_t5, model_t5, device)
   
        for i in range(len(candidates)):
            idx_new = i if i < len(att_candidates) else len(att_candidates) - 1

            candidates[i].append(att_candidates[idx_new][1])
            candidates[i].append(att_candidates[idx_new][2])      
            candidates[i].append(cand_sim_group[i][1])  
            if len(att_candidates[idx_new]) > 5:
                candidates[i].append(att_candidates[idx_new][5])
            else:
                candidates[i].append(att_candidates[idx_new][1]) 
   
        doc_pairs, count = generate_doc_pairs(doc, candidates, idx)
        docs_pairs.extend(doc_pairs)
        t_n += count
        docs_candidates.append(candidates)

    print("candidate_num: ", candidate_num)
    dataset = KPE_Dataset(docs_pairs)
    print("examples: ", dataset.total_examples)

    en_model.close()
    return dataset, doc_list, labels, labels_stemed


