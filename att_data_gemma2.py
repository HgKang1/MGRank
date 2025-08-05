#coding=utf-8
import re
import pandas as pd
import torch
from transformers import GemmaTokenizer, GemmaTokenizerFast
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
import torch
import torch.nn.functional as F

MAX_LEN = None


# update the CoreNLP path
StanfordCoreNLP_path = 'stanford-corenlp-full-2018-02-27'

stopword_dict = set(stopwords.words('english'))
en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)
tokenizer = None

GRAMMAR = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""


def init(setting_dict):
    global MAX_LEN, temp_en1, temp_de1, tokenizer, tokenizer_fast, enable_filter
    MAX_LEN = setting_dict["max_len"]
    temp_en1 = setting_dict["temp_en1"]
    temp_de1 = setting_dict["temp_de1"]
    enable_filter = setting_dict["enable_filter"]
    
    tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2-9b")
    tokenizer_fast = GemmaTokenizerFast.from_pretrained("google/gemma-2-9b")

    
###############################################################################
#               Calculate self-attention scores for candidates                #
###############################################################################
def extract_candidate_position_t5(doc, candidates, tokenizer_fast):
    """
    Extract token positions for candidate words in the document.
    """
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

        # Handle duplicate word occurrences using regex
        matches = list(re.finditer(re.escape(word_lower), doc_lower))
        matched = False

        for match in matches:
            start_char = match.start()
            end_char = match.end()

            # Skip if position already used
            if any(pos in used_char_positions for pos in range(start_char, end_char)):
                continue

            for pos in range(start_char, end_char):
                used_char_positions.add(pos)

            # Find token indices based on character positions
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


def extract_candidate_position(doc, candidates, tokenizer_fast):
    """
    Extract candidate word positions with sentence boundary information.
    """
    tokenized = tokenizer_fast(doc, return_offsets_mapping=True, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    offsets = tokenized["offset_mapping"][0].tolist()
    tokens = tokenizer_fast.convert_ids_to_tokens(input_ids.tolist())

    words = [item[0] for item in candidates]
    word_to_token_indices = {}
    candidate_idx = []

    used_char_positions = set()

    # Find sentence start positions
    sentences = sent_tokenize(doc)
    sentence_start_char_positions = []
    start = 0
    for sent in sentences:
        match = re.search(re.escape(sent), doc[start:], re.IGNORECASE)
        if match:
            start_idx = start + match.start()
            sentence_start_char_positions.append(start_idx)
            start = start + match.end()

    # Map sentence start positions to token indices
    sentence_start_token_idxs = []
    for char_start in sentence_start_char_positions:
        for idx, (token_start, token_end) in enumerate(offsets):
            if token_start <= char_start < token_end:
                sentence_start_token_idxs.append(idx)
                break

    # Map candidate word positions
    doc_lower = doc.lower()
    last_used_token_end_idx = -1
    for word in words:
        word_lower = word.lower().strip()
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

    return word_to_token_indices, candidate_idx, sentence_start_token_idxs


def get_attention_position(doc, candidates, tokenizer, tokenizer_fast):
    """
    Get attention positions for candidates with sentence information.
    """
    word_to_token_indices, candidates_idx, sentence_start_token_idxs = extract_candidate_position(doc, candidates, tokenizer_fast)
    att_candidates = []
    i = 0

    for j in range(len(candidates)):
        if i >= len(candidates_idx):
            i = len(candidates_idx) - 1

        new_candidate = candidates[j].copy()  

        if candidates[j][0] == candidates_idx[i][0]:
            new_candidate.append(candidates_idx[i][1])
            i += 1
        else:
            new_candidate.append(candidates_idx[i][1])

        att_candidates.append(new_candidate)
        
    return att_candidates, sentence_start_token_idxs


def get_sentence_blocks(sentence_start_token_idxs, input_len, window_size):
    """
    Create sentence blocks with sliding window approach.
    Each block contains center sentence ± window_size sentences.
    """
    num_sents = len(sentence_start_token_idxs)
    block_size = 2 * window_size + 1
    blocks = []
    input_len = input_len - 1

    # Create blocks for middle sentences
    for i in range(num_sents):
        if i < window_size + 1 or i > num_sents - window_size:
            continue  # Handle first/last blocks separately
        block_start = sentence_start_token_idxs[i - window_size]
        block_end = (
            sentence_start_token_idxs[i + window_size + 1] 
            if i + window_size + 1 < num_sents
            else input_len
        )
        blocks.append((i, block_start, block_end))  # Center sentence index, block range

    # First block
    first_center = window_size
    first_start = sentence_start_token_idxs[0] 
    first_end = (
        sentence_start_token_idxs[first_center + window_size + 1]
        if first_center + window_size + 1 < num_sents
        else input_len
    )
    blocks.insert(0, (first_center, first_start, first_end))

    # Last block
    last_center = num_sents - window_size - 1
    last_start = sentence_start_token_idxs[last_center - window_size]
    last_end = input_len
    blocks.append((last_center, last_start, last_end))

    return blocks


def get_self_att_score(
    doc, candidates, window_size, model, tokenizer, tokenizer_fast, device, att_layer,
): 
    """
    Compute self-attention scores for candidate words using block-wise processing.
    
    Args:
        doc: Input document
        candidates: List of candidate words
        window_size: Window size for sentence blocks
        att_layer: Attention layer index to use
    
    Returns:
        List of attention scores for each candidate
    """
    model.eval()
    input_ids = tokenizer.encode(doc, return_tensors="pt").to(device)
    att_candidates, sentence_start_token_idxs = get_attention_position(
        doc, candidates, tokenizer, tokenizer_fast
    )
    
    num_sents = len(sentence_start_token_idxs)
    whole_scores = []
    blockwise_scores = []
    cand_embedding = []
    cand_only_embedding = []
    
    # Handle short documents (fewer sentences than window size)
    if num_sents < (2 * window_size + 1):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)
            attentions = outputs.attentions
        sequence_length = input_ids.shape[1]
        last_attn = attentions[att_layer].squeeze(0)
        avg_attn = last_attn.mean(dim=0) 
        block_candidates = []
        
        for word, _, (start_idx, end_idx) in att_candidates:
            att_values = avg_attn[:, start_idx:end_idx + 1]
            att_score = att_values.sum(dim=0).mean().item()
            length = end_idx - start_idx + 1
            block_candidates.append((word, att_score, length, (start_idx, end_idx)))
            
        # Normalize scores
        score_sum = sum(score for _, score, *_ in block_candidates) + 1e-10
        for word, att_score, length, span in block_candidates:
            norm_score = round((att_score / score_sum), 4)
            blockwise_scores.append([word, norm_score, length, span])
        return blockwise_scores
    
    else:
        # Process full document for global scores
        with torch.no_grad():
            input_ids2 = input_ids[:, :-1]
            outputs = model(input_ids=input_ids2, output_attentions=True)
            attentions = outputs.attentions
        sequence_length = input_ids.shape[1]
        last_attn = attentions[att_layer].squeeze(0)
        avg_attn = last_attn.mean(dim=0) 

        # Compute whole document attention scores
        for word, _, (start_idx, end_idx) in att_candidates:
            att_values = avg_attn[:, start_idx:end_idx + 1]
            att_score = att_values.sum(dim=0).mean().item()
            length = end_idx - start_idx + 1
            whole_scores.append([word, att_score, length])

        # Process sentence blocks
        sentence_blocks = get_sentence_blocks(sentence_start_token_idxs, input_ids.shape[1], window_size)
        
        for block_idx, (center_idx, block_start, block_end) in enumerate(sentence_blocks):
            local_input_ids = input_ids[:, block_start:block_end]
            block_len = block_end - block_start
            
            with torch.no_grad():
                outputs = model(input_ids=local_input_ids, output_attentions=True)
                attentions = outputs.attentions

            last_attn = attentions[att_layer].squeeze(0) 
            avg_attn = last_attn.mean(dim=0)     # Average over attention heads

            # Define center sentence token range
            center_start = sentence_start_token_idxs[center_idx]
            center_end = (
                sentence_start_token_idxs[center_idx + 1]
                if center_idx + 1 < len(sentence_start_token_idxs)
                else input_ids.shape[1]
            )

            # Check if this is first or last block (special cases)
            is_first_block = (block_idx == 0)
            is_last_block = (block_idx == len(sentence_blocks) - 1)
            block_candidates = []
            
            for word, _, (start_idx, end_idx) in att_candidates:
                # Word must be within the block
                if not (block_start <= start_idx < block_end):
                    continue

                # Apply selection criteria based on block type
                if is_first_block:
                    if start_idx >= center_end:
                        continue
                elif is_last_block:
                    if end_idx < center_start:
                        continue
                else:
                    if not (center_start <= start_idx < center_end):
                        continue

                # Convert to local indices and compute attention
                local_start = start_idx - block_start
                local_end = end_idx - block_start
                att_values = avg_attn[:, local_start:local_end + 1]
                length = local_end - local_start + 1
                att_score = att_values.sum(dim=0).mean().item() 
                block_candidates.append((word, att_score, length, (start_idx, end_idx), (block_start, block_end)))
                
            # Normalize scores within block
            cand = len(block_candidates)
            score_sum = sum(score for _, score, *_ in block_candidates) + 1e-10
            for word, att_score, length, span, blk in block_candidates:
                norm_score = round((att_score / score_sum) * cand, 4)
                blockwise_scores.append([word, norm_score, length, span, blk])
        
        # Add whole document scores to blockwise results
        for b, w in zip(blockwise_scores, whole_scores):
            att_score = w[1]  # whole_scores attention score
            b.append(att_score)
            
    return blockwise_scores


###############################################################################
#                        Find top-k most similar words                        #
###############################################################################
def get_topk_similar_words_exclusive(cand_embedding, top_k=3):
    """
    Find top-k most similar words using cosine similarity, excluding words with overlapping stems.
    
    Args:
        cand_embedding: List of tuples (word, embedding, span)
        top_k: Number of similar words to return
    
    Returns:
        List of tuples (word, list_of_similar_word_positions)
    """
    porter = nltk.PorterStemmer()
    words = [word for word, _, _ in cand_embedding]
    spans = [span for _, _, span in cand_embedding]
    embeddings = torch.stack([emb for _, emb, _ in cand_embedding])  
    embeddings = F.normalize(embeddings, p=2, dim=1)  

    # Compute full cosine similarity matrix (N x N)
    cosine_sim_matrix = torch.matmul(embeddings, embeddings.T)

    # Set self-similarity to -1 to exclude
    cosine_sim_matrix.fill_diagonal_(-1.0)
    

    stem_sets = [set(porter.stem(w.lower()) for w in word.split()) for word in words]

    results = []
    for i, (word, word_set) in enumerate(zip(words, stem_sets)):
        cosine_sim = cosine_sim_matrix[i]
        sorted_indices = torch.argsort(cosine_sim, descending=True)

        similar_words_idx = []
        count = 0
        for idx in sorted_indices:
            candidate_word_set = stem_sets[idx]
            candidate_span = spans[idx]
            if candidate_span[0] <= 1 or candidate_span[0] >= 780:
                continue
            # Only include words with non-overlapping stems
            if word_set.isdisjoint(candidate_word_set):
                similar_words_idx.append(int((candidate_span[0] + candidate_span[1]) / 2))
                count += 1
                if count == top_k:
                    break

        results.append((word, similar_words_idx))
    return results


def get_sim_cand_idx(
    doc, candidates, tokenizer_t5, model_t5, device
): 
    """
    Returns top-k most similar words for each candidate.
    """
    model_t5.eval()

    word_to_token_indices, candidates_idx = extract_candidate_position_t5(doc, candidates, tokenizer_t5)
    att_candidates = []
    i = 0
    
    for j in range(len(candidates)):
        if i >= len(candidates_idx):
            i = len(candidates_idx) - 1

        new_candidate = candidates[j].copy()  

        if candidates[j][0] == candidates_idx[i][0]:
            new_candidate.append(candidates_idx[i][1])
            i += 1
        else:
            new_candidate.append(candidates_idx[i][1])

        att_candidates.append(new_candidate)

    # Tokenize and get embeddings
    input_ids = tokenizer_t5(doc, return_tensors="pt").to(device)
    cand_embedding = []
    cand_only_embedding = []
    input_ids = input_ids["input_ids"].to(device)
    
    # Extract word embeddings
    with torch.no_grad():
        outputs = model_t5(input_ids=input_ids)
        last_hidden_state = outputs.last_hidden_state       
        
    for word, _, (gemma_start, gemma_end), (start_idx, end_idx) in att_candidates:
        # Average embeddings across word tokens
        word_embedding = last_hidden_state[0, start_idx:end_idx + 1, :].mean(dim=0)
        cand_embedding.append([word, word_embedding, (gemma_start, gemma_end)])
        
    # Find similar word groups
    cand_sim_group = get_topk_similar_words_exclusive(cand_embedding, top_k=3)
    
    return cand_sim_group

