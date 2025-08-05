import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import logging
import time
import torch

from data import data_process
from inference import keyphrases_selection
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from transformers import GemmaTokenizer
from transformers import Gemma2ForCausalLM
from test_transformers.src.test2_transformers.models.gemma2.modeling_gemma2_normal import Gemma2ForCausalLM_our


DATASET_PARAM_MAP = {
    "SemEval2017": {"std_scaling": 0.1, "att_weight": 0.1},
    "DUC2001":     {"std_scaling": 0.4, "att_weight": 0.7},
    "nus":         {"std_scaling": 0.8, "att_weight": 0.7},
    "wikihow":     {"std_scaling": 0.1, "att_weight": 1.0}
}

def get_setting_dict():
    setting_dict = {}
    setting_dict["max_len"] = 512
    setting_dict["model"] = "base"
    setting_dict["enable_pos"] = False
    setting_dict["enable_att"] = True
    setting_dict["length_factor"] = 0.6
    return setting_dict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run T5-based keyphrase extraction")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., DUC2001)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--att_layer", type=int, default=23, help="att_layer")
    parser.add_argument("--std_scaling", type=float, default=0.1, help="att_layer")
    parser.add_argument("--att_weight", type=float, default=0.7, help="att_layer")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--log_dir", type=str, default="path/to/log", help="Directory for logging")
    return parser.parse_args()

def main():
    args = parse_arguments()
    setting_dict = get_setting_dict()
 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2-9b")
    quant_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )    

    model = Gemma2ForCausalLM.from_pretrained(
        "google/gemma-2-9b",
        quantization_config=quant_config,
#         device_map="auto",
        device_map={"":args.gpu},
        torch_dtype=torch.bfloat16,
        output_attentions=True 
    )
    if args.dataset_name in DATASET_PARAM_MAP:
        std_scaling = DATASET_PARAM_MAP[args.dataset_name]["std_scaling"]
        att_weight = DATASET_PARAM_MAP[args.dataset_name]["att_weight"]
        print(f"[INFO] Using preset params for {args.dataset_name}: std_scaling={std_scaling}, att_weight={att_weight}")
    else:
        std_scaling = args.std_scaling
        att_weight = args.att_weight
        print(f"[INFO] Using CLI/std defaults: std_scaling={std_scaling}, att_weight={att_weight}")
        
    dataset, doc_list, labels, labels_stemed = data_process(setting_dict, args.dataset_dir, args.dataset_name, 1, args.att_layer, model, device)
    dataloader = DataLoader(dataset, num_workers=9, batch_size=args.batch_size)
    
    del model
    torch.cuda.empty_cache()
    
    model2 = Gemma2ForCausalLM_our.from_pretrained(
        "google/gemma-2-9b",
        quantization_config=quant_config,
#         device_map="auto",
        device_map={"":args.gpu},
        torch_dtype=torch.bfloat16,
        output_attentions=True 
    )
    
    keyphrases_selection(setting_dict, doc_list, labels_stemed, labels, dataloader, 0.4,0.7, model2, device)
    # cosine_similarity_rank, pred_labels = keyphrases_selection(setting_dict, doc_list, labels_stemed, labels, model2, dataloader,0.3,0.8,15,1, device)

if __name__ == "__main__":
    main()

