import argparse
import torch
from data_t5 import data_process
from inference_t5 import keyphrases_selection
from torch.utils.data import DataLoader

# fix the path to the T5ForConditionalGeneration model
# from test_transformers.src.test2_transformers.models.t5.modeling_t5 import T5ForConditionalGeneration


DATASET_PARAM_MAP = {
    "SemEval2017": {"std_scaling": 0.1, "att_weight": 0.1},
    "DUC2001":     {"std_scaling": 0.1, "att_weight": 0.9},
    "nus":         {"std_scaling": 1, "att_weight": 0.1},
    "wikihow":     {"std_scaling": 0.7, "att_weight": 1.0}
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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--att_layer", type=int, default=9, help="att_layer")
    parser.add_argument("--std_scaling", type=float, default=0.1, help="att_layer")
    parser.add_argument("--att_weight", type=float, default=0.9, help="att_layer")
    parser.add_argument("--log_dir", type=str, default="path/to/log", help="Directory for logging")
    return parser.parse_args()

def main():
    args = parse_arguments()
    setting_dict = get_setting_dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained("t5-"+ setting_dict["model"])
    model.to(device)
    
    # Automatically set std_scaling and att_weight for each dataset
    if args.dataset_name in DATASET_PARAM_MAP:
        std_scaling = DATASET_PARAM_MAP[args.dataset_name]["std_scaling"]
        att_weight = DATASET_PARAM_MAP[args.dataset_name]["att_weight"]
        print(f"[INFO] Using preset params for {args.dataset_name}: std_scaling={std_scaling}, att_weight={att_weight}")
    else:
        std_scaling = args.std_scaling
        att_weight = args.att_weight
        print(f"[INFO] Using CLI/std defaults: std_scaling={std_scaling}, att_weight={att_weight}")


    dataset, doc_list, labels, labels_stemed = data_process(setting_dict, args.dataset_dir, args.dataset_name, 1, args.att_layer, model, device)
    dataloader = DataLoader(dataset, num_workers=12, batch_size=args.batch_size)
    
    keyphrases_selection(setting_dict, doc_list, labels_stemed, labels, dataloader,std_scaling,att_weight, model, device)

if __name__ == "__main__":
    main()
    
