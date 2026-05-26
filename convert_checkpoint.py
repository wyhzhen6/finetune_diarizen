import torch
import argparse
import os

def convert_ckpt(ckpt_path, base_model_path, output_path):
    print(f"Loading fine-tuned checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_state_dict = ckpt["state_dict"]

    print(f"Loading base model template from {base_model_path}...")
    base_model = torch.load(base_model_path, map_location="cpu")
    
    # Filter and rename keys
    new_state_dict = {}
    for k, v in ckpt_state_dict.items():
        if k.startswith("backbone."):
            # strip "backbone." prefix
            new_key = k[len("backbone."):]
            new_state_dict[new_key] = v
            
    # Verify match with base model state dict
    base_state_dict = base_model["state_dict"]
    missing_in_base = [k for k in new_state_dict if k not in base_state_dict]
    missing_in_new = [k for k in base_state_dict if k not in new_state_dict]
    
    if len(missing_in_base) > 0:
        print(f"Warning: Keys in converted state dict but not in base model: {missing_in_base}")
    if len(missing_in_new) > 0:
        print(f"Warning: Keys in base model but not in converted state dict: {missing_in_new}")
        
    base_model["state_dict"] = new_state_dict
    
    print(f"Saving converted model to {output_path}...")
    torch.save(base_model, output_path)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Lightning checkpoint back to PyAnnote-compatible bin")
    parser.add_argument("--ckpt", type=str, default="exp_wespeaker_finetune/checkpoints/wespeaker-epoch=97-val/loss=25.4246.ckpt", help="Path to Lightning checkpoint")
    parser.add_argument("--base", type=str, default="/home3/yihao/Research/Code/DiariZen/recipes/base_model/pyannote3/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin", help="Path to base model template")
    parser.add_argument("--output", type=str, default="exp_wespeaker_finetune/pytorch_model_finetuned.bin", help="Path to save output .bin file")
    
    args = parser.parse_args()
    convert_ckpt(args.ckpt, args.base, args.output)
