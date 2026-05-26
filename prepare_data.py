import os
import soundfile as sf
import random
from pathlib import Path

# Paths
audio_dir = "/home3/yihao/Research/basebend/sim_final2/train/se"
rttm_dir = "/home3/yihao/Research/basebend/sim_final2/train/rttm"
output_dir = "/home3/yihao/Research/Code/DiariZen/recipes/diar_ssl/data/custom_finetune"

# Create output directories
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

def prepare_split(files, split_name):
    print(f"Preparing {split_name} split with {len(files)} files...")
    
    scp_path = os.path.join(output_dir, split_name, "wav.scp")
    uem_path = os.path.join(output_dir, split_name, "all.uem")
    rttm_path = os.path.join(output_dir, split_name, "rttm")
    
    with open(scp_path, "w") as scp_f, \
         open(uem_path, "w") as uem_f, \
         open(rttm_path, "w") as rttm_f:
        
        for audio_file in files:
            rec_id = audio_file.stem
            audio_path = str(audio_file.absolute())
            rttm_file = Path(rttm_dir) / f"{rec_id}.rttm"
            
            if not rttm_file.exists():
                print(f"Warning: RTTM not found for {rec_id}, skipping.")
                continue
            
            # Get duration
            try:
                info = sf.info(audio_path)
                duration = info.duration
            except Exception as e:
                print(f"Error reading {audio_path}: {e}")
                continue
            
            # Write wav.scp
            scp_f.write(f"{rec_id} {audio_path}\n")
            
            # Write all.uem (format: rec_id 1 0.000 duration)
            uem_f.write(f"{rec_id} 1 0.000 {duration:.3f}\n")
            
            # Write combined rttm
            with open(rttm_file, "r") as rf:
                content = rf.read().strip()
                if content:
                    rttm_f.write(content + "\n")

def main():
    # Source paths
    train_audio_dir = "/home3/yihao/Research/basebend/sim_final2/train/se"
    train_rttm_dir = "/home3/yihao/Research/basebend/sim_final2/train/rttm"
    val_audio_dir = "/home3/yihao/Research/basebend/sim_final2/val/se"
    val_rttm_dir = "/home3/yihao/Research/basebend/sim_final2/val/rttm"

    # Prepare Train
    train_files = sorted(list(Path(train_audio_dir).glob("*.wav")))
    print(f"Found {len(train_files)} training wav files.")
    
    # Prepare Val
    val_files = sorted(list(Path(val_audio_dir).glob("*.wav")))
    print(f"Found {len(val_files)} validation wav files.")

    if not train_files:
        print("No training files found! Check the path.")
        return

    # Process splits (using specific RTTM dirs)
    def process_split(files, rttm_src, split_name):
        # Temporarily override global rttm_dir for this call
        global rttm_dir
        old_rttm_dir = rttm_dir
        rttm_dir = rttm_src
        prepare_split(files, split_name)
        rttm_dir = old_rttm_dir

    process_split(train_files, train_rttm_dir, "train")
    process_split(val_files, val_rttm_dir, "val")
    
    print("\nData preparation complete!")
    print(f"Metadata saved to: {output_dir}")

if __name__ == "__main__":
    main()
