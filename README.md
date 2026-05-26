# DiariZen Fine-Tuning & EEND Module (finetune_diarizen)

This repository contains recipes and scripts for training, fine-tuning, and evaluating the **DiariZen EEND (End-to-End Neural Diarization)** module and the **WeSpeaker ResNet34** speaker embedding model.

---

## 🚀 Key Features & Improvements

### 1. WeSpeaker Embedding Fine-Tuning
* **ArcFace (AAM-Softmax) Loss**: Fine-tunes the WeSpeaker ResNet34 backbone using `ArcFaceLoss` to optimize speaker discriminative representations.
* **PyTorch Lightning Framework**: Modernized training loop with automatic checkpointing and early stopping.
* **Differential Learning Rates**: Optimizes the pretrained backbone with a 100x smaller learning rate relative to the classification head to protect existing representations from catastrophic forgetting.
* **Online Audio Augmentation**: Dataloader (`SpeakerDataset`) dynamically applies random gain perturbations and additive Gaussian noise (at random SNRs) to prevent overfitting.
* **Format Converter**: Converts PyTorch Lightning checkpoints back to standard PyAnnote/DiariZen-compatible `.bin` format using `convert_checkpoint.py`.

### 2. DiariZen EEND Optimization
* **Dual-Optimizer Training**: Trains WavLM SSL encoder with a smaller learning rate (using `optimizer_small`) and conformer/diarization layers with a standard learning rate (`optimizer_big`).
* **Auto Gradient Norm Clipping**: Dynamically clips gradient norms based on historical percentile statistics (`auto_clip_grad_norm_`), stabilizing multi-speaker powerset EEND training.
* **Model Checkpoint Averaging**: Implements checkpoint averaging over the best $N$ epochs (`infer_avg.py`) to reduce variance and improve DER.
* **Multi-Dataset Pipelines**: Out-of-the-box support for training and evaluation on **AMI**, **AliMeeting**, and **AISHELL-4** benchmarks.

---

## 🛠️ Setup & Environment

Before running any script, make sure to activate the conda environment and set the `PYTHONPATH`:

```bash
conda activate diarizen
export PYTHONPATH=$PYTHONPATH:/home3/yihao/Research/Code/DiariZen
```

Ensure your target database configurations are correctly specified in [database.yml](file:///home3/yihao/Research/Code/DiariZen/recipes/diar_ssl/database.yml).

---

## 📖 How to Use

### Part 1: Fine-Tuning WeSpeaker Embeddings (Optional)

#### Step 1: Run Fine-Tuning
Configure paths in [finetune_wespeaker.py](file:///home3/yihao/Research/Code/DiariZen/recipes/diar_ssl/finetune_wespeaker.py) and execute:
```bash
python finetune_wespeaker.py
```
This script reads speaker metadata (`wav.scp` and `rttm`) and fine-tunes the ResNet34 model. Checkpoints will be saved in `exp_wespeaker_finetune/checkpoints`.

#### Step 2: Convert Checkpoint to PyAnnote Format
Convert the best PyTorch Lightning checkpoint to a PyAnnote-compatible `.bin` file:
```bash
python convert_checkpoint.py \
  --ckpt exp_wespeaker_finetune/checkpoints/wespeaker-epoch=<EPOCH>-val/loss=<VAL_LOSS>.ckpt \
  --base /home3/yihao/Research/Code/DiariZen/recipes/base_model/pyannote3/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
  --output exp_wespeaker_finetune/pytorch_model_finetuned.bin
```
The output bin file can then be used directly in downstream diarization pipelines.

---

### Part 2: Fine-Tuning the DiariZen EEND Module

The entire training, inference, and scoring pipeline can be executed via the main recipe script [run_stage.sh](file:///home3/yihao/Research/Code/DiariZen/recipes/diar_ssl/run_stage.sh). 

Configure the configuration file path (e.g., under `conf/`) and the target stages in the script, then execute:

```bash
bash run_stage.sh
```

#### Manual Stage Execution

* **Stage 1: Model Training (Dual Optimizer)**
  ```bash
  accelerate launch --num_processes 4 --main_process_port 1134 \
    run_dual_opt.py -C conf/wavlm_updated_conformer.toml -M train
  ```
  *(For single-optimizer configuration like `fbank_conformer.toml`, use `run_single_opt.py` instead.)*

* **Stage 2: Model Inference (with checkpoint averaging)**
  ```bash
  python infer_avg.py -C <CONFIG_TOML> \
    -i <WAV_SCP_PATH> \
    -o <OUTPUT_DIR> \
    --embedding_model <EMBEDDING_MODEL_PATH> \
    --avg_ckpt_num 5 \
    --val_metric Loss \
    --val_mode best \
    --val_metric_summary <VAL_METRIC_SUMMARY_LST>
  ```

* **Stage 3: Scoring via `dscore`**
  ```bash
  python /path/to/dscore/score.py \
    -r <REF_RTTM_DIR> \
    -s <SYS_RTTM_DIR>/*.rttm \
    --collar 0 \
    > <OUTPUT_DIR>/result_collar0
  ```

---

## 📊 Results (collar=0s)

| System     | Features       | AMI  | AISHELL-4 | AliMeeting |
|:------------|:----------------:|:------:|:------------:|:------------:|
| [Pyannote v3.1](https://github.com/pyannote/pyannote-audio)  | SincNet        | 22.4 | 12.2       | 24.4       |
| DiariZen   | Fbank          | 19.7 | 12.5       | 21.0       |
|            | WavLM-frozen   | 17.0 | 11.7       | 19.9       |
|            | WavLM-updated  | **15.4** | **11.7**       | **17.6**       |

---

## ✍️ Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{han2025leveraging,
  title={Leveraging self-supervised learning for speaker diarization},
  author={Han, Jiangyu and Landini, Federico and Rohdin, Johan and Silnova, Anna and Diez, Mireia and booktitle={Proc. ICASSP},
  year={2025}
}
```
