# Licensed under the MIT license.
# Adopted from https://github.com/espnet/espnet/blob/master/egs2/chime8_task1/diar_asr1/local/pyannote_diarize.py
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import os
import argparse
import toml
from pathlib import Path
from typing import Dict

import torch
import numpy as np
import torchaudio

from scipy.ndimage import median_filter

from pyannote.metrics.segmentation import Annotation, Segment
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.utils.signal import Binarize

from diarizen.ckpt_utils import load_metric_summary


def load_scp(scp_file: str) -> Dict[str, str]:
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}

def diarize_session(
    sess_name,
    in_wav,
    pipeline,
    min_speakers=1,
    max_speakers=20,
    apply_median_filtering=True
):
    print('Extracting segmentations...')
    waveform, sample_rate = torchaudio.load(in_wav) 
    waveform = torch.unsqueeze(waveform[0], 0)      # force to use the SDM data
    segmentations = pipeline.get_segmentations({"waveform": waveform, "sample_rate": sample_rate}, soft=False)

    if apply_median_filtering:
        segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

    # binarize segmentation
    binarized_segmentations = segmentations     # powerset

    # estimate frame-level number of instantaneous speakers
    count = pipeline.speaker_count(
        binarized_segmentations,
        pipeline._segmentation.model._receptive_field,
        warm_up=(0.0, 0.0),
    )

    print("Extracting Embeddings.")
    embeddings = pipeline.get_embeddings(
        {"waveform": waveform, "sample_rate": sample_rate},
        binarized_segmentations,
        exclude_overlap=pipeline.embedding_exclude_overlap,
    )

    #  shape: (num_chunks, local_num_speakers, dimension)
    print("Clustering.")
    hard_clusters, _, _ = pipeline.clustering(
        embeddings=embeddings,
        segmentations=binarized_segmentations,
        min_clusters=min_speakers, 
        max_clusters=max_speakers,  
    )

    # during counting, we could possibly overcount the number of instantaneous
    # speakers due to segmentation errors, so we cap the maximum instantaneous number
    # of speakers by the `max_speakers` value
    count.data = np.minimum(count.data, max_speakers).astype(np.int8)

    # keep track of inactive speakers
    inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
    #   shape: (num_chunks, num_speakers)

    # reconstruct discrete diarization from raw hard clusters
    hard_clusters[inactive_speakers] = -2
    discrete_diarization, _ = pipeline.reconstruct(
        segmentations,
        hard_clusters,
        count,
    )

    # convert to annotation
    to_annotation = Binarize(
        onset=0.5,
        offset=0.5,
        min_duration_on=0.0,
        min_duration_off=0.0
    )
    result = to_annotation(discrete_diarization)
    result.uri = sess_name

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This script performs diarization using DiariZen pipeline ",
        add_help=True,
        usage="%(prog)s [options]",
    )

    # Required arguments
    parser.add_argument(
        "-C",
        "--configuration",
        type=str,
        required=True,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "-i", 
        "--in_wav_scp",
        type=str,
        required=True,
        help="test wav.scp.",
        dest="in_wav_scp",
    )
    parser.add_argument(
        "-o", 
        "--out_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to pretrained embedding model.",
    )

    # Optional arguments
    parser.add_argument(
        "--diarizen_hub",
        type=str,
        help="Path to DiariZen model hub directory."
    )
    parser.add_argument(
        "--avg_ckpt_num",
        type=int,
        default=5,
        help="the number of chckpoints of model averaging",
    )
    parser.add_argument(
        "--val_metric",
        type=str,
        default="Loss",
        help="validation metric",
        choices=["Loss", "DER"],
    )
    parser.add_argument(
        "--val_mode",
        type=str,
        default="best",
        help="validation metric mode",
        choices=["best", "prev", "center"],
    )
    parser.add_argument(
        "--val_metric_summary",
        type=str,
        default="",
        help="val_metric_summary",
    )
    parser.add_argument(
        "--segmentation_model",
        type=str,
        default="",
        help="Path to pretrained segmentation model.",
    )

    # Inference parameters
    parser.add_argument(
        "--seg_duration",
        type=int,
        default=16,
        help="Segment duration in seconds.",
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Shifting ratio during segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Input batch size for inference.",
    )
    parser.add_argument(
        "--apply_median_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply median filtering to segmentation output.",
    )

    # Clustering parameters
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="VBxClustering",
        choices=["VBxClustering", "AgglomerativeClustering"],
        help="Clustering method to use.",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=1,
        help="Minimum number of speakers.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=20,
        help="Maximum number of speakers.",
    )
    parser.add_argument(
        "--ahc_criterion",
        type=str,
        default="distance",
        help="AHC criterion (for VBx).",
    )
    parser.add_argument(
        "--ahc_threshold",
        type=float,
        default=0.6,
        help="AHC threshold.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=13,
        help="Minimum cluster size (for AHC).",
    )
    parser.add_argument(
        "--Fa",
        type=float,
        default=0.07,
        help="VBx Fa parameter.",
    )
    parser.add_argument(
        "--Fb",
        type=float,
        default=0.8,
        help="VBx Fb parameter.",
    )
    parser.add_argument(
        "--lda_dim",
        type=int,
        default=128,
        help="VBx LDA dimension.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="VBx maximum iterations.",
    )

    args = parser.parse_args()
    print(args)

    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())
    
    ckpt_path = config_path.parent / 'checkpoints'
    segmentation = args.segmentation_model
    if args.val_metric_summary:
        val_metric_lst = load_metric_summary(args.val_metric_summary, ckpt_path)
        val_metric_lst_sorted = sorted(val_metric_lst, key=lambda i: i[args.val_metric])
        best_val_metric_idx = val_metric_lst.index(val_metric_lst_sorted[0])
        if args.val_mode == "best":
            segmentation = val_metric_lst_sorted[:args.avg_ckpt_num]
        elif args.val_mode == "prev":
            segmentation = val_metric_lst[
                best_val_metric_idx - args.avg_ckpt_num + 1 :
                best_val_metric_idx + 1
            ]
        else:
            segmentation = val_metric_lst[
                best_val_metric_idx - args.avg_ckpt_num // 2 :
                best_val_metric_idx + args.avg_ckpt_num // 2 + 1
            ]
        assert len(segmentation) == args.avg_ckpt_num

    # create, instantiate and apply the pipeline
    diarization_pipeline = SpeakerDiarizationPipeline(
        config=config,      
        seg_duration=args.seg_duration,
        segmentation=segmentation,
        segmentation_step=args.segmentation_step,
        embedding=args.embedding_model,
        embedding_exclude_overlap=True,
        clustering=args.clustering_method,
        embedding_batch_size=args.batch_size,
        segmentation_batch_size=args.batch_size,
        device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    if args.clustering_method == "AgglomerativeClustering":
        PIPELINE_PARAMS = {
            "clustering": {
                "method": "centroid",
                "min_cluster_size": args.min_cluster_size,
                "threshold": args.ahc_threshold,
            }
        }
    elif args.clustering_method == "VBxClustering":
        PIPELINE_PARAMS = {
            "clustering": {
                "ahc_criterion": args.ahc_criterion,
                "ahc_threshold": args.ahc_threshold,
                "Fa": args.Fa,
                "Fb": args.Fb,
            }
        }
        diarization_pipeline.clustering.plda_dir = os.path.join(args.diarizen_hub, "plda")
        diarization_pipeline.clustering.lda_dim = args.lda_dim
        diarization_pipeline.clustering.maxIters = args.max_iters
    else:
        raise ValueError(f"Unsupported clustering method: {args.clustering_method}")

    diarization_pipeline.instantiate(PIPELINE_PARAMS)
    
    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    audio_dict = load_scp(args.in_wav_scp)

    for sess, in_wav in audio_dict.items():
        print(f"Diarizing Session: {sess}")
        diar_result = diarize_session(
            sess_name=sess,
            in_wav=in_wav,
            pipeline=diarization_pipeline,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            apply_median_filtering=args.apply_median_filtering
        )
        rttm_out = os.path.join(args.out_dir, sess + ".rttm")
        with open(rttm_out, "w") as f:
            f.write(diar_result.to_rttm())
# # Licensed under the MIT license.
# # Adopted from https://github.com/espnet/espnet/blob/master/egs2/chime8_task1/diar_asr1/local/pyannote_diarize.py
# # Copyright 2024 Brno University of Technology

# import os
# import argparse
# import toml
# from pathlib import Path
# from typing import Dict
# import torch
# import numpy as np
# import torchaudio
# from scipy.ndimage import median_filter
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# from pyannote.metrics.segmentation import Annotation, Segment
# from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
# from pyannote.audio.utils.signal import Binarize

# from diarizen.ckpt_utils import load_metric_summary


# # ===============================
# # Utility
# # ===============================

# def load_scp(scp_file: str) -> Dict[str, str]:
#     lines = [line.strip().split(None, 1) for line in open(scp_file)]
#     return {x[0]: x[1] for x in lines}


# # ===============================
# # t-SNE Visualization (Paper Style)
# # ===============================

# def generate_tsne_plot(sess_name, embeddings, hard_clusters, out_dir):

#     tsne_dir = os.path.join(out_dir, "tsne_plots")
#     os.makedirs(tsne_dir, exist_ok=True)

#     num_chunks, local_spk, dim = embeddings.shape
#     flat_embeddings = embeddings.reshape(-1, dim)
#     flat_clusters = hard_clusters.reshape(-1)

#     # remove inactive speakers
#     valid_idx = flat_clusters >= 0
#     flat_embeddings = flat_embeddings[valid_idx]
#     flat_clusters = flat_clusters[valid_idx]

#     if len(flat_embeddings) < 10:
#         return

#     # ---- downsample if too large (for speed) ----
#     MAX_POINTS = 6000
#     if len(flat_embeddings) > MAX_POINTS:
#         idx = np.random.choice(len(flat_embeddings), MAX_POINTS, replace=False)
#         flat_embeddings = flat_embeddings[idx]
#         flat_clusters = flat_clusters[idx]

#     # ---- PCA pre-reduction ----
#     pca = PCA(n_components=min(50, dim))
#     flat_embeddings = pca.fit_transform(flat_embeddings)

#     # ---- t-SNE ----
#     tsne = TSNE(
#         n_components=2,
#         perplexity=30,
#         learning_rate="auto",
#         init="pca",
#         random_state=42,
#     )
#     reduced = tsne.fit_transform(flat_embeddings)

#     # ---- Plot (paper style) ----
#     plt.figure(figsize=(7, 6))
#     unique_labels = np.unique(flat_clusters)

#     cmap = plt.cm.get_cmap("tab20", len(unique_labels))

#     for i, label in enumerate(unique_labels):
#         idx = flat_clusters == label
#         plt.scatter(
#             reduced[idx, 0],
#             reduced[idx, 1],
#             s=8,
#             alpha=0.7,
#             label=f"Speaker {label}",
#             color=cmap(i),
#         )

#     plt.legend(
#         markerscale=2,
#         fontsize=9,
#         frameon=False,
#         loc="best"
#     )

#     plt.xticks([])
#     plt.yticks([])
#     plt.title(f"t-SNE of Speaker Embeddings\n{sess_name}", fontsize=12)
#     plt.tight_layout()

#     save_path = os.path.join(tsne_dir, f"{sess_name}_tsne.png")
#     plt.savefig(save_path, dpi=400, bbox_inches="tight")
#     plt.close()


# # ===============================
# # Diarization
# # ===============================

# def diarize_session(
#     sess_name,
#     in_wav,
#     pipeline,
#     out_dir,
#     min_speakers=1,
#     max_speakers=20,
#     apply_median_filtering=True,
# ):

#     print('Extracting segmentations...')
#     waveform, sample_rate = torchaudio.load(in_wav)
#     waveform = torch.unsqueeze(waveform[0], 0)

#     segmentations = pipeline.get_segmentations(
#         {"waveform": waveform, "sample_rate": sample_rate},
#         soft=False,
#     )

#     if apply_median_filtering:
#         segmentations.data = median_filter(
#             segmentations.data,
#             size=(1, 11, 1),
#             mode='reflect'
#         )

#     binarized_segmentations = segmentations

#     count = pipeline.speaker_count(
#         binarized_segmentations,
#         pipeline._segmentation.model._receptive_field,
#         warm_up=(0.0, 0.0),
#     )

#     print("Extracting Embeddings.")
#     embeddings = pipeline.get_embeddings(
#         {"waveform": waveform, "sample_rate": sample_rate},
#         binarized_segmentations,
#         exclude_overlap=pipeline.embedding_exclude_overlap,
#     )

#     print("Clustering.")
#     hard_clusters, _, _ = pipeline.clustering(
#         embeddings=embeddings,
#         segmentations=binarized_segmentations,
#         min_clusters=min_speakers,
#         max_clusters=max_speakers,
#     )

#     count.data = np.minimum(count.data, max_speakers).astype(np.int8)

#     inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
#     hard_clusters[inactive_speakers] = -2

#     # ======= Generate t-SNE visualization =======
#     print("Generating t-SNE visualization...")
#     try:
#         generate_tsne_plot(sess_name, embeddings, hard_clusters, out_dir)
#     except Exception as e:
#         print(f"t-SNE failed: {e}")

#     discrete_diarization, _ = pipeline.reconstruct(
#         segmentations,
#         hard_clusters,
#         count,
#     )

#     to_annotation = Binarize(
#         onset=0.5,
#         offset=0.5,
#         min_duration_on=0.0,
#         min_duration_off=0.0
#     )

#     result = to_annotation(discrete_diarization)
#     result.uri = sess_name
#     return result


# # ===============================
# # Main
# # ===============================

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(
#         "This script performs diarization using DiariZen pipeline ",
#         add_help=True,
#         usage="%(prog)s [options]",
#     )

#     parser.add_argument("-C", "--configuration", type=str, required=True)
#     parser.add_argument("-i", "--in_wav_scp", type=str, required=True)
#     parser.add_argument("-o", "--out_dir", type=str, required=True)
#     parser.add_argument("--embedding_model", type=str, required=True)

#     parser.add_argument("--diarizen_hub", type=str)
#     parser.add_argument("--avg_ckpt_num", type=int, default=5)
#     parser.add_argument("--val_metric", type=str, default="Loss")
#     parser.add_argument("--val_mode", type=str, default="best")
#     parser.add_argument("--val_metric_summary", type=str, default="")
#     parser.add_argument("--segmentation_model", type=str, default="")

#     parser.add_argument("--seg_duration", type=int, default=16)
#     parser.add_argument("--segmentation_step", type=float, default=0.1)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--apply_median_filtering",
#                         action=argparse.BooleanOptionalAction,
#                         default=True)

#     parser.add_argument("--clustering_method",
#                         type=str,
#                         default="VBxClustering",
#                         choices=["VBxClustering", "AgglomerativeClustering"])

#     parser.add_argument("--min_speakers", type=int, default=1)
#     parser.add_argument("--max_speakers", type=int, default=20)
#     parser.add_argument("--ahc_criterion", type=str, default="distance")
#     parser.add_argument("--ahc_threshold", type=float, default=0.6)
#     parser.add_argument("--min_cluster_size", type=int, default=13)
#     parser.add_argument("--Fa", type=float, default=0.07)
#     parser.add_argument("--Fb", type=float, default=0.8)
#     parser.add_argument("--lda_dim", type=int, default=128)
#     parser.add_argument("--max_iters", type=int, default=20)

#     args = parser.parse_args()

#     config_path = Path(args.configuration).expanduser().absolute()
#     config = toml.load(config_path.as_posix())

#     diarization_pipeline = SpeakerDiarizationPipeline(
#         config=config,
#         seg_duration=args.seg_duration,
#         segmentation=args.segmentation_model,
#         segmentation_step=args.segmentation_step,
#         embedding=args.embedding_model,
#         embedding_exclude_overlap=True,
#         clustering=args.clustering_method,
#         embedding_batch_size=args.batch_size,
#         segmentation_batch_size=args.batch_size,
#         device=torch.device("cuda:0") if torch.cuda.is_available()
#         else torch.device("cpu")
#     )

#     diarization_pipeline.instantiate({})

#     Path(args.out_dir).mkdir(exist_ok=True, parents=True)

#     audio_dict = load_scp(args.in_wav_scp)

#     for sess, in_wav in audio_dict.items():
#         print(f"Diarizing Session: {sess}")

#         diar_result = diarize_session(
#             sess_name=sess,
#             in_wav=in_wav,
#             pipeline=diarization_pipeline,
#             out_dir=args.out_dir,
#             min_speakers=args.min_speakers,
#             max_speakers=args.max_speakers,
#             apply_median_filtering=args.apply_median_filtering,
#         )

#         rttm_out = os.path.join(args.out_dir, sess + ".rttm")
#         with open(rttm_out, "w") as f:
#             f.write(diar_result.to_rttm())
