#!/usr/bin/env bash
set -euo pipefail

# ====== 路径配置 ======
LIST_FILE="/home3/yihao/Research/Code/DiariZen/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4/dev/all.uem"
AMI_ROOT="/home3/yihao/Research/Code/DiariZen/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4/train"
OUT_DIR="/home3/yihao/Research/Code/DiariZen/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4/dev"

mkdir -p "$OUT_DIR"

# ====== 主逻辑 ======
while read -r meeting_id _; do
    src_wav="${AMI_ROOT}/${meeting_id}.wav"
    dst_wav="${OUT_DIR}/${meeting_id}.wav"

    if [[ -f "$src_wav" ]]; then
        echo "Moving $src_wav -> $dst_wav"
        mv "$src_wav" "$dst_wav"
    else
        echo "WARNING: not found $src_wav"
    fi
done < "$LIST_FILE"
