#!/usr/bin/env bash
set -e

AMI_ROOT=amicorpus
AUDIO_SUBDIR=audio

MEETINGS=(
TS3006c TS3006d
TS3007a TS3007b TS3007c TS3007d
TS3008a TS3008b TS3008c TS3008d
TS3009a TS3009b TS3009c TS3009d
TS3010a TS3010b TS3010c TS3010d
TS3011a TS3011b TS3011c TS3011d
TS3012a TS3012b TS3012c TS3012d
)

BASE_URL="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

echo "Downloading AMI microphone array audio (Array1 channel 01 only)..."

for m in "${MEETINGS[@]}"; do
  out_dir="$AMI_ROOT/$m/$AUDIO_SUBDIR"
  mkdir -p "$out_dir"

  fname="$m.Array1-01.wav"
  url="$BASE_URL/$m/audio/$fname"

  echo "Downloading $m (Array1-01)..."
  wget -c \
    --inet4-only \
    --tries=0 \
    --timeout=30 \
    --read-timeout=30 \
    --retry-connrefused \
    -O "$out_dir/$fname" \
    "$url"
done

echo "Downloading license file..."
wget -c --inet4-only \
  https://groups.inf.ed.ac.uk/ami/download/CCBY4.0.txt \
  -O "$AMI_ROOT/CCBY4.0.txt"

echo "✅ AMI Array1 ch1 download finished."
