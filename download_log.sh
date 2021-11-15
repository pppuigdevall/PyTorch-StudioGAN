#!/usr/bin/env bash
# usage bash download_log.sh logs/SAGAN-train-2021_11_12_16_55_27/
LOGS_PATH=$1

echo $LOGS_PATH

if [ -d "$LOGS_PATH" ]; then
  # Control will enter here if $DIRECTORY exists.
  echo It exists
  zip -r "${LOGS_PATH%/}.zip" $LOGS_PATH
  echo "${LOGS_PATH%/}.zip"
fi
