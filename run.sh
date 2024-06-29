#!/bin/bash

# TEST_DIRS=("single_card" "dual_cards")
# TEST_DIRS=("single_card")
TEST_DIRS=("single_card/qwen1.5-14b")
LOG_DIR="logs"

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for TEST_DIR in "${TEST_DIRS[@]}"; do
    if [ ! -d "$TEST_DIR" ]; then
        echo "Directory $TEST_DIR does not exist."
        continue
    fi

    for SCRIPT in "$TEST_DIR"/*.sh; do
        if [ ! -f "$SCRIPT" ]; then
            continue
        fi

        SCRIPT_NAME=$(basename "$SCRIPT")
        PARENT_DIR=$(basename "$(dirname "$TEST_DIR")")
        CURRENT_DIR=$(basename "$TEST_DIR")
        LOG_FILE="${LOG_DIR}/${PARENT_DIR}_${CURRENT_DIR}_${SCRIPT_NAME}.log"
        echo "Running $SCRIPT, logging to $LOG_FILE..."

        echo "### Script Content: ${SCRIPT}" > "$LOG_FILE"
        cat "$SCRIPT" >> "$LOG_FILE"
        echo -e "\n### Script Output:\n" >> "$LOG_FILE"

        bash "$SCRIPT" 2>&1 | tee -a "$LOG_FILE"
        echo "$SCRIPT finished. Output logged to $LOG_FILE"
        echo
    done
done

echo "All scripts in specified directories have been run."
