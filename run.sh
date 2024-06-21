#!/bin/bash

TEST_DIRS=("single_card" "dual_cards")
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
        if [ -x "$SCRIPT" ]; then
            SCRIPT_NAME=$(basename "$SCRIPT")
            LOG_FILE="${LOG_DIR}/${TEST_DIR}_${SCRIPT_NAME}.log"
            echo "Running $SCRIPT, logging to $LOG_FILE..."

            echo "### Script Content: ${SCRIPT}" > "$LOG_FILE"
            cat "$SCRIPT" >> "$LOG_FILE"
            echo -e "\n### Script Output:\n" >> "$LOG_FILE"

            ./"$SCRIPT" &>> "$LOG_FILE"
            echo "$SCRIPT finished. Output logged to $LOG_FILE"
            echo
        else
            echo "$SCRIPT is not executable or not a valid file."
        fi
    done
done

echo "All scripts in specified directories have been run."
