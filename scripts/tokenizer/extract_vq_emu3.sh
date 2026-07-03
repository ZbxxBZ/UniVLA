NGPUS=${NGPUS:-8}
PYTHON=${PYTHON:-python3}
TOKENIZER_SCRIPT=${TOKENIZER_SCRIPT:-models/tokenizer/emu3_tokenizer.py}
PROCESS_DATA=${PROCESS_DATA:-Calvin}

if [ -n "${CUDA_DEVICES:-}" ]; then
    IFS=',' read -ra DEVICES <<< "${CUDA_DEVICES}"
    NGPUS=${#DEVICES[@]}
else
    DEVICES=()
    for ((rank = 0; rank < NGPUS; rank++)); do
        DEVICES+=("${rank}")
    done
fi

for ((rank = 0; rank < NGPUS; rank++)); do
    CUDA_VISIBLE_DEVICES=${DEVICES[$rank]} \
        ${PYTHON} "${TOKENIZER_SCRIPT}" "${rank}" \
            --world-size "${NGPUS}" \
            --process-data "${PROCESS_DATA}" \
            "$@" &
done

wait
