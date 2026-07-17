set -uo pipefail
SIF="/work/hezhang/docker_images/compsteer-groot.sif"
REPO="/work/hezhang/latentSteerVLA"
ROBOCASA_DATA="/work/hezhang/robocasa_data"   # persistent isolated venv + ~10GB kitchen assets
HF_CACHE="/work/hezhang/hf_cache"
SCRATCH="/tmp/${SLURM_JOB_ID:-manual}"        # node-local scratch for apptainer's own tmp/cache
MODEL_PATH="nvidia/GR00T-N1.6-3B"
EMBODIMENT_TAG="ROBOCASA_PANDA_OMRON"
PORT=5555




N_EPISODES="${1:-20}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -eq 0 ]; then
    TASKS=(all)
else
    TASKS=("$@")
fi

mkdir -p logs "$ROBOCASA_DATA" "$HF_CACHE" "$SCRATCH"

module load apptainer-1.4.1

export APPTAINER_TMPDIR="$SCRATCH/apptainer_tmp"
export APPTAINER_CACHEDIR="$SCRATCH/apptainer_cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

SERVER_OVERLAY="$SCRATCH/overlay_server"
CLIENT_OVERLAY="$SCRATCH/overlay_client"
mkdir -p "$SERVER_OVERLAY" "$CLIENT_OVERLAY"

BINDS=(-B "${REPO}:/workspace/compsteer")
BINDS+=(-B "${ROBOCASA_DATA}:/opt/robocasa_persist")
BINDS+=(-B "${HF_CACHE}:/opt/hf_cache")

APP_ENV=(--env "HF_HOME=/opt/hf_cache" --env "UV_LINK_MODE=copy" --env "ROBOCASA_PERSIST_DIR=/opt/robocasa_persist")
if [ -n "${HF_TOKEN:-}" ]; then
    APP_ENV+=(--env "HF_TOKEN=${HF_TOKEN}")
fi


echo "======================================"
echo "GR00T x RoboCasa raw eval"
echo "  Node:       $(hostname)"
echo "  SIF:        $SIF"
echo "  Model:      $MODEL_PATH"
echo "  Embodiment: $EMBODIMENT_TAG"
echo "  Episodes:   $N_EPISODES"
echo "  Tasks:      ${TASKS[*]}"
echo "======================================"


SERVER_LOG="logs/robocasa_eval_${SLURM_JOB_ID:-manual}_server.log"
apptainer exec --nv --cleanenv --overlay "$SERVER_OVERLAY" "${BINDS[@]}" "${APP_ENV[@]}" "$SIF" \
    bash /workspace/compsteer/scripts/run_groot_server.sh "$MODEL_PATH" "$EMBODIMENT_TAG" "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "Stopping GR00T server (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# ── 2. Wait for the server socket to come up ───────────────────────────────
echo "Waiting for GR00T server on port $PORT..."
UP=0
for i in $(seq 1 60); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
        exec 3>&- 3<&-
        echo "Server socket is up after $((i * 10))s."
        UP=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server process died during startup. Log tail:"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
    sleep 10
done

if [ "$UP" -eq 0 ]; then
    echo "ERROR: server did not open port $PORT within 600s. Log tail:"
    tail -50 "$SERVER_LOG"
    exit 1
fi

# Socket accepting connections doesn't guarantee the model has finished
# loading yet — a short grace period before hammering it with rollouts.
sleep 30


RESULTS_DIR="results/raw_groot/robocasa_${SLURM_JOB_ID:-manual}"
apptainer exec --nv --cleanenv --overlay "$CLIENT_OVERLAY" "${BINDS[@]}" "${APP_ENV[@]}" "$SIF" \
    bash -c '
        set -e
        bash /workspace/compsteer/scripts/setup_robocasa_env.sh
        python /workspace/compsteer/scripts/eval_groot_robocasa.py \
            --tasks "$@" \
            --n_episodes "'"$N_EPISODES"'" \
            --policy_host 127.0.0.1 \
            --policy_port "'"$PORT"'" \
            --results_root "/workspace/compsteer/'"$RESULTS_DIR"'"
    ' _ "${TASKS[@]}"

echo "Done. Results -> $REPO/$RESULTS_DIR"