#!/bin/bash

# Working directory (current directory)
WORKSPACE_DIR=$(pwd)

# One timestamp per full script execution (Option 1)
RUN_TS=$(date +"%Y%m%d_%H%M%S")

# Function to terminate all Python processes and free the port
cleanup() {
    echo "Cleaning up background jobs..."
    kill $(jobs -p) 2>/dev/null
    echo "Freeing port $1 if occupied..."
    lsof -ti :$1 | xargs kill -9 2>/dev/null
    echo "Cleanup completed."
}

# Function to run a simulation and save the output in the log file
run_simulation() {
    local mode=$1
    local protocol=$2
    local fl_method=$3
    local port=$4
    local client_number=$5
    local alpha=$6
    local mu=$7
    local variant=$8
    local file_name=$9

    local log_file="$LOG_DIR/${mode}_${client_number}.log"
	# Default rounds for most algorithms
    local rounds=5
	
	# FedBuff needs enough updates to fill the buffer (K) several times
	if [ "$fl_method" == "FedBuff" ]; then
		local K="${FEDBUFF_K:-10}"   # FEDBUFF_K est optionnel, défaut=10
		local FLUSHES=5
		rounds=$((K * FLUSHES))
	fi

	# extra_args=()
	# if [ "$fl_method" == "FedBuff" ]; then
		# extra_args+=(--fedbuff_buffer_size "$K")
	# fi
	extra_args=()
	if [ "$fl_method" == "FedBuff" ]; then
		extra_args+=(--fedbuff_buffer_size "$K")

		# Timeout (optionnel). Si FEDBUFF_FLUSH_TIMEOUT_S vaut 0 => désactivé
		# TIMEOUT_S="${FEDBUFF_FLUSH_TIMEOUT_S:-0}"
		# CHECK_S="${FEDBUFF_TIMEOUT_CHECK_INTERVAL_S:-0.2}"
		# MIN_UPDATES="${FEDBUFF_TIMEOUT_MIN_UPDATES:-1}"

		if [[ "$TIMEOUT_S" != "0" && "$TIMEOUT_S" != "0.0" ]]; then
			extra_args+=(--fedbuff_flush_timeout_s "$TIMEOUT_S")
			extra_args+=(--fedbuff_timeout_check_interval_s "$CHECK_S")
			extra_args+=(--fedbuff_timeout_min_updates "$MIN_UPDATES")
		fi
	fi

	
	echo "Running: mode=$mode, fl_method=$fl_method, port=$port, rounds=$rounds"
	python3 "$WORKSPACE_DIR/FedOpt/src/run.py" \
		--mode "$mode" \
		--rounds "$rounds" \
		--protocol "$protocol" \
		--port "$port" \
		--fl_method "$fl_method" \
		--alpha "$alpha" \
		--accuracy_file "$file_name" \
		"${extra_args[@]}" \
		&> "$log_file" &

    # echo "Running: mode=$mode, fl_method=$fl_method, port=$port, rounds=$rounds"
    # python3 "$WORKSPACE_DIR/FedOpt/src/run.py" \
        # --mode "$mode" \
        # --rounds "$rounds" \
        # --protocol "$protocol" \
        # --port "$port" \
        # --fl_method "$fl_method" \
        # --alpha "$alpha" \
        # --accuracy_file "$file_name" \
        # &> "$log_file" &
    # echo "Log saved to $log_file"
}

#PORT="8082"
#PROTOCOL="grpc"

PORT="1883"
PROTOCOL="mqtt"

NUM_CLIENTS=5

#ALGORITHMs=("Unweighted")
ALGORITHMs=("FedBuff")
MUs=(0.1)
ALPHA=(10)
VARIANTS=("hinge" "polynomial")

# Run cleanup before starting new simulations
cleanup "$PORT"

# If using MQTT protocol, ensure Mosquitto exists and start the broker
if [ "$PROTOCOL" == "mqtt" ]; then
    if ! command -v mosquitto >/dev/null 2>&1; then
        echo "ERROR: mosquitto not found. Install it with:"
        echo "  sudo apt update && sudo apt install -y mosquitto mosquitto-clients"
        exit 1
    fi
    mosquitto -p "$PORT" -d
fi

# Ensure cleanup runs when the script exits
trap 'cleanup "$PORT"' EXIT

# Loop through the algorithms
for alpha in "${ALPHA[@]}"; do
    for ALGORITHM in "${ALGORITHMs[@]}"; do
        if [ "$ALGORITHM" == "FedAsync" ]; then
            for variant in "${VARIANTS[@]}"; do
                for MU_0 in "${MUs[@]}"; do
                    echo "Starting simulations for algorithm: $ALGORITHM"

                    LOG_DIR="$WORKSPACE_DIR/logs/$PROTOCOL/$ALGORITHM/$variant/${RUN_TS}_alpha${alpha}_${NUM_CLIENTS}clients"
                    mkdir -p "$LOG_DIR"
                    ACC_FILE="$LOG_DIR/accuracy_alpha${alpha}_${NUM_CLIENTS}clients.png"

                    # Start the server
                    run_simulation "server" "$PROTOCOL" "$ALGORITHM" "$PORT" "0" "$alpha" "$MU_0" "$variant" "$ACC_FILE"
                    sleep 5 # Wait for the server to start

                    # Start clients 1..NUM_CLIENTS
                    for i in $(seq 1 $NUM_CLIENTS); do
                        run_simulation "client" "$PROTOCOL" "$ALGORITHM" "$PORT" "$i" "$alpha" "$MU_0" "$variant" "$ACC_FILE"
                    done

                    wait
                done
            done
        else
            for MU_0 in "${MUs[@]}"; do
                echo "Starting simulations for algorithm: $ALGORITHM"

                LOG_DIR="$WORKSPACE_DIR/logs/$PROTOCOL/$ALGORITHM/${RUN_TS}_alpha${alpha}_${NUM_CLIENTS}clients"
                mkdir -p "$LOG_DIR"
                ACC_FILE="$LOG_DIR/accuracy_alpha${alpha}_${NUM_CLIENTS}clients.png"

                # Start the server
                run_simulation "server" "$PROTOCOL" "$ALGORITHM" "$PORT" "0" "$alpha" "$MU_0" "" "$ACC_FILE"
                sleep 5 # Wait for the server to start

                # Start clients 1..NUM_CLIENTS
                for i in $(seq 1 $NUM_CLIENTS); do
                    run_simulation "client" "$PROTOCOL" "$ALGORITHM" "$PORT" "$i" "$alpha" "$MU_0" "" "$ACC_FILE"
                done

                wait
            done
        fi
    done
done

echo "All simulations have been completed!"
