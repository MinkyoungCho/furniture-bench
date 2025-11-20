#!/bin/bash
# Quick test script for round_table dual-arm assembly

echo "Starting round_table dual-arm assembly test..."
echo "=============================================="

cd /y/hymanzzs/furniture-bench

python -m furniture_bench.scripts.run_sim_env \
    --furniture round_table \
    --scripted \
    --num-envs 1 \
    --compute-device-id 0 \
    --graphics-device-id 0 \
    "$@"

echo ""
echo "Test complete!"
