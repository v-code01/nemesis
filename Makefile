.PHONY: build test sim bench clean proto-gen

proto-gen:
	mkdir -p agents/nemesis/grpc
	python -m grpc_tools.protoc \
		-I proto \
		--python_out=agents/nemesis/grpc \
		--grpc_python_out=agents/nemesis/grpc \
		proto/telemetry.proto proto/topology.proto proto/healer.proto

build:
	cd substrate && cargo build --release
	cd sim/crates/nemesis-sim && cargo build --release
	$(MAKE) proto-gen
	cd agents && pip install -e . -q

test:
	cd substrate && cargo test --workspace
	cd sim/crates/nemesis-sim && cargo test
	cd agents && pytest tests/ -v

sim:
	minikube start --cpus=4 --memory=8g --driver=docker 2>/dev/null || true
	kubectl apply -f sim/k8s/
	kubectl apply -f agents/k8s/ 2>/dev/null || true
	kubectl wait --for=condition=available deployment/nemesis-sim --timeout=120s

bench: bench-ecc bench-scheduler bench-nccl

bench-ecc:
	python benchmarks/p1_ecc_prediction/run.py --trace-dir data/alibaba --seed 42 --output results/ecc.json
	python benchmarks/assert_gate.py --result results/ecc.json --metric f1_2h --min 0.90

bench-scheduler:
	python benchmarks/p2_scheduler_mfu/run.py --seed 42 --output results/mfu.json
	python benchmarks/assert_gate.py --result results/mfu.json --metric mfu_ratio --min 1.4

bench-nccl:
	python benchmarks/p3_nccl_shrink/run_full.py --seed 42 --output results/nccl.json
	python benchmarks/assert_gate.py --result results/nccl.json --metric resumption_seconds --max 30
	python benchmarks/assert_gate.py --result results/nccl.json --metric job_restart_count --max 0

clean:
	cd substrate && cargo clean
	rm -rf results/
