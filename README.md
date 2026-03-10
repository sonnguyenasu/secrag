# SecureRAG MVP (Pseudo-Remote)

This repository implements a runnable MVP of the SecureRAG API design using a pseudo-remote backend service on localhost.

In addition to HTTP, the remote backend now supports a gRPC + Protobuf service endpoint.

It also includes a Rust backend crate (`securerag-rs`) with a PyO3 bridge that can be selected via `backend="rust://local"` after building the extension.

## What is implemented

- Protocol-centric API (`PrivacyProtocol`, `PrivacyConfig`)
- `BudgetManager` with epsilon ledger
- `SecureCorpus` + `CorpusBuilder`
- `PrivacyRetriever` registry/factory
- Functional retrievers:
  - `BASELINE`
  - `OBFUSCATION`
  - `DIFF_PRIVACY`
- Contract-complete retrievers with explicit capability errors:
  - `ENCRYPTED_SEARCH`
  - `PIR`
- `SecureRAGAgent` multi-round orchestration
- Pseudo-remote backend server (`FastAPI`) reachable on localhost

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn securerag.sim_server:app --host 127.0.0.1 --port 8099
```

In another terminal:

```bash
python examples/quickstart.py
```

## Rust backend (PyO3)

Build and install the Rust extension module:

```bash
source .venv/bin/activate
pip install maturin
cd securerag-rs
maturin develop --features python-bridge
cd ..
```

Then set backend in config:

```python
config = PrivacyConfig(
  protocol=PrivacyProtocol.DIFF_PRIVACY,
  backend="rust://local",
)
```

If the extension is unavailable, the framework raises a clear error and you can continue using the pseudo-remote backend at `http://127.0.0.1:8099`.

## gRPC backend (Protobuf)

Start the gRPC server:

```bash
source .venv/bin/activate
python -m securerag.grpc_server --host 127.0.0.1 --port 50051
```

Then configure backend:

```python
config = PrivacyConfig(
  protocol=PrivacyProtocol.BASELINE,
  backend="grpc://127.0.0.1:50051",
)
```

## Provider-agnostic LLM agent

The agent is framework/provider-agnostic and supports both Ollama and Hugging Face backends through one interface:

- `ModelAgentLLM(provider="ollama", model="qwen3:0.6b")`
- `ModelAgentLLM(provider="huggingface", model="google/flan-t5-base")`

Quickstart environment variables:

- `SECURERAG_LLM_PROVIDER=ollama|huggingface`
- `SECURERAG_OLLAMA_MODEL=qwen3:0.6b`
- `SECURERAG_HF_MODEL=google/flan-t5-base`
- `SECURERAG_USE_OLLAMA=1` to enable live Ollama calls
- `SECURERAG_USE_HUGGINGFACE=1` to enable live Hugging Face calls
- `HF_TOKEN=<token>` (optional/required depending on endpoint limits)
- `HF_INFERENCE_BASE_URL=https://api-inference.huggingface.co`

If remote model calls are disabled or unavailable, deterministic local fallback remains active.
