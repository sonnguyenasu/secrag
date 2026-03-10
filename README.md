# SecureRAG Framework (Research Prototype)

SecureRAG is a protocol-driven retrieval augmented generation framework focused on privacy-aware retrieval.

This repository provides:

- A Python orchestration layer for corpus construction, retrieval routing, budget accounting, and LLM interaction.
- Multiple backend transports: HTTP RPC, strict gRPC + Protobuf service, and an in-process Rust bridge (`rust://local`).
- Protocol-specific retrieval behavior for baseline retrieval, traffic obfuscation, differential privacy, and encrypted search.

The goal of this implementation is to mirror the framework API shape while remaining runnable for local experiments.

## Framework model

At a high level, the system is organized into four layers:

1. Policy layer: `PrivacyConfig`, `PrivacyProtocol`, and budget controls define privacy and runtime behavior.
2. Data layer: `CorpusBuilder` builds protocol-aware corpora (`SSECorpus`, `EmbeddingCorpus`, `PIRDatabase`, or `GenericCorpus`).
3. Retrieval layer: `PrivacyRetriever` selects a retriever implementation based on protocol.
4. Agent layer: `SecureRAGAgent` runs multi-round retrieve-then-generate with model-provider abstraction (`ModelAgentLLM`).

Execution flow:

1. Configure protocol and backend.
2. Build corpus and backend index.
3. Instantiate `SecureRAGAgent` with selected LLM provider.
4. Run iterative retrieval rounds until answer condition is met.
5. Return answer plus context and budget telemetry.

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
- Typed corpora:
  - `SSECorpus`
  - `EmbeddingCorpus`
  - `PIRDatabase`
- Backend transports:
  - HTTP pseudo-remote (`FastAPI`)
  - strict gRPC + Protobuf service (`SecureRetrieval`)
  - Rust local bridge (`rust://local`)

## Repository layout

- `securerag/`: Python framework code (agent, config, retrievers, corpus, backends, gRPC client)
- `securerag/proto/`: Protobuf contract and generated gRPC stubs
- `securerag-rs/`: Rust crate with retrieval engines, builders, PyO3 bridge, and native gRPC server binary
- `examples/`: runnable usage patterns
- `tests/`: parity and behavior tests

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

`examples/quickstart.py` is the primary end-to-end reference and supports env-based backend/provider switching.

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

Start the Rust-native gRPC server:

```bash
source .venv/bin/activate
cargo build --manifest-path securerag-rs/Cargo.toml --bin securerag_grpc_server
./securerag-rs/target/debug/securerag_grpc_server --host 127.0.0.1 --port 50051
```

Then configure backend:

```python
config = PrivacyConfig(
  protocol=PrivacyProtocol.BASELINE,
  backend="grpc://127.0.0.1:50051",
)
```

The gRPC surface is strict and typed. The `SecureRetrieval` service exposes explicit methods for each operation (`Chunk`, `BuildIndex`, `BatchRetrieve`, `SseEncryptTerms`, `SseSearch`, etc.) rather than a generic string-dispatched RPC envelope.

Proto file:

- `securerag/proto/secure_retrieval.proto`

Rust server entrypoint:

- `securerag-rs/src/bin/securerag_grpc_server.rs`

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

## Example scripts

- `examples/quickstart.py`
  - End-to-end encrypted-search run with configurable backend and model provider.
- `examples/grpc_quickstart.py`
  - End-to-end run against strict gRPC backend (`grpc://127.0.0.1:50051`) served by the Rust binary.
- `examples/protocol_walkthrough.py`
  - Runs multiple protocols and shows resulting corpus type, answer, and budget snapshot.

## Notes and current boundaries

- This codebase is intended for research prototyping and API validation.
- `PIR` remains contract-complete but intentionally unimplemented as a retrieval algorithm in this repository.
- For tests that use default HTTP backend values, start `securerag.sim_server` (or use the provided test command patterns that launch it temporarily).
