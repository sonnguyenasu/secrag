# SecureRAG — Encrypted Search Refactor Specification

**Target:** Make the encrypted search subsystem open to researcher extension without touching
framework core, while achieving realistic benchmarking performance on large corpora.

**Scope:** Python layer (`retrievers.py`, `corpus.py`, `backend_client.py`), sim_server
(`sim_server.py`), Rust PyO3 bridge (`pyo3_bridge.rs`), gRPC server
(`securerag_grpc_server.rs`), and proto (`secure_retrieval.proto`). The `PrivacyProtocol`
enum, `SecureRAGAgent`, `BudgetManager`, and all DP/PIR code are out of scope.

---

## 1. Root Cause Summary

The current encrypted search path has three interlocking problems:

1. **Scheme dispatch is a hard-coded if/elif chain** in three files. Adding any new scheme
   requires modifying framework core.

2. **The per-document index structure is fixed.** Every row stores `enc_terms: list[str]`
   and `struct_terms: list[str]`. A new scheme that needs a different representation
   (encrypted counts, Bloom filters, label/value pairs for OXT) has nowhere to put it.

3. **Search is O(N) linear scan.** Both sim_server and the Rust bridge iterate every chunk
   for every query. This is fine for 3-doc test corpora; it makes MS-MARCO benchmarks
   completely unworkable.

The three problems share a fix: move all scheme-specific logic — key generation, corpus
preparation, query encryption, and search — into a plugin object that the framework calls
without inspecting. The framework becomes scheme-agnostic. The inverted index moves into
the plugin's `build_index` hook, which is called once at corpus build time.

---

## 2. New Abstraction: `EncryptedSchemePlugin`

### 2.1 Class definition

Add a new file `securerag/scheme_plugin.py`:

```python
# securerag/scheme_plugin.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

_REGISTRY: dict[str, "EncryptedSchemePlugin"] = {}


class EncryptedSchemePlugin(ABC):
    """
    All scheme-specific logic for one encrypted search algorithm.

    The framework calls these methods in the following order:

        Corpus build time (client-side, once):
            key  = plugin.generate_key()
            rows = [plugin.prepare_chunk(text, key) for text in corpus]
            idx  = plugin.build_server_index(rows)       # stored on server

        Query time (client-side, per query):
            enc_query = plugin.encrypt_query(query, key)

        Query time (server-side, per query):
            results   = plugin.search(server_index, enc_query, top_k)
    """

    # -----------------------------------------------------------------
    # Client-side: corpus build
    # -----------------------------------------------------------------

    @abstractmethod
    def generate_key(self) -> Any:
        """
        Generate and return a scheme key.  Called once per corpus build.
        The key stays on the client; it is never sent to the server.
        """

    @abstractmethod
    def prepare_chunk(self, text: str, key: Any) -> dict[str, Any]:
        """
        Transform one plaintext chunk into the server-stored representation.
        Return a dict of arbitrary fields; these are stored verbatim as the
        chunk's `scheme_data` on the server.

        Example (plain SSE):
            {"enc_terms": ["hmac(tok1,k)", "hmac(tok2,k)", ...]}

        Example (OXT):
            {"t_set": [...], "x_set": [...]}
        """

    # -----------------------------------------------------------------
    # Server-side: index build  (optional override)
    # -----------------------------------------------------------------

    def build_server_index(self, rows: list[dict[str, Any]]) -> Any:
        """
        Post-process the list of prepared rows into whatever server-side
        structure enables efficient search.  The return value is stored
        opaquely by the framework and passed back as `server_index` in
        `search()`.

        Default: return the rows list unchanged (linear scan).
        Override to build an inverted index, sorted buckets, B-tree, etc.
        """
        return rows

    # -----------------------------------------------------------------
    # Client-side: query time
    # -----------------------------------------------------------------

    @abstractmethod
    def encrypt_query(self, query: str, key: Any) -> dict[str, Any]:
        """
        Transform a plaintext query into an encrypted query token set.
        Return a dict; it is sent to the server as the `encrypted_query`
        field and passed to `search()` unchanged.

        Example (plain SSE):
            {"enc_terms": ["hmac(tok1,k)", "hmac(tok2,k)"]}
        """

    # -----------------------------------------------------------------
    # Server-side: search
    # -----------------------------------------------------------------

    @abstractmethod
    def search(
        self,
        server_index: Any,
        encrypted_query: dict[str, Any],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Execute the encrypted search.  `server_index` is the value returned
        by `build_server_index`.  Return a list of dicts, each with at least
        `doc_id`, `text`, and `score`.
        """

    # -----------------------------------------------------------------
    # Registry
    # -----------------------------------------------------------------

    @classmethod
    def register(cls, name: str, plugin: "EncryptedSchemePlugin") -> None:
        _REGISTRY[name.lower()] = plugin

    @classmethod
    def get(cls, name: str) -> "EncryptedSchemePlugin":
        key = name.lower()
        if key not in _REGISTRY:
            raise KeyError(
                f"No encrypted search scheme '{name}' is registered. "
                f"Available: {sorted(_REGISTRY)}"
            )
        return _REGISTRY[key]

    @classmethod
    def registered_names(cls) -> list[str]:
        return sorted(_REGISTRY)
```

### 2.2 Built-in plugins: `SSEPlugin` and `StructuredPlugin`

Add `securerag/builtin_schemes.py`. These are drop-in replacements for the current
hard-coded SSE and structured logic. The crypto is identical to what is already in
sim_server and the Rust bridge; it just moves here.

```python
# securerag/builtin_schemes.py
import hashlib
import hmac
import re
import secrets
from typing import Any

from securerag.scheme_plugin import EncryptedSchemePlugin


def _tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for tok in raw:
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        out.append(tok)
    return out


def _encrypt_token(token: str, key: str) -> str:
    return hmac.new(key.encode(), token.encode(), hashlib.sha256).hexdigest()


# ------------------------------------------------------------------
# Plain SSE (token-level HMAC)
# ------------------------------------------------------------------

class SSEPlugin(EncryptedSchemePlugin):
    def generate_key(self) -> str:
        return secrets.token_hex(16)

    def prepare_chunk(self, text: str, key: str) -> dict[str, Any]:
        return {"enc_terms": [_encrypt_token(t, key) for t in _tokenize(text)]}

    def build_server_index(self, rows: list[dict]) -> dict[str, list[int]]:
        """Inverted index: encrypted_token -> list of row indices."""
        inv: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            for term in row.get("scheme_data", {}).get("enc_terms", []):
                inv.setdefault(term, []).append(i)
        return {"rows": rows, "inv": inv}

    def encrypt_query(self, query: str, key: str) -> dict[str, Any]:
        return {"enc_terms": [_encrypt_token(t, key) for t in _tokenize(query)]}

    def search(self, server_index: Any, encrypted_query: dict, top_k: int) -> list[dict]:
        rows = server_index["rows"]
        inv  = server_index["inv"]
        q_terms = encrypted_query.get("enc_terms", [])
        if not q_terms:
            return []
        # Posting list intersection gives candidate set; score = Jaccard
        counts: dict[int, int] = {}
        for term in q_terms:
            for idx in inv.get(term, []):
                counts[idx] = counts.get(idx, 0) + 1
        q_len = len(q_terms)
        scored = []
        for idx, inter in counts.items():
            row = rows[idx]
            doc_terms = row.get("scheme_data", {}).get("enc_terms", [])
            union = q_len + len(doc_terms) - inter
            score = inter / union if union else 0.0
            scored.append({"doc_id": row["doc_id"], "text": row["text"],
                           "metadata": row.get("metadata", {}), "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ------------------------------------------------------------------
# Structured encryption (unigrams + bigrams)
# ------------------------------------------------------------------

class StructuredPlugin(EncryptedSchemePlugin):
    def __init__(self, use_bigrams: bool = True):
        self.use_bigrams = use_bigrams

    def generate_key(self) -> str:
        return secrets.token_hex(16)

    def _make_terms(self, text: str, key: str) -> list[str]:
        tokens = _tokenize(text)
        out = [_encrypt_token(f"tok:{t}", key) for t in tokens]
        if self.use_bigrams and len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                out.append(_encrypt_token(f"bi:{tokens[i]}_{tokens[i+1]}", key))
        return out

    def prepare_chunk(self, text: str, key: str) -> dict[str, Any]:
        return {"struct_terms": self._make_terms(text, key)}

    def build_server_index(self, rows: list[dict]) -> dict:
        inv: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            for term in row.get("scheme_data", {}).get("struct_terms", []):
                inv.setdefault(term, []).append(i)
        return {"rows": rows, "inv": inv}

    def encrypt_query(self, query: str, key: str) -> dict[str, Any]:
        return {"struct_terms": self._make_terms(query, key)}

    def search(self, server_index: Any, encrypted_query: dict, top_k: int) -> list[dict]:
        rows = server_index["rows"]
        inv  = server_index["inv"]
        q_terms = encrypted_query.get("struct_terms", [])
        if not q_terms:
            return []
        counts: dict[int, int] = {}
        for term in q_terms:
            for idx in inv.get(term, []):
                counts[idx] = counts.get(idx, 0) + 1
        q_len = len(q_terms)
        scored = []
        for idx, inter in counts.items():
            row = rows[idx]
            doc_terms = row.get("scheme_data", {}).get("struct_terms", [])
            union = q_len + len(doc_terms) - inter
            score = inter / union if union else 0.0
            scored.append({"doc_id": row["doc_id"], "text": row["text"],
                           "metadata": row.get("metadata", {}), "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# Register built-ins so they're available by name without any import in user code
EncryptedSchemePlugin.register("sse", SSEPlugin())
EncryptedSchemePlugin.register("structured", StructuredPlugin(use_bigrams=True))
EncryptedSchemePlugin.register("structured_encryption", StructuredPlugin(use_bigrams=True))
```

The built-in registration happens at import time. Add the following to
`securerag/__init__.py`:

```python
import securerag.builtin_schemes  # noqa: F401 — registers SSE and structured plugins
```

---

## 3. Changes to `corpus.py`

### 3.1 `CorpusBuilder.build()` — remove scheme dispatch, call plugin

The entire `if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH:` block in `build()`
is replaced with plugin calls. The scheme name string is no longer parsed inside the
builder.

```python
# corpus.py  —  CorpusBuilder.build()  (ENCRYPTED_SEARCH branch only)

if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH:
    plugin = EncryptedSchemePlugin.get(self._encrypted_search_scheme)
    enc_key = plugin.generate_key()

    for chunk in chunks:
        chunk["scheme_data"] = plugin.prepare_chunk(chunk["text"], enc_key)

    extras["enc_key"] = enc_key
    extras["encrypted_search_scheme"] = self._encrypted_search_scheme
    extras["plugin"] = plugin          # kept client-side only, not sent to server
```

The `sse_prepare_chunks` backend RPC call is removed entirely from this path. The
backend no longer needs to know how to encrypt anything; encryption is purely
client-side.

### 3.2 `with_encrypted_search_scheme` — no validation needed

Remove the `elif scheme != "sse": raise ValueError(...)` guard. The plugin registry
raises `KeyError` at `EncryptedSchemePlugin.get()` time if the scheme is unknown, which
gives a cleaner error message.

### 3.3 `SSECorpus` — add `plugin` accessor

```python
class SSECorpus(SecureCorpus):
    @property
    def plugin(self) -> EncryptedSchemePlugin:
        return self.extras["plugin"]    # always set by CorpusBuilder.build()
```

---

## 4. Changes to `retrievers.py`

### 4.1 `EncryptedSearchRetriever.retrieve()` — remove if/elif dispatch

```python
# retrievers.py  —  EncryptedSearchRetriever

class EncryptedSearchRetriever(PrivacyRetriever):
    def retrieve(self, query: str, round_n: int):
        enc_key = self.corpus.extras.get("enc_key")
        if not enc_key:
            raise UnsupportedCapabilityError(
                "ENCRYPTED_SEARCH requires a corpus built with EncryptedSchemePlugin"
            )

        plugin: EncryptedSchemePlugin = self.corpus.extras.get("plugin")
        if plugin is None:
            # Fallback for corpora built before this refactor
            scheme_name = self.corpus.extras.get("encrypted_search_scheme", "sse")
            plugin = EncryptedSchemePlugin.get(scheme_name)

        encrypted_query = plugin.encrypt_query(query, enc_key)

        self._debug(
            "encrypted-search retrieval",
            round_n=round_n,
            scheme=self.corpus.extras.get("encrypted_search_scheme"),
            encrypted_query_keys=list(encrypted_query),
        )

        rows = self._backend.encrypted_search(
            index_id=self.corpus.index_id,
            encrypted_query=encrypted_query,
            top_k=self.config.top_k,
        )
        return self._to_docs(rows)
```

The `sse_encrypt_terms`, `sse_encrypt_structured_terms`, `sse_search`, and
`structured_search` backend calls are gone. A single `encrypted_search` call replaces
all of them.

### 4.2 `UnsupportedCapabilityError` message

Update the error message in `PIRRetriever` to remove the reference to encrypted search
schemes, since the validation logic has moved.

---

## 5. Changes to `backend_client.py`

### 5.1 Remove SSE-specific methods from the abstract base

Remove from `BackendClient`:
- `sse_generate_key()`
- `sse_encrypt_terms()`
- `sse_encrypt_structured_terms()`
- `sse_prepare_chunks()`
- `sse_search()`
- `structured_search()`

Add one new abstract method:

```python
@abstractmethod
def encrypted_search(
    self,
    index_id: str,
    encrypted_query: dict[str, Any],
    top_k: int,
) -> list[dict]:
    """
    Execute a scheme-specific encrypted search on the server.
    `encrypted_query` is the opaque dict returned by plugin.encrypt_query().
    """
    raise NotImplementedError
```

### 5.2 `HttpBackend.encrypted_search`

```python
def encrypted_search(
    self,
    index_id: str,
    encrypted_query: dict[str, Any],
    top_k: int,
) -> list[dict]:
    return self._call(
        "encrypted_search",
        {"index_id": index_id, "encrypted_query": encrypted_query, "top_k": top_k},
    )
```

### 5.3 `RustBackend.encrypted_search`

```python
def encrypted_search(
    self,
    index_id: str,
    encrypted_query: dict[str, Any],
    top_k: int,
) -> list[dict]:
    return self._call(
        "encrypted_search",
        {"index_id": index_id, "encrypted_query": encrypted_query, "top_k": top_k},
    )
```

### 5.4 `GrpcBackend.encrypted_search`

```python
def encrypted_search(
    self,
    index_id: str,
    encrypted_query: dict[str, Any],
    top_k: int,
) -> list[dict]:
    req = self._grpc_pb2.EncryptedSearchRequest(
        index_id=index_id,
        encrypted_query=_dict_to_struct(encrypted_query),
        top_k=top_k,
    )
    resp = self._invoke("EncryptedSearch", req)
    return [self._struct_to_dict(r) for r in resp.rows]
```

### 5.5 Remove `create_backend` routing for deprecated ops

Remove the `HttpBackend` implementations of all six removed operations. The
`RustBackend._call` and `GrpcBackend._invoke` dispatchers are purely data-driven so no
further changes are needed there.

---

## 6. Changes to `sim_server.py`

### 6.1 Remove deprecated RPC handlers

Remove the handlers for:
- `sse_generate_key`
- `sse_encrypt_terms`
- `sse_encrypt_structured_terms`
- `sse_prepare_chunks`
- `sse_search`
- `structured_search`

Also remove the helper functions `_encrypt_terms`, `_encrypt_structured_terms`, and
`_encrypt_token` (they move to `builtin_schemes.py`).

### 6.2 `build_index`: store `scheme_data` per chunk; call plugin's `build_server_index`

The sim_server is a Python process, so it can import and call the plugin directly.
This gives exact parity with what a Rust production backend would do, without requiring
a second implementation of any scheme.

```python
# sim_server.py  —  op == "build_index"

if op == "build_index":
    protocol  = p["protocol"]
    epsilon   = float(p.get("epsilon", 1_000_000.0))
    delta     = float(p.get("delta",   1e-5))
    chunks    = p["chunks"]
    scheme    = p.get("encrypted_search_scheme")   # None for non-SSE protocols

    rows = []
    for c in chunks:
        row = {
            "doc_id":    c["doc_id"],
            "text":      c["text"],
            "metadata":  c.get("metadata", {}),
            "embedding": _embed(c["text"]),
            "scheme_data": c.get("scheme_data", {}),  # pre-encrypted by client
        }
        rows.append(row)

    # Let the plugin build its server-side index structure (e.g. inverted index)
    if scheme:
        try:
            from securerag.scheme_plugin import EncryptedSchemePlugin
            plugin = EncryptedSchemePlugin.get(scheme)
            server_index = plugin.build_server_index(rows)
        except KeyError:
            server_index = rows   # unknown scheme: fall back to linear scan
    else:
        server_index = rows

    index_id = str(uuid.uuid4())
    _INDEXES[index_id] = {
        "protocol":     protocol,
        "server_index": server_index,
        "rows":         rows,          # kept for embed-based search by other protocols
        "scheme":       scheme,
        "epsilon":      epsilon,
        "delta":        delta,
        "rdp_acc":      [0.0] * 5,
    }
    return {"ok": True, "data": {"index_id": index_id, "doc_count": len(rows)}}
```

### 6.3 Add `encrypted_search` handler

```python
if op == "encrypted_search":
    index          = _INDEXES[p["index_id"]]
    encrypted_query = p["encrypted_query"]
    top_k          = int(p["top_k"])
    scheme         = index.get("scheme")

    if scheme:
        from securerag.scheme_plugin import EncryptedSchemePlugin
        plugin  = EncryptedSchemePlugin.get(scheme)
        results = plugin.search(index["server_index"], encrypted_query, top_k)
    else:
        return {"ok": False, "error": "Index was not built with an encrypted scheme"}

    return {"ok": True, "data": results}
```

### 6.4 Update `build_index` to accept `encrypted_search_scheme`

The `build_index` call from `HttpBackend` now needs to carry the scheme name so
sim_server knows which plugin to invoke for `build_server_index`. Add it to the
payload in `corpus.py` and read it in `build_index` above.

---

## 7. Changes to `pyo3_bridge.rs`

The Rust PyO3 bridge is used for in-process high-speed retrieval. For encrypted search,
the simplest and most correct approach is to delegate search back to the Python plugin
via a stored callable. This avoids duplicating any scheme logic in Rust.

### 7.1 `IndexState`: replace `enc_terms`/`struct_terms` with `scheme_data` + Python search callable

```rust
// pyo3_bridge.rs

use pyo3::types::PyAny;

struct Row {
    doc_id:      String,
    text:        String,
    embedding:   Vec<f64>,
    scheme_data: Py<PyDict>,   // opaque, passed back to Python plugin.search()
}

struct IndexState {
    protocol:     PrivacyProtocol,
    rows:         Vec<Row>,
    dp_budget:    Option<Arc<Mutex<RDPAccountant>>>,
    // For encrypted search: the Python plugin object is stored and called at query time.
    // None for all non-ENCRYPTED_SEARCH protocols.
    enc_plugin:   Option<Py<PyAny>>,
    server_index: Option<Py<PyAny>>,   // return value of plugin.build_server_index(rows)
}
```

### 7.2 `build_index` handler: call `plugin.build_server_index` from Rust

```rust
// pyo3_bridge.rs  —  "build_index" handler (ENCRYPTED_SEARCH branch)

let (enc_plugin, server_index) = if protocol == PrivacyProtocol::EncryptedSearch {
    let scheme: String = payload
        .get_item("encrypted_search_scheme")
        .ok().flatten()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_else(|| "sse".to_string());

    // Import the Python plugin registry and retrieve the plugin
    let plugin_mod = py.import("securerag.scheme_plugin")?;
    let plugin = plugin_mod
        .getattr("EncryptedSchemePlugin")?
        .call_method1("get", (&scheme,))?;

    // Convert rows to Python list of dicts for build_server_index
    let py_rows = PyList::empty(py);
    for row in &rows {
        let d = PyDict::new(py);
        d.set_item("doc_id",      &row.doc_id)?;
        d.set_item("text",        &row.text)?;
        d.set_item("scheme_data", row.scheme_data.bind(py))?;
        py_rows.append(d)?;
    }
    let srv_idx = plugin.call_method1("build_server_index", (py_rows,))?;

    (Some(plugin.into()), Some(srv_idx.into()))
} else {
    (None, None)
};

guard.insert(index_id.clone(), IndexState {
    protocol,
    rows,
    dp_budget,
    enc_plugin,
    server_index,
});
```

### 7.3 Add `encrypted_search` handler

```rust
// pyo3_bridge.rs  —  "encrypted_search" handler

"encrypted_search" => {
    let index_id: String = required_item(&payload, "index_id")?.extract()?;
    let top_k: usize     = required_item(&payload, "top_k")?.extract()?;
    let enc_query        = required_item(&payload, "encrypted_query")?;

    let guard = INDEXES.lock()
        .map_err(|_| PyRuntimeError::new_err("index lock poisoned"))?;
    let state = guard
        .get(&index_id)
        .ok_or_else(|| PyRuntimeError::new_err("index not found"))?;

    let plugin = state.enc_plugin.as_ref()
        .ok_or_else(|| PyRuntimeError::new_err(
            "index was not built with an encrypted scheme"
        ))?;
    let srv_idx = state.server_index.as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("server index not initialised"))?;

    // Delegate entirely to the Python plugin
    let results = plugin
        .bind(py)
        .call_method1("search", (srv_idx.bind(py), enc_query, top_k as i64))?;

    Ok(results.into_any().unbind())
}
```

### 7.4 Remove deprecated handlers

Remove from `pyo3_bridge.rs`:
- `"sse_generate_key"`
- `"sse_encrypt_terms"`
- `"sse_encrypt_structured_terms"`
- `"sse_prepare_chunks"`
- `"sse_search"`
- `"structured_search"`

Remove the corresponding `generate_sse_key`, `encrypt_terms`, `encrypt_structured_terms`
functions. Remove the `enc_terms` and `struct_terms` fields from `Row`.

---

## 8. Changes to `securerag_grpc_server.rs`

The gRPC server is the production-speed path. Unlike the PyO3 bridge, it cannot easily
call back into Python. Scheme implementations for production use must be in Rust. The
pattern is identical to the plugin pattern on the Python side: a trait.

### 8.1 New trait `EncryptedScheme`

```rust
// In a new file: securerag-rs/src/encrypted_scheme.rs

use std::collections::HashMap;

pub struct SearchResult {
    pub doc_id:   String,
    pub text:     String,
    pub score:    f64,
}

pub trait EncryptedScheme: Send + Sync {
    /// Post-process prepared rows into an efficient server-side index.
    /// `rows[i].scheme_data` contains the dict returned by the client's `prepare_chunk`.
    fn build_index(&self, rows: &[SchemeRow]) -> Box<dyn SchemeIndex>;
}

pub trait SchemeIndex: Send + Sync {
    fn search(
        &self,
        encrypted_query: &HashMap<String, serde_json::Value>,
        top_k: usize,
    ) -> Vec<SearchResult>;
}

pub struct SchemeRow {
    pub doc_id:      String,
    pub text:        String,
    pub scheme_data: HashMap<String, serde_json::Value>,
}
```

### 8.2 Built-in Rust implementations

Add `securerag-rs/src/builtin_schemes.rs` with `RustSSEScheme` and
`RustStructuredScheme`. These mirror the Python plugins using the same inverted-index
approach. The crypto (HMAC-SHA256 of token + key) uses the `hmac` and `sha2` crates and
must produce identical output to the Python `_encrypt_token` function so that a corpus
prepared client-side in Python can be searched server-side in Rust.

> **Parity contract:** `HMAC-SHA256(token.encode("utf-8"), key.encode("utf-8")).hexdigest()`
> must match `hex(hmac::Hmac::<Sha256>::new_from_slice(key.as_bytes())?.chain_update(token.as_bytes()).finalize().into_bytes())`.
> Add a unit test asserting exact hex output for a fixed token and key.

### 8.3 Scheme registry in the gRPC server

```rust
// securerag_grpc_server.rs

use once_cell::sync::Lazy;
use std::collections::HashMap;

static SCHEME_REGISTRY: Lazy<HashMap<&'static str, Box<dyn EncryptedScheme>>> =
    Lazy::new(|| {
        let mut m: HashMap<&str, Box<dyn EncryptedScheme>> = HashMap::new();
        m.insert("sse",          Box::new(RustSSEScheme));
        m.insert("structured",   Box::new(RustStructuredScheme { use_bigrams: true }));
        m
    });
```

### 8.4 Update `Index` struct and `build_index` handler

```rust
struct Index {
    _protocol:    String,
    dp_budget:    Option<RDPAccountant>,
    chunks:       Vec<Row>,      // retained for embed-based search
    scheme_index: Option<Box<dyn SchemeIndex>>,
}
```

In `build_index`:

```rust
let scheme_index = if req.protocol == "EncryptedSearch" {
    let scheme_name = req.encrypted_search_scheme.to_ascii_lowercase();
    let scheme = SCHEME_REGISTRY.get(scheme_name.as_str())
        .ok_or_else(|| Status::unimplemented(
            format!("encrypted scheme '{}' not registered in Rust server", scheme_name)
        ))?;
    let scheme_rows: Vec<SchemeRow> = rows.iter().map(|r| SchemeRow {
        doc_id:      r.doc_id.clone(),
        text:        r.text.clone(),
        scheme_data: r.scheme_data.clone(),
    }).collect();
    Some(scheme.build_index(&scheme_rows))
} else {
    None
};
```

### 8.5 Add `EncryptedSearch` RPC handler, remove deprecated handlers

```rust
async fn encrypted_search(
    &self,
    request: Request<EncryptedSearchRequest>,
) -> Result<Response<EncryptedSearchResponse>, Status> {
    let req   = request.into_inner();
    let top_k = req.top_k.max(0) as usize;
    let guard = INDEXES.lock().map_err(|_| Status::internal("lock poisoned"))?;
    let index = guard.get(&req.index_id)
        .ok_or_else(|| Status::not_found("index not found"))?;
    let scheme_index = index.scheme_index.as_ref()
        .ok_or_else(|| Status::failed_precondition(
            "index was not built with an encrypted scheme"
        ))?;

    let encrypted_query: HashMap<String, serde_json::Value> =
        struct_to_map(&req.encrypted_query)?;

    let results = scheme_index.search(&encrypted_query, top_k);
    let rows = results.into_iter()
        .map(|r| row_to_struct(&r.doc_id, &r.text, BTreeMap::new(), r.score))
        .collect();
    Ok(Response::new(EncryptedSearchResponse { rows }))
}
```

Remove handlers for `sse_generate_key`, `sse_encrypt_terms`, `sse_encrypt_structured_terms`,
`sse_prepare_chunks`, `sse_search`, `structured_search`.

---

## 9. Changes to `secure_retrieval.proto`

### 9.1 Remove deprecated messages and RPCs

Remove:
```protobuf
rpc SseGenerateKey (SseGenerateKeyRequest) returns (SseGenerateKeyResponse);
rpc SseEncryptTerms (SseEncryptTermsRequest) returns (SseEncryptTermsResponse);
rpc SseEncryptStructuredTerms (...) returns (...);
rpc SsePrepareChunks (SsePrepareChunksRequest) returns (SsePrepareChunksResponse);
rpc SseSearch (SseSearchRequest) returns (SseSearchResponse);
rpc StructuredSearch (StructuredSearchRequest) returns (StructuredSearchResponse);
```

And all their associated message definitions.

### 9.2 Add `EncryptedSearch` RPC and messages

```protobuf
rpc EncryptedSearch (EncryptedSearchRequest) returns (EncryptedSearchResponse);

message EncryptedSearchRequest {
  string index_id                       = 1;
  google.protobuf.Struct encrypted_query = 2;   // opaque plugin-defined dict
  int32  top_k                          = 3;
}

message EncryptedSearchResponse {
  repeated google.protobuf.Struct rows = 1;
}
```

### 9.3 Update `BuildIndexRequest`

Add the scheme name so the server knows which plugin/Rust scheme to use:

```protobuf
message BuildIndexRequest {
  string protocol                      = 1;
  repeated google.protobuf.Struct chunks = 2;
  double epsilon                       = 3;
  double delta                         = 4;
  string encrypted_search_scheme       = 5;   // empty string for non-SSE protocols
}
```

Regenerate `secure_retrieval_pb2.py` after these changes.

---

## 10. Benchmark Support: Batch Corpus Preparation

The `CorpusBuilder.build()` currently calls `plugin.prepare_chunk()` once per document
in Python, then makes a single bulk `build_index` RPC call. For large corpora this is
acceptable because all work stays in Python. However, the `chunk()` and `sanitize()`
backend calls iterate documents serially over HTTP. Add a `batch_build` mode that keeps
all preprocessing local:

```python
# corpus.py  —  CorpusBuilder

def build_local(self, *, workers: int = 4) -> SecureCorpus:
    """
    Like build(), but chunking and sanitization run locally in Python
    using concurrent.futures rather than via backend RPC.  Suitable for
    benchmarking large corpora where RPC overhead would dominate.
    Requires: pip install nltk  (or any local chunker you prefer)
    """
    from concurrent.futures import ThreadPoolExecutor
    import itertools

    # Local chunking
    chunks = _local_chunk(self._docs, self._chunk_size, self._overlap)

    if self._sanitize:
        chunks = _local_sanitize(chunks)

    if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH:
        plugin  = EncryptedSchemePlugin.get(self._encrypted_search_scheme)
        enc_key = plugin.generate_key()

        def _prep(chunk):
            chunk["scheme_data"] = plugin.prepare_chunk(chunk["text"], enc_key)
            return chunk

        with ThreadPoolExecutor(max_workers=workers) as ex:
            chunks = list(ex.map(_prep, chunks))

    # Single bulk build_index call — only one RPC regardless of corpus size
    index_payload = self._backend.build_index(
        self._protocol.wire_name,
        chunks,
        epsilon=self._epsilon,
        delta=self._delta,
        encrypted_search_scheme=self._encrypted_search_scheme
            if self._protocol is PrivacyProtocol.ENCRYPTED_SEARCH else "",
    )
    # ... (same return logic as build())
```

---

## 11. Researcher Usage After This Refactor

A researcher implementing a new encrypted search scheme (e.g., a forward-private scheme
or one with access pattern obfuscation) only needs to write one class, registered once:

```python
# my_scheme.py  —  entirely outside the securerag package

from securerag.scheme_plugin import EncryptedSchemePlugin
from typing import Any

class MyNewScheme(EncryptedSchemePlugin):
    def generate_key(self) -> bytes:
        import os
        return os.urandom(32)

    def prepare_chunk(self, text: str, key: bytes) -> dict[str, Any]:
        # ... scheme-specific corpus preparation
        return {"my_field": ...}

    def build_server_index(self, rows: list[dict]) -> Any:
        # Build whatever structure enables fast search
        return my_fast_index(rows)

    def encrypt_query(self, query: str, key: bytes) -> dict[str, Any]:
        return {"my_query_token": ...}

    def search(self, server_index: Any, encrypted_query: dict, top_k: int) -> list[dict]:
        return my_search(server_index, encrypted_query, top_k)

# Register once, at the top of the experiment script
EncryptedSchemePlugin.register("my_scheme", MyNewScheme())
```

Then use it identically to the built-in schemes:

```python
corpus = (
    CorpusBuilder(PrivacyProtocol.ENCRYPTED_SEARCH)
    .with_encrypted_search_scheme("my_scheme")
    .with_privacy_budget(epsilon=1.0, delta=1e-5)
    .add_documents(docs)
    .build()
)

cfg = PrivacyConfig(
    protocol=PrivacyProtocol.ENCRYPTED_SEARCH,
    encrypted_search_scheme="my_scheme",
    top_k=10,
)
agent = SecureRAGAgent.from_config(cfg, corpus=corpus, llm=OllamaLLM())
result = agent.run("query text")
```

No framework files are modified. The scheme runs at full Python speed in the sim_server
and PyO3 backends (which call back into the plugin). For the gRPC production backend,
a Rust implementation of the `EncryptedScheme` trait must also be provided and
registered in `SCHEME_REGISTRY`, but the Python prototype and benchmarks can proceed
entirely without it.

---

## 12. Change Surface Summary

| File | Action | Reason |
|------|--------|--------|
| `securerag/scheme_plugin.py` | **New** | Plugin base class and registry |
| `securerag/builtin_schemes.py` | **New** | `SSEPlugin`, `StructuredPlugin` with inverted indexes |
| `securerag/__init__.py` | Add 1 import | Auto-register built-in schemes |
| `securerag/corpus.py` | Replace SSE branch in `build()` | Call plugin instead of backend RPC |
| `securerag/retrievers.py` | Replace `EncryptedSearchRetriever.retrieve()` | Single `encrypted_search` call |
| `securerag/backend_client.py` | Remove 6 methods, add 1 | `encrypted_search` replaces all SSE ops |
| `securerag/sim_server.py` | Remove 6 handlers + helpers, add 2 | `build_index` stores `scheme_data`, new `encrypted_search` handler |
| `securerag-rs/src/pyo3_bridge.rs` | Remove 6 handlers, add 1, update `Row`/`IndexState` | Delegate to Python plugin at search time |
| `securerag-rs/src/encrypted_scheme.rs` | **New** | `EncryptedScheme` + `SchemeIndex` traits |
| `securerag-rs/src/builtin_schemes.rs` | **New** | `RustSSEScheme`, `RustStructuredScheme` |
| `securerag-rs/src/bin/securerag_grpc_server.rs` | Remove 6 handlers, add 1, update `Index` | Rust scheme registry, `EncryptedSearch` RPC |
| `securerag/proto/secure_retrieval.proto` | Remove 6 RPCs/messages, add 1 | Wire protocol matches new surface |
| `securerag-rs/proto/secure_retrieval.proto` | Same changes | Must stay in sync |
| `securerag/proto/secure_retrieval_pb2.py` | Regenerate | Reflects proto changes |

**Lines removed:** ~280 (six RPC handlers × three backends + six proto message blocks
+ six abstract method signatures + if/elif dispatch in corpus and retriever)

**Lines added:** ~350 (plugin base class, two built-in plugins, inverted index
implementations, three new single-op handlers, Rust trait + two implementations)

The net result is a smaller total codebase, zero scheme-specific code in framework core,
and O(matching_docs) query performance for all inverted-index-based schemes.