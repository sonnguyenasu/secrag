use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use sha2::{Digest, Sha256};

use crate::builders::{EmbeddingIndexBuilder, PlainIndexBuilder, UnsupportedBuilder};
use crate::core::CorpusProcessor;
use crate::core::RetrieverCore;
use crate::dp::RDPAccountant;
use crate::engines::BaselineEngine;
use crate::protocol::PrivacyProtocol;
use crate::types::{Chunk, IndexPayload};

const LEXICAL_WEIGHT: f64 = 0.65;
const EMBEDDING_WEIGHT: f64 = 0.35;

struct Row {
    doc_id: String,
    text: String,
    embedding: Vec<f64>,
    scheme_data: Py<PyDict>,
}

struct IndexState {
    protocol: PrivacyProtocol,
    rows: Vec<Row>,
    dp_budget: Option<Arc<Mutex<RDPAccountant>>>,
    enc_plugin: Option<Py<PyAny>>,
    server_index: Option<Py<PyAny>>,
}

static INDEXES: Lazy<Mutex<HashMap<String, IndexState>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn required_item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyRuntimeError::new_err(format!("missing {}", key)))
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut t = s.to_ascii_lowercase();
            if t.len() > 3 && t.ends_with('s') {
                t.pop();
            }
            t
        })
        .collect()
}

fn embed(text: &str, dim: usize) -> Vec<f64> {
    let mut vec = vec![0.0; dim];
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return vec;
    }
    for tok in tokens {
        let h = Sha256::digest(tok.as_bytes());
        let idx = (h[31] as usize) % dim;
        let sign = if (h[30] & 1) == 0 { 1.0 } else { -1.0 };
        vec[idx] += sign;
    }
    let norm = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut vec {
            *x /= norm;
        }
    }
    vec
}

fn cos(a: &[f64], b: &[f64]) -> f64 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn lexical_jaccard(query: &str, text: &str) -> f64 {
    let qset: std::collections::HashSet<String> = tokenize(query).into_iter().collect();
    let tset: std::collections::HashSet<String> = tokenize(text).into_iter().collect();
    if qset.is_empty() || tset.is_empty() {
        return 0.0;
    }
    let inter = qset.intersection(&tset).count() as f64;
    let union = qset.union(&tset).count() as f64;
    if union == 0.0 { 0.0 } else { inter / union }
}

#[pyclass]
pub struct BackendBridge;

#[pymethods]
impl BackendBridge {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn rpc(
        &self,
        py: Python<'_>,
        op: &str,
        payload: Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {

        match op {
            "chunk" => {
                let docs_any = required_item(&payload, "docs")?;
                let chunk_size: usize = payload
                    .get_item("chunk_size")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<usize>().ok())
                    .unwrap_or(512);
                let overlap: usize = payload
                    .get_item("overlap")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<usize>().ok())
                    .unwrap_or(64);
                let step = if chunk_size > overlap {
                    chunk_size - overlap
                } else {
                    1
                };

                let out = PyList::empty(py);
                for item in docs_any.downcast::<PyList>()?.iter() {
                    let d = item.downcast::<PyDict>()?;
                    let doc_id: String = d
                        .get_item("doc_id")
                        ?
                        .ok_or_else(|| PyRuntimeError::new_err("missing doc_id"))?
                        .extract()?;
                    let text: String = d
                        .get_item("text")
                        ?
                        .ok_or_else(|| PyRuntimeError::new_err("missing text"))?
                        .extract()?;

                    if text.len() <= chunk_size {
                        let row = PyDict::new(py);
                        row.set_item("doc_id", doc_id.clone())?;
                        row.set_item("text", text.clone())?;
                        row.set_item("metadata", PyDict::new(py))?;
                        out.append(row)?;
                        continue;
                    }

                    let mut i = 0usize;
                    while i < text.len() {
                        let end = usize::min(i + chunk_size, text.len());
                        let snippet = &text[i..end];
                        if !snippet.is_empty() {
                            if i == 0 || snippet.len() >= usize::max(24, chunk_size / 3) {
                                let row = PyDict::new(py);
                                row.set_item("doc_id", doc_id.clone())?;
                                row.set_item("text", snippet)?;
                                row.set_item("metadata", PyDict::new(py))?;
                                out.append(row)?;
                            }
                        }
                        i += step;
                    }
                }
                Ok(out.into_any().unbind())
            }
            "sanitize" => {
                let chunks_any = required_item(&payload, "chunks")?;
                Ok(chunks_any.into_any().unbind())
            }
            "encrypted_search" => {
                let index_id: String = required_item(&payload, "index_id")?.extract()?;
                let top_k: usize = required_item(&payload, "top_k")?.extract()?;
                let enc_query = required_item(&payload, "encrypted_query")?;

                let (plugin, srv_idx) = {
                    let guard = INDEXES
                        .lock()
                        .map_err(|_| PyRuntimeError::new_err("index lock poisoned"))?;
                    let state = guard
                        .get(&index_id)
                        .ok_or_else(|| PyRuntimeError::new_err("index not found"))?;
                    let plugin = state.enc_plugin.as_ref().ok_or_else(|| {
                        PyRuntimeError::new_err("index was not built with an encrypted scheme")
                    })?;
                    let srv_idx = state
                        .server_index
                        .as_ref()
                        .ok_or_else(|| PyRuntimeError::new_err("server index not initialised"))?;
                    (plugin.clone_ref(py), srv_idx.clone_ref(py))
                };

                let results = plugin
                    .bind(py)
                    .call_method1("search", (srv_idx.bind(py), enc_query, top_k as i64))?;
                Ok(results.into_any().unbind())
            }
            "build_index" => {
                let protocol_s: String = required_item(&payload, "protocol")?.extract()?;
                let epsilon: f64 = payload
                    .get_item("epsilon")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<f64>().ok())
                    .unwrap_or(1_000_000.0);
                let delta: f64 = payload
                    .get_item("delta")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<f64>().ok())
                    .unwrap_or(1e-5);
                let protocol = PrivacyProtocol::from_wire(&protocol_s)
                    .ok_or_else(|| PyRuntimeError::new_err("invalid protocol"))?;
                let chunks_any = required_item(&payload, "chunks")?;
                let chunks_list = chunks_any.downcast::<PyList>()?;

                let mut rows: Vec<Row> = Vec::new();
                let mut core_chunks: Vec<Chunk> = Vec::new();
                for item in chunks_list.iter() {
                    let row = item.downcast::<PyDict>()?;
                    let doc_id: String = row
                        .get_item("doc_id")?
                        .ok_or_else(|| PyRuntimeError::new_err("missing doc_id"))?
                        .extract()?;
                    let text: String = row
                        .get_item("text")?
                        .ok_or_else(|| PyRuntimeError::new_err("missing text"))?
                        .extract()?;
                    let scheme_data_dict = PyDict::new(py);
                    if let Some(v) = row.get_item("scheme_data")? {
                        if let Ok(d) = v.downcast::<PyDict>() {
                            for (k, val) in d.iter() {
                                scheme_data_dict.set_item(k, val)?;
                            }
                        }
                    }
                    let scheme_data = scheme_data_dict.unbind();
                    core_chunks.push(Chunk {
                        doc_id: doc_id.clone(),
                        text: text.clone(),
                    });
                    rows.push(Row {
                        doc_id,
                        text,
                        embedding: Vec::new(),
                        scheme_data,
                    });
                }

                for row in &mut rows {
                    row.embedding = embed(&row.text, 64);
                }

                let processor: Box<dyn CorpusProcessor> = match protocol {
                    PrivacyProtocol::Baseline | PrivacyProtocol::Obfuscation | PrivacyProtocol::EncryptedSearch => {
                        Box::new(PlainIndexBuilder)
                    }
                    PrivacyProtocol::DiffPrivacy => Box::new(EmbeddingIndexBuilder),
                    PrivacyProtocol::Pir => Box::new(UnsupportedBuilder {
                        protocol_value: PrivacyProtocol::Pir,
                    }),
                };
                let processed = processor.build(core_chunks);
                let _processed_count = match processed {
                    IndexPayload::Chunks(v) => v.len(),
                };

                let index_id = format!(
                    "rs-{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .as_nanos()
                );

                let dp_budget = if protocol == PrivacyProtocol::DiffPrivacy {
                    Some(Arc::new(Mutex::new(RDPAccountant::new(epsilon, delta))))
                } else {
                    None
                };

                let (enc_plugin, server_index) = if protocol == PrivacyProtocol::EncryptedSearch {
                    let scheme: String = payload
                        .get_item("encrypted_search_scheme")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<String>().ok())
                        .unwrap_or_else(|| "sse".to_string());

                    let plugin_mod = py.import("securerag.scheme_plugin")?;
                    let plugin = plugin_mod
                        .getattr("EncryptedSchemePlugin")?
                        .call_method1("get", (&scheme,))?;

                    let py_rows = PyList::empty(py);
                    for row in &rows {
                        let d = PyDict::new(py);
                        d.set_item("doc_id", &row.doc_id)?;
                        d.set_item("text", &row.text)?;
                        d.set_item("scheme_data", row.scheme_data.bind(py))?;
                        py_rows.append(d)?;
                    }
                    let srv_idx = plugin.call_method1("build_server_index", (py_rows,))?;
                    (Some(plugin.into()), Some(srv_idx.into()))
                } else {
                    (None, None)
                };

                let mut guard = INDEXES
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("index lock poisoned"))?;

                guard.insert(
                    index_id.clone(),
                    IndexState {
                        protocol,
                        rows,
                        dp_budget,
                        enc_plugin,
                        server_index,
                    },
                );

                let out = PyDict::new(py);
                out.set_item("index_id", index_id)?;
                out.set_item("doc_count", chunks_list.len())?;
                Ok(out.into_any().unbind())
            }
            "batch_retrieve" => {
                let index_id: String = required_item(&payload, "index_id")?.extract()?;
                let top_k: usize = required_item(&payload, "top_k")?.extract()?;
                let queries_any = required_item(&payload, "queries")?;
                let queries = queries_any.downcast::<PyList>()?;

                let guard = INDEXES
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("index lock poisoned"))?;
                let state = guard
                    .get(&index_id)
                    .ok_or_else(|| PyRuntimeError::new_err("index not found"))?;

                let engine_chunks: Vec<Chunk> = state
                    .rows
                    .iter()
                    .map(|r| Chunk {
                        doc_id: r.doc_id.clone(),
                        text: r.text.clone(),
                    })
                    .collect();
                let engine = BaselineEngine {
                    chunks: std::sync::Arc::new(engine_chunks),
                };

                let all = PyList::empty(py);
                for q in queries.iter() {
                    let query: String = q.extract()?;
                    let out_q = PyList::empty(py);
                    for doc in engine.retrieve(&query, top_k) {
                        let row = PyDict::new(py);
                        row.set_item("doc_id", doc.doc_id)?;
                        row.set_item("text", doc.text)?;
                        row.set_item("metadata", PyDict::new(py))?;
                        row.set_item("score", doc.score)?;
                        out_q.append(row)?;
                    }
                    all.append(out_q)?;
                }
                Ok(all.into_any().unbind())
            }
            "generate_decoys" => {
                let index_id: String = required_item(&payload, "index_id")?.extract()?;
                let query: String = required_item(&payload, "query")?.extract()?;
                let k: usize = required_item(&payload, "k")?.extract()?;

                let guard = INDEXES
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("index lock poisoned"))?;
                let rows = &guard
                    .get(&index_id)
                    .ok_or_else(|| PyRuntimeError::new_err("index not found"))?
                    .rows;

                let out = PyList::empty(py);
                if rows.is_empty() {
                    for _ in 0..k {
                        out.append(format!("{} decoy", query))?;
                    }
                    return Ok(out.into_any().unbind());
                }

                // Deterministic decoy selection from corpus chunks to mimic obfuscation traffic.
                let seed = query
                    .bytes()
                    .fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(b as u64));
                for i in 0..k {
                    let idx = ((seed as usize).wrapping_add(i * 17)) % rows.len();
                    let text = rows[idx].text.clone();
                    out.append(text)?;
                }
                Ok(out.into_any().unbind())
            }
            "embed_with_noise" => {
                let query: String = required_item(&payload, "query")?.extract()?;
                let sigma: f64 = required_item(&payload, "sigma")?.extract()?;
                let base = embed(&query, 64);
                let digest = Sha256::digest(query.as_bytes());
                let seed = u64::from_le_bytes(
                    digest[..8]
                        .try_into()
                        .map_err(|_| PyRuntimeError::new_err("invalid seed bytes"))?,
                );
                let mut rng = StdRng::seed_from_u64(seed);
                let normal = Normal::new(0.0, sigma)
                    .map_err(|e| PyRuntimeError::new_err(format!("invalid sigma: {e}")))?;
                let out: Vec<f64> = base
                    .iter()
                    .map(|v| *v + normal.sample(&mut rng))
                    .collect();
                Ok(PyList::new(py, out)?.into_any().unbind())
            }
            "retrieve_by_embedding" => {
                let index_id: String = required_item(&payload, "index_id")?.extract()?;
                let top_k: usize = required_item(&payload, "top_k")?.extract()?;
                let embedding: Vec<f64> = required_item(&payload, "embedding")?.extract()?;
                let sigma: f64 = payload
                    .get_item("sigma")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<f64>().ok())
                    .unwrap_or(1.0);
                let query = payload
                    .get_item("query")
                    .ok()
                    .flatten()
                    .and_then(|q| q.extract::<String>().ok())
                    .unwrap_or_default()
                    .to_lowercase();

                let guard = INDEXES
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("index lock poisoned"))?;
                let state = guard
                    .get(&index_id)
                    .ok_or_else(|| PyRuntimeError::new_err("index not found"))?;

                if state.protocol == PrivacyProtocol::DiffPrivacy {
                    if let Some(budget) = &state.dp_budget {
                        let mut b = budget
                            .lock()
                            .map_err(|_| PyRuntimeError::new_err("budget lock poisoned"))?;
                        b.consume_rdp(sigma).map_err(PyRuntimeError::new_err)?;
                    }
                }

                let mut scored: Vec<(f64, String, String)> = state
                    .rows
                    .iter()
                    .map(|r| {
                        let emb_score = cos(&embedding, &r.embedding);
                        let score = if query.is_empty() {
                            emb_score
                        } else {
                            let lex_score = lexical_jaccard(&query, &r.text);
                            LEXICAL_WEIGHT * lex_score + EMBEDDING_WEIGHT * emb_score
                        };
                        (score, r.doc_id.clone(), r.text.clone())
                    })
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                let out = PyList::empty(py);
                for (score, doc_id, text) in scored.into_iter().take(top_k) {
                    let row = PyDict::new(py);
                    row.set_item("doc_id", doc_id)?;
                    row.set_item("text", text)?;
                    row.set_item("metadata", PyDict::new(py))?;
                    row.set_item("score", score)?;
                    out.append(row)?;
                }
                Ok(out.into_any().unbind())
            }
            _ => Err(PyRuntimeError::new_err(format!("unsupported operation: {}", op))),
        }
    }
}
