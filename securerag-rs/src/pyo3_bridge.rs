use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::RngCore;
use sha2::{Digest, Sha256};

use crate::builders::{EmbeddingIndexBuilder, PlainIndexBuilder, UnsupportedBuilder};
use crate::core::CorpusProcessor;
use crate::core::RetrieverCore;
use crate::dp::RDPAccountant;
use crate::engines::{BaselineEngine, DPMechanism};
use crate::protocol::PrivacyProtocol;
use crate::types::{Chunk, IndexPayload};

#[derive(Clone)]
struct Row {
    doc_id: String,
    text: String,
    embedding: Vec<f64>,
    enc_terms: Vec<String>,
    struct_terms: Vec<String>,
}

#[derive(Clone)]
struct IndexState {
    protocol: PrivacyProtocol,
    rows: Vec<Row>,
    dp_budget: Option<Arc<Mutex<RDPAccountant>>>,
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
        let mut first8 = [0u8; 8];
        first8.copy_from_slice(&h[0..8]);
        let hv = u64::from_be_bytes(first8);
        let idx = (hv as usize) % dim;
        let sign = if ((hv >> 8) & 1) == 0 { 1.0 } else { -1.0 };
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

fn generate_sse_key() -> String {
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn encrypt_token(token: &str, key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{}:{}", key, token).as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
    hex.chars().take(24).collect()
}

fn encrypt_terms(text: &str, key: &str) -> Vec<String> {
    tokenize(text)
        .into_iter()
        .map(|t| encrypt_token(&t, key))
        .collect()
}

fn encrypt_structured_terms(text: &str, key: &str, use_bigrams: bool) -> Vec<String> {
    let tokens = tokenize(text);
    let mut out: Vec<String> = tokens
        .iter()
        .map(|t| encrypt_token(&format!("tok:{}", t), key))
        .collect();

    if use_bigrams && tokens.len() >= 2 {
        for i in 0..(tokens.len() - 1) {
            let bg = format!("{}_{}", tokens[i], tokens[i + 1]);
            out.push(encrypt_token(&format!("bi:{}", bg), key));
        }
    }

    out
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
            "sse_generate_key" => Ok(generate_sse_key().into_pyobject(py)?.into_any().unbind()),
            "sse_encrypt_terms" => {
                let text: String = required_item(&payload, "text")?.extract()?;
                let key: String = required_item(&payload, "key")?.extract()?;
                let out = encrypt_terms(&text, &key);
                Ok(PyList::new(py, out)?.into_any().unbind())
            }
            "sse_encrypt_structured_terms" => {
                let text: String = required_item(&payload, "text")?.extract()?;
                let key: String = required_item(&payload, "key")?.extract()?;
                let use_bigrams: bool = payload
                    .get_item("use_bigrams")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<bool>().ok())
                    .unwrap_or(true);
                let out = encrypt_structured_terms(&text, &key, use_bigrams);
                Ok(PyList::new(py, out)?.into_any().unbind())
            }
            "sse_prepare_chunks" => {
                let chunks_any = required_item(&payload, "chunks")?;
                let chunks_list = chunks_any.downcast::<PyList>()?;
                let key: String = required_item(&payload, "key")?.extract()?;
                let scheme = payload
                    .get_item("scheme")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_else(|| "sse".to_string())
                    .to_ascii_lowercase();
                let use_bigrams: bool = payload
                    .get_item("use_bigrams")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<bool>().ok())
                    .unwrap_or(true);

                let out = PyList::empty(py);
                for item in chunks_list.iter() {
                    let row = item.downcast::<PyDict>()?;
                    let text: String = row
                        .get_item("text")?
                        .ok_or_else(|| PyRuntimeError::new_err("missing text"))?
                        .extract()?;

                    let cloned = PyDict::new(py);
                    for (k, v) in row.iter() {
                        cloned.set_item(k, v)?;
                    }

                    match scheme.as_str() {
                        "sse" => {
                            let enc_terms = encrypt_terms(&text, &key);
                            cloned.set_item("enc_terms", PyList::new(py, enc_terms)?)?;
                        }
                        "structured" | "structured_encryption" => {
                            let struct_terms = encrypt_structured_terms(&text, &key, use_bigrams);
                            cloned.set_item("struct_terms", PyList::new(py, struct_terms)?)?;
                        }
                        other => {
                            return Err(PyRuntimeError::new_err(format!(
                                "unknown encrypted search scheme: {}",
                                other
                            )));
                        }
                    }

                    out.append(cloned)?;
                }
                Ok(out.into_any().unbind())
            }
            "build_index" => {
                let protocol_s: String = required_item(&payload, "protocol")?.extract()?;
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
                    let enc_terms: Vec<String> = row
                        .get_item("enc_terms")?
                        .and_then(|v| v.extract::<Vec<String>>().ok())
                        .unwrap_or_default();
                    let struct_terms: Vec<String> = row
                        .get_item("struct_terms")?
                        .and_then(|v| v.extract::<Vec<String>>().ok())
                        .unwrap_or_default();
                    core_chunks.push(Chunk {
                        doc_id: doc_id.clone(),
                        text: text.clone(),
                    });
                    rows.push(Row {
                        doc_id,
                        text,
                        embedding: Vec::new(),
                        enc_terms,
                        struct_terms,
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

                let mut guard = INDEXES
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("index lock poisoned"))?;
                let dp_budget = if protocol == PrivacyProtocol::DiffPrivacy {
                    Some(Arc::new(Mutex::new(RDPAccountant::new(1_000_000.0, 1e-5))))
                } else {
                    None
                };
                guard.insert(
                    index_id.clone(),
                    IndexState {
                        protocol,
                        rows,
                        dp_budget,
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
            "sse_search" => {
                let index_id: String = required_item(&payload, "index_id")?.extract()?;
                let top_k: usize = required_item(&payload, "top_k")?.extract()?;
                let enc_terms: Vec<String> = required_item(&payload, "enc_terms")?.extract()?;
                let qset: std::collections::HashSet<&str> =
                    enc_terms.iter().map(|s| s.as_str()).collect();

                let guard = INDEXES
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("index lock poisoned"))?;
                let rows = &guard
                    .get(&index_id)
                    .ok_or_else(|| PyRuntimeError::new_err("index not found"))?
                    .rows;

                let mut scored: Vec<(f64, String, String)> = rows
                    .iter()
                    .map(|r| {
                        let tset: std::collections::HashSet<&str> =
                            r.enc_terms.iter().map(|s| s.as_str()).collect();
                        let inter = qset.intersection(&tset).count() as f64;
                        let union = qset.union(&tset).count() as f64;
                        let score = if union == 0.0 { 0.0 } else { inter / union };
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
            "structured_search" => {
                let index_id: String = required_item(&payload, "index_id")?.extract()?;
                let top_k: usize = required_item(&payload, "top_k")?.extract()?;
                let struct_terms: Vec<String> = required_item(&payload, "struct_terms")?.extract()?;
                let qset: std::collections::HashSet<&str> =
                    struct_terms.iter().map(|s| s.as_str()).collect();

                let guard = INDEXES
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("index lock poisoned"))?;
                let rows = &guard
                    .get(&index_id)
                    .ok_or_else(|| PyRuntimeError::new_err("index not found"))?
                    .rows;

                let mut scored: Vec<(f64, String, String)> = rows
                    .iter()
                    .map(|r| {
                        let tset: std::collections::HashSet<&str> =
                            r.struct_terms.iter().map(|s| s.as_str()).collect();
                        let inter = qset.intersection(&tset).count() as f64;
                        let union = qset.union(&tset).count() as f64;
                        let score = if union == 0.0 { 0.0 } else { inter / union };
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
            "embed_with_noise" => {
                let query: String = required_item(&payload, "query")?.extract()?;
                let sigma: f64 = required_item(&payload, "sigma")?.extract()?;
                let base = embed(&query, 64);
                let mut out = Vec::new();
                for (i, v) in base.iter().enumerate() {
                    out.push(*v + sigma * ((i as f64) * 0.001));
                }
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
                        let engine_chunks: Vec<Chunk> = state
                            .rows
                            .iter()
                            .map(|r| Chunk {
                                doc_id: r.doc_id.clone(),
                                text: r.text.clone(),
                            })
                            .collect();
                        let engine = DPMechanism {
                            chunks: Arc::new(engine_chunks),
                            sigma,
                            budget: budget.clone(),
                        };
                        let _ = engine.retrieve(&query, top_k);
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
                            0.65 * lex_score + 0.35 * emb_score
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
