use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::protocol::PrivacyProtocol;

#[derive(Clone)]
struct Row {
    doc_id: String,
    text: String,
    enc_terms: Vec<String>,
    struct_terms: Vec<String>,
}

static INDEXES: Lazy<Mutex<HashMap<String, Vec<Row>>>> =
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

fn lexical_score(query: &str, text: &str) -> f64 {
    let q = tokenize(query);
    let t = tokenize(text);
    if q.is_empty() || t.is_empty() {
        return 0.0;
    }

    let qset: std::collections::HashSet<&str> = q.iter().map(|s| s.as_str()).collect();
    let tset: std::collections::HashSet<&str> = t.iter().map(|s| s.as_str()).collect();
    let inter = qset.intersection(&tset).count() as f64;
    let union = qset.union(&tset).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
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
            "build_index" => {
                let protocol_s: String = required_item(&payload, "protocol")?.extract()?;
                let _protocol = PrivacyProtocol::from_wire(&protocol_s)
                    .ok_or_else(|| PyRuntimeError::new_err("invalid protocol"))?;
                let chunks_any = required_item(&payload, "chunks")?;
                let chunks_list = chunks_any.downcast::<PyList>()?;

                let mut rows: Vec<Row> = Vec::new();
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
                    rows.push(Row {
                        doc_id,
                        text,
                        enc_terms,
                        struct_terms,
                    });
                }

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
                guard.insert(index_id.clone(), rows);

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
                let rows = guard
                    .get(&index_id)
                    .ok_or_else(|| PyRuntimeError::new_err("index not found"))?;

                let all = PyList::empty(py);
                for q in queries.iter() {
                    let query: String = q.extract()?;
                    let mut scored: Vec<(f64, String, String)> = rows
                        .iter()
                        .map(|r| {
                            let score = lexical_score(&query, &r.text);
                            (score, r.doc_id.clone(), r.text.clone())
                        })
                        .collect();
                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                    let out_q = PyList::empty(py);
                    for (score, doc_id, text) in scored.into_iter().take(top_k) {
                        let row = PyDict::new(py);
                        row.set_item("doc_id", doc_id)?;
                        row.set_item("text", text)?;
                        row.set_item("metadata", PyDict::new(py))?;
                        row.set_item("score", score)?;
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
                let rows = guard
                    .get(&index_id)
                    .ok_or_else(|| PyRuntimeError::new_err("index not found"))?;

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
                let rows = guard
                    .get(&index_id)
                    .ok_or_else(|| PyRuntimeError::new_err("index not found"))?;

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
                let rows = guard
                    .get(&index_id)
                    .ok_or_else(|| PyRuntimeError::new_err("index not found"))?;

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
                let base = query.bytes().map(|b| b as f64 / 255.0).collect::<Vec<_>>();
                let mut out = Vec::new();
                for (i, v) in base.iter().enumerate().take(16) {
                    out.push(*v + sigma * ((i as f64) * 0.01));
                }
                while out.len() < 16 {
                    out.push(0.0);
                }
                Ok(PyList::new(py, out)?.into_any().unbind())
            }
            "retrieve_by_embedding" => {
                let index_id: String = required_item(&payload, "index_id")?.extract()?;
                let top_k: usize = required_item(&payload, "top_k")?.extract()?;
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
                let rows = guard
                    .get(&index_id)
                    .ok_or_else(|| PyRuntimeError::new_err("index not found"))?;

                let mut scored: Vec<(f64, String, String)> = rows
                    .iter()
                    .map(|r| {
                        let score = if query.is_empty() {
                            0.5
                        } else if r.text.to_lowercase().contains(&query) {
                            1.0
                        } else {
                            0.1
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
