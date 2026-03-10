use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use prost_types::{value, Struct, Value};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use securerag_rs::builtin_schemes::{RustSSEScheme, RustStructuredScheme};
use securerag_rs::dp::RDPAccountant;
use securerag_rs::encrypted_scheme::{EncryptedScheme, SchemeIndex, SchemeRow};
use sha2::{Digest, Sha256};
use tonic::{transport::Server, Request, Response, Status};

pub mod pb {
    tonic::include_proto!("securerag");
}

use pb::secure_retrieval_server::{SecureRetrieval, SecureRetrievalServer};
use pb::{
    BatchRetrieveRequest, BatchRetrieveResponse, BuildIndexRequest, BuildIndexResponse, ChunkRequest,
    ChunkResponse, EmbedWithNoiseRequest, EmbedWithNoiseResponse, GenerateDecoysRequest,
    GenerateDecoysResponse, RetrievalList, RetrieveByEmbeddingRequest, RetrieveByEmbeddingResponse,
    EncryptedSearchRequest, EncryptedSearchResponse, SanitizeRequest, SanitizeResponse,
};

#[derive(Clone)]
struct Row {
    doc_id: String,
    text: String,
    metadata: Struct,
    embedding: Vec<f64>,
    scheme_data: HashMap<String, serde_json::Value>,
}

struct Index {
    _protocol: String,
    dp_budget: Option<RDPAccountant>,
    chunks: Vec<Row>,
    scheme: Option<String>,
    encrypted_search_version: String,
    scheme_index: Option<Box<dyn SchemeIndex>>,
}

const LEXICAL_WEIGHT: f64 = 0.65;
const EMBEDDING_WEIGHT: f64 = 0.35;
const ENCRYPTED_SEARCH_VERSION: &str = "hmac-sha256-v1";

static INDEXES: Lazy<Arc<Mutex<HashMap<String, Index>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

static SCHEME_REGISTRY: Lazy<HashMap<&'static str, Box<dyn EncryptedScheme>>> = Lazy::new(|| {
    let mut m: HashMap<&str, Box<dyn EncryptedScheme>> = HashMap::new();
    m.insert("sse", Box::new(RustSSEScheme));
    m.insert(
        "structured",
        Box::new(RustStructuredScheme { use_bigrams: true }),
    );
    m
});

fn str_value(s: &str) -> Value {
    Value {
        kind: Some(value::Kind::StringValue(s.to_string())),
    }
}

fn num_value(n: f64) -> Value {
    Value {
        kind: Some(value::Kind::NumberValue(n)),
    }
}

fn struct_value(s: Struct) -> Value {
    Value {
        kind: Some(value::Kind::StructValue(s)),
    }
}

fn get_string(s: &Struct, key: &str) -> Option<String> {
    s.fields.get(key).and_then(|v| match &v.kind {
        Some(value::Kind::StringValue(x)) => Some(x.clone()),
        _ => None,
    })
}

fn prost_value_to_json(v: &Value) -> serde_json::Value {
    match &v.kind {
        Some(value::Kind::NullValue(_)) | None => serde_json::Value::Null,
        Some(value::Kind::NumberValue(n)) => serde_json::Value::from(*n),
        Some(value::Kind::StringValue(s)) => serde_json::Value::from(s.clone()),
        Some(value::Kind::BoolValue(b)) => serde_json::Value::from(*b),
        Some(value::Kind::StructValue(st)) => serde_json::Value::Object(
            st.fields
                .iter()
                .map(|(k, val)| (k.clone(), prost_value_to_json(val)))
                .collect(),
        ),
        Some(value::Kind::ListValue(lv)) => {
            serde_json::Value::Array(lv.values.iter().map(prost_value_to_json).collect())
        }
    }
}

fn struct_to_map(st: &Struct) -> HashMap<String, serde_json::Value> {
    st.fields
        .iter()
        .map(|(k, v)| (k.clone(), prost_value_to_json(v)))
        .collect()
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

fn jaccard_tokens(a: &str, b: &str) -> f64 {
    let sa: HashSet<String> = tokenize(a).into_iter().collect();
    let sb: HashSet<String> = tokenize(b).into_iter().collect();
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let uni = sa.union(&sb).count() as f64;
    if uni == 0.0 { 0.0 } else { inter / uni }
}

fn row_to_struct(doc_id: &str, text: &str, metadata: Struct, score: f64) -> Struct {
    let mut out = Struct { fields: BTreeMap::new() };
    out.fields.insert("doc_id".to_string(), str_value(doc_id));
    out.fields.insert("text".to_string(), str_value(text));
    out.fields.insert("metadata".to_string(), struct_value(metadata));
    out.fields.insert("score".to_string(), num_value(score));
    out
}

#[derive(Default)]
struct SecureRetrievalSvc;

#[tonic::async_trait]
impl SecureRetrieval for SecureRetrievalSvc {
    async fn chunk(&self, request: Request<ChunkRequest>) -> Result<Response<ChunkResponse>, Status> {
        let req = request.into_inner();
        let chunk_size = if req.chunk_size <= 0 { 512usize } else { req.chunk_size as usize };
        let overlap = if req.overlap < 0 { 0usize } else { req.overlap as usize };
        let step = if chunk_size > overlap { chunk_size - overlap } else { 1 };

        let mut out = Vec::new();
        for d in req.docs {
            let doc_id = get_string(&d, "doc_id").ok_or_else(|| Status::invalid_argument("missing doc_id"))?;
            let text = get_string(&d, "text").ok_or_else(|| Status::invalid_argument("missing text"))?;
            let metadata = match d.fields.get("metadata").and_then(|v| v.kind.clone()) {
                Some(value::Kind::StructValue(s)) => s,
                _ => Struct { fields: BTreeMap::new() },
            };

            if text.len() <= chunk_size {
                let mut row = Struct { fields: BTreeMap::new() };
                row.fields.insert("doc_id".to_string(), str_value(&doc_id));
                row.fields.insert("text".to_string(), str_value(&text));
                row.fields.insert("metadata".to_string(), struct_value(metadata));
                out.push(row);
                continue;
            }

            let mut i = 0usize;
            while i < text.len() {
                let end = usize::min(i + chunk_size, text.len());
                let snippet = &text[i..end];
                if !snippet.is_empty() && (i == 0 || snippet.len() >= usize::max(24, chunk_size / 3)) {
                    let mut row = Struct { fields: BTreeMap::new() };
                    row.fields.insert("doc_id".to_string(), str_value(&doc_id));
                    row.fields.insert("text".to_string(), str_value(snippet));
                    row.fields.insert("metadata".to_string(), struct_value(metadata.clone()));
                    out.push(row);
                }
                i += step;
            }
        }

        Ok(Response::new(ChunkResponse { chunks: out }))
    }

    async fn sanitize(&self, request: Request<SanitizeRequest>) -> Result<Response<SanitizeResponse>, Status> {
        let req = request.into_inner();
        let mut out = Vec::new();
        for mut c in req.chunks {
            let text = get_string(&c, "text").unwrap_or_default();
            let mut cleaned = text;
            for bad in ["ignore previous instructions", "system prompt", "developer instructions"] {
                cleaned = cleaned.replace(bad, "");
                cleaned = cleaned.replace(&bad.to_ascii_uppercase(), "");
            }
            c.fields.insert("text".to_string(), str_value(&cleaned));
            out.push(c);
        }
        Ok(Response::new(SanitizeResponse { chunks: out }))
    }

    async fn build_index(&self, request: Request<BuildIndexRequest>) -> Result<Response<BuildIndexResponse>, Status> {
        let req = request.into_inner();
        let mut rows = Vec::new();
        for c in req.chunks {
            let doc_id = get_string(&c, "doc_id").ok_or_else(|| Status::invalid_argument("missing doc_id"))?;
            let text = get_string(&c, "text").ok_or_else(|| Status::invalid_argument("missing text"))?;
            let metadata = match c.fields.get("metadata").and_then(|v| v.kind.clone()) {
                Some(value::Kind::StructValue(s)) => s,
                _ => Struct { fields: BTreeMap::new() },
            };
            let scheme_data = c
                .fields
                .get("scheme_data")
                .and_then(|v| match &v.kind {
                    Some(value::Kind::StructValue(s)) => Some(struct_to_map(s)),
                    _ => None,
                })
                .unwrap_or_default();
            rows.push(Row {
                doc_id,
                text: text.clone(),
                metadata,
                embedding: embed(&text, 64),
                scheme_data,
            });
        }

        let scheme_index = if req.protocol == "EncryptedSearch" {
            let scheme_name = req.encrypted_search_scheme.to_ascii_lowercase();
            let scheme = SCHEME_REGISTRY
                .get(scheme_name.as_str())
                .ok_or_else(|| Status::unimplemented(format!(
                    "encrypted scheme '{}' not registered in Rust server",
                    scheme_name
                )))?;
            let scheme_rows: Vec<SchemeRow> = rows
                .iter()
                .map(|r| SchemeRow {
                    doc_id: r.doc_id.clone(),
                    text: r.text.clone(),
                    metadata: r.metadata.clone(),
                    scheme_data: r.scheme_data.clone(),
                })
                .collect();
            Some(scheme.build_index(&scheme_rows))
        } else {
            None
        };

        let scheme_name = if req.protocol == "EncryptedSearch" {
            Some(req.encrypted_search_scheme.to_ascii_lowercase())
        } else {
            None
        };
        let encrypted_search_version = if req.encrypted_search_version.is_empty() {
            "sha256-v0".to_string()
        } else {
            req.encrypted_search_version.clone()
        };

        let index_id = format!("rsgrpc-{}", uuid::Uuid::new_v4());
        let doc_count = rows.len() as i32;
        let mut guard = INDEXES.lock().map_err(|_| Status::internal("index lock poisoned"))?;
        let dp_budget = if req.protocol == "DiffPrivacy" {
            Some(RDPAccountant::new(req.epsilon, req.delta))
        } else {
            None
        };
        guard.insert(
            index_id.clone(),
            Index {
                _protocol: req.protocol,
                dp_budget,
                chunks: rows,
                scheme: scheme_name,
                encrypted_search_version,
                scheme_index,
            },
        );
        Ok(Response::new(BuildIndexResponse { index_id, doc_count }))
    }

    async fn generate_decoys(&self, request: Request<GenerateDecoysRequest>) -> Result<Response<GenerateDecoysResponse>, Status> {
        let req = request.into_inner();
        let guard = INDEXES.lock().map_err(|_| Status::internal("index lock poisoned"))?;
        let index = guard.get(&req.index_id).ok_or_else(|| Status::not_found("index not found"))?;
        let k = req.k.max(0) as usize;
        if index.chunks.is_empty() {
            return Ok(Response::new(GenerateDecoysResponse { decoys: vec![req.query; k] }));
        }
        let seed = req
            .query
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(b as u64));
        let mut out = Vec::new();
        for i in 0..k {
            let idx = ((seed as usize).wrapping_add(i * 17)) % index.chunks.len();
            out.push(index.chunks[idx].text.clone());
        }
        Ok(Response::new(GenerateDecoysResponse { decoys: out }))
    }

    async fn batch_retrieve(&self, request: Request<BatchRetrieveRequest>) -> Result<Response<BatchRetrieveResponse>, Status> {
        let req = request.into_inner();
        let top_k = req.top_k.max(0) as usize;
        let guard = INDEXES.lock().map_err(|_| Status::internal("index lock poisoned"))?;
        let index = guard.get(&req.index_id).ok_or_else(|| Status::not_found("index not found"))?;

        let mut all = Vec::new();
        for q in req.queries {
            let mut scored: Vec<(f64, usize)> = index
                .chunks
                .iter()
                .enumerate()
                .map(|(i, r)| (jaccard_tokens(&q, &r.text), i))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let rows = scored
                .into_iter()
                .take(top_k)
                .map(|(score, i)| {
                    let r = &index.chunks[i];
                    row_to_struct(&r.doc_id, &r.text, r.metadata.clone(), score)
                })
                .collect();
            all.push(RetrievalList { rows });
        }
        Ok(Response::new(BatchRetrieveResponse { rows: all }))
    }

    async fn embed_with_noise(&self, request: Request<EmbedWithNoiseRequest>) -> Result<Response<EmbedWithNoiseResponse>, Status> {
        let req = request.into_inner();
        let base = embed(&req.query, 64);
        let digest = Sha256::digest(req.query.as_bytes());
        let seed = u64::from_le_bytes(
            digest[..8]
                .try_into()
                .map_err(|_| Status::internal("invalid seed bytes"))?,
        );
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, req.sigma).map_err(|e| Status::invalid_argument(format!("invalid sigma: {e}")))?;
        let out = base.into_iter().map(|v| v + normal.sample(&mut rng)).collect();
        Ok(Response::new(EmbedWithNoiseResponse { embedding: out }))
    }

    async fn retrieve_by_embedding(&self, request: Request<RetrieveByEmbeddingRequest>) -> Result<Response<RetrieveByEmbeddingResponse>, Status> {
        let req = request.into_inner();
        let top_k = req.top_k.max(0) as usize;
        let mut guard = INDEXES.lock().map_err(|_| Status::internal("index lock poisoned"))?;
        let index = guard.get_mut(&req.index_id).ok_or_else(|| Status::not_found("index not found"))?;

        if let Some(budget) = index.dp_budget.as_mut() {
            budget
                .consume_rdp(req.sigma)
                .map_err(Status::resource_exhausted)?;
        }

        let qemb = req.embedding;
        let qtok: HashSet<String> = tokenize(&req.query).into_iter().collect();

        let mut scored: Vec<(f64, usize)> = index
            .chunks
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let emb_score = cos(&qemb, &r.embedding);
                let lex = if qtok.is_empty() {
                    0.0
                } else {
                    let t: HashSet<String> = tokenize(&r.text).into_iter().collect();
                    let inter = qtok.intersection(&t).count() as f64;
                    let uni = qtok.union(&t).count() as f64;
                    if uni == 0.0 { 0.0 } else { inter / uni }
                };
                let score = if qtok.is_empty() {
                    emb_score
                } else {
                    LEXICAL_WEIGHT * lex + EMBEDDING_WEIGHT * emb_score
                };
                (score, i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let rows = scored
            .into_iter()
            .take(top_k)
            .map(|(score, i)| {
                let r = &index.chunks[i];
                row_to_struct(&r.doc_id, &r.text, r.metadata.clone(), score)
            })
            .collect();
        Ok(Response::new(RetrieveByEmbeddingResponse { rows }))
    }

    async fn encrypted_search(&self, request: Request<EncryptedSearchRequest>) -> Result<Response<EncryptedSearchResponse>, Status> {
        let req = request.into_inner();
        let top_k = req.top_k.max(0) as usize;
        let guard = INDEXES.lock().map_err(|_| Status::internal("index lock poisoned"))?;
        let index = guard.get(&req.index_id).ok_or_else(|| Status::not_found("index not found"))?;
        let scheme_index = index
            .scheme_index
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("index was not built with an encrypted scheme"))?;

        if index.scheme.is_some() && index.encrypted_search_version != ENCRYPTED_SEARCH_VERSION {
            return Err(Status::failed_precondition(format!(
                "Index built with crypto version '{}' is incompatible with current '{}'. Rebuild the corpus.",
                index.encrypted_search_version,
                ENCRYPTED_SEARCH_VERSION
            )));
        }

        let encrypted_query = req
            .encrypted_query
            .as_ref()
            .map(struct_to_map)
            .unwrap_or_default();
        let results = scheme_index.search(&encrypted_query, top_k);

        let rows = results
            .into_iter()
            .map(|r| {
                row_to_struct(
                    &r.doc_id,
                    &r.text,
                    r.metadata,
                    r.score,
                )
            })
            .collect();
        Ok(Response::new(EncryptedSearchResponse { rows }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 50051;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => {
                if let Some(v) = args.next() {
                    host = v;
                }
            }
            "--port" => {
                if let Some(v) = args.next() {
                    if let Ok(parsed) = v.parse::<u16>() {
                        port = parsed;
                    }
                }
            }
            _ => {}
        }
    }

    let addr = format!("{}:{}", host, port).parse()?;
    let svc = SecureRetrievalSvc;
    Server::builder()
        .add_service(SecureRetrievalServer::new(svc))
        .serve(addr)
        .await?;
    Ok(())
}
