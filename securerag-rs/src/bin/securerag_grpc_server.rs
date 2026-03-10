use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use prost_types::{value, ListValue, Struct, Value};
use rand::RngCore;
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
    SanitizeRequest, SanitizeResponse, SseEncryptStructuredTermsRequest,
    SseEncryptStructuredTermsResponse, SseEncryptTermsRequest, SseEncryptTermsResponse,
    SseGenerateKeyRequest, SseGenerateKeyResponse, SsePrepareChunksRequest, SsePrepareChunksResponse,
    SseSearchRequest, SseSearchResponse, StructuredSearchRequest, StructuredSearchResponse,
};

#[derive(Clone)]
struct Row {
    doc_id: String,
    text: String,
    metadata: Struct,
    embedding: Vec<f64>,
    enc_terms: Vec<String>,
    struct_terms: Vec<String>,
}

#[derive(Clone)]
struct Index {
    _protocol: String,
    chunks: Vec<Row>,
}

static INDEXES: Lazy<Arc<Mutex<HashMap<String, Index>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

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

fn list_value(values: Vec<Value>) -> Value {
    Value {
        kind: Some(value::Kind::ListValue(ListValue { values })),
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

fn get_string_list(s: &Struct, key: &str) -> Option<Vec<String>> {
    s.fields.get(key).and_then(|v| match &v.kind {
        Some(value::Kind::ListValue(list)) => {
            let mut out = Vec::new();
            for item in &list.values {
                if let Some(value::Kind::StringValue(x)) = &item.kind {
                    out.push(x.clone());
                }
            }
            Some(out)
        }
        _ => None,
    })
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
            let enc_terms = get_string_list(&c, "enc_terms").unwrap_or_default();
            let struct_terms = get_string_list(&c, "struct_terms").unwrap_or_default();
            rows.push(Row {
                doc_id,
                text: text.clone(),
                metadata,
                embedding: embed(&text, 64),
                enc_terms,
                struct_terms,
            });
        }
        let index_id = format!("rsgrpc-{}", uuid::Uuid::new_v4());
        let doc_count = rows.len() as i32;
        let mut guard = INDEXES.lock().map_err(|_| Status::internal("index lock poisoned"))?;
        guard.insert(index_id.clone(), Index { _protocol: req.protocol, chunks: rows });
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
        let out = base
            .into_iter()
            .enumerate()
            .map(|(i, v)| v + req.sigma * ((i as f64) * 0.001))
            .collect();
        Ok(Response::new(EmbedWithNoiseResponse { embedding: out }))
    }

    async fn retrieve_by_embedding(&self, request: Request<RetrieveByEmbeddingRequest>) -> Result<Response<RetrieveByEmbeddingResponse>, Status> {
        let req = request.into_inner();
        let top_k = req.top_k.max(0) as usize;
        let guard = INDEXES.lock().map_err(|_| Status::internal("index lock poisoned"))?;
        let index = guard.get(&req.index_id).ok_or_else(|| Status::not_found("index not found"))?;

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
                let score = if qtok.is_empty() { emb_score } else { 0.65 * lex + 0.35 * emb_score };
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

    async fn sse_generate_key(&self, _request: Request<SseGenerateKeyRequest>) -> Result<Response<SseGenerateKeyResponse>, Status> {
        let mut bytes = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut bytes);
        let key = bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        Ok(Response::new(SseGenerateKeyResponse { key }))
    }

    async fn sse_encrypt_terms(&self, request: Request<SseEncryptTermsRequest>) -> Result<Response<SseEncryptTermsResponse>, Status> {
        let req = request.into_inner();
        Ok(Response::new(SseEncryptTermsResponse {
            terms: encrypt_terms(&req.text, &req.key),
        }))
    }

    async fn sse_encrypt_structured_terms(
        &self,
        request: Request<SseEncryptStructuredTermsRequest>,
    ) -> Result<Response<SseEncryptStructuredTermsResponse>, Status> {
        let req = request.into_inner();
        Ok(Response::new(SseEncryptStructuredTermsResponse {
            terms: encrypt_structured_terms(&req.text, &req.key, req.use_bigrams),
        }))
    }

    async fn sse_prepare_chunks(&self, request: Request<SsePrepareChunksRequest>) -> Result<Response<SsePrepareChunksResponse>, Status> {
        let req = request.into_inner();
        let scheme = req.scheme.to_ascii_lowercase();
        let mut out = Vec::new();
        for mut c in req.chunks {
            let text = get_string(&c, "text").unwrap_or_default();
            if scheme == "sse" {
                let terms = encrypt_terms(&text, &req.key)
                    .into_iter()
                    .map(|x| str_value(&x))
                    .collect();
                c.fields.insert("enc_terms".to_string(), list_value(terms));
            } else if scheme == "structured" || scheme == "structured_encryption" {
                let terms = encrypt_structured_terms(&text, &req.key, req.use_bigrams)
                    .into_iter()
                    .map(|x| str_value(&x))
                    .collect();
                c.fields.insert("struct_terms".to_string(), list_value(terms));
            } else {
                return Err(Status::invalid_argument("unknown encrypted search scheme"));
            }
            out.push(c);
        }
        Ok(Response::new(SsePrepareChunksResponse { chunks: out }))
    }

    async fn sse_search(&self, request: Request<SseSearchRequest>) -> Result<Response<SseSearchResponse>, Status> {
        let req = request.into_inner();
        let top_k = req.top_k.max(0) as usize;
        let qset: HashSet<String> = req.enc_terms.into_iter().collect();
        let guard = INDEXES.lock().map_err(|_| Status::internal("index lock poisoned"))?;
        let index = guard.get(&req.index_id).ok_or_else(|| Status::not_found("index not found"))?;

        let mut scored: Vec<(f64, usize)> = index
            .chunks
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let tset: HashSet<String> = r.enc_terms.iter().cloned().collect();
                let inter = qset.intersection(&tset).count() as f64;
                let uni = qset.union(&tset).count() as f64;
                (if uni == 0.0 { 0.0 } else { inter / uni }, i)
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
        Ok(Response::new(SseSearchResponse { rows }))
    }

    async fn structured_search(&self, request: Request<StructuredSearchRequest>) -> Result<Response<StructuredSearchResponse>, Status> {
        let req = request.into_inner();
        let top_k = req.top_k.max(0) as usize;
        let qset: HashSet<String> = req.struct_terms.into_iter().collect();
        let guard = INDEXES.lock().map_err(|_| Status::internal("index lock poisoned"))?;
        let index = guard.get(&req.index_id).ok_or_else(|| Status::not_found("index not found"))?;

        let mut scored: Vec<(f64, usize)> = index
            .chunks
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let tset: HashSet<String> = r.struct_terms.iter().cloned().collect();
                let inter = qset.intersection(&tset).count() as f64;
                let uni = qset.union(&tset).count() as f64;
                (if uni == 0.0 { 0.0 } else { inter / uni }, i)
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
        Ok(Response::new(StructuredSearchResponse { rows }))
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
