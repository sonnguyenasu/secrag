use std::sync::{Arc, Mutex};

use crate::core::RetrieverCore;
use crate::dp::RDPAccountant;
use crate::protocol::PrivacyProtocol;
use crate::types::{Chunk, Document};

fn score_lexical(query: &str, text: &str) -> f64 {
    let q: Vec<&str> = query.split_whitespace().collect();
    let t: Vec<&str> = text.split_whitespace().collect();
    if q.is_empty() || t.is_empty() {
        return 0.0;
    }
    let mut inter = 0usize;
    for token in q.iter() {
        if t.iter().any(|x| x.eq_ignore_ascii_case(token)) {
            inter += 1;
        }
    }
    inter as f64 / (q.len() + t.len()) as f64
}

pub struct BaselineEngine {
    pub chunks: Arc<Vec<Chunk>>,
}

impl RetrieverCore for BaselineEngine {
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<Document> {
        let mut rows: Vec<Document> = self
            .chunks
            .iter()
            .map(|c| Document {
                doc_id: c.doc_id.clone(),
                text: c.text.clone(),
                score: score_lexical(query, &c.text),
            })
            .collect();
        rows.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows.into_iter().take(top_k).collect()
    }

    fn protocol(&self) -> PrivacyProtocol {
        PrivacyProtocol::Baseline
    }

    fn index_size(&self) -> usize {
        self.chunks.len()
    }
}

pub struct DPMechanism {
    pub chunks: Arc<Vec<Chunk>>,
    pub sigma: f64,
    pub budget: Arc<Mutex<RDPAccountant>>,
}

impl RetrieverCore for DPMechanism {
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<Document> {
        if let Ok(mut b) = self.budget.lock() {
            let _ = b.consume_rdp(self.sigma);
        }
        let mut rows: Vec<Document> = self
            .chunks
            .iter()
            .map(|c| Document {
                doc_id: c.doc_id.clone(),
                text: c.text.clone(),
                score: score_lexical(query, &c.text),
            })
            .collect();
        rows.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows.into_iter().take(top_k).collect()
    }

    fn protocol(&self) -> PrivacyProtocol {
        PrivacyProtocol::DiffPrivacy
    }

    fn index_size(&self) -> usize {
        self.chunks.len()
    }
}

pub struct UnsupportedEngine {
    pub protocol_value: PrivacyProtocol,
}

impl RetrieverCore for UnsupportedEngine {
    fn retrieve(&self, _query: &str, _top_k: usize) -> Vec<Document> {
        vec![]
    }

    fn protocol(&self) -> PrivacyProtocol {
        self.protocol_value
    }

    fn index_size(&self) -> usize {
        0
    }
}
