use std::collections::HashMap;

#[cfg(test)]
use hmac::{Hmac, Mac};
#[cfg(test)]
use sha2::Sha256;

use crate::encrypted_scheme::{EncryptedScheme, SchemeIndex, SchemeRow, SearchResult};

pub struct RustSSEScheme;

pub struct RustStructuredScheme {
    pub use_bigrams: bool,
}

struct InvertedIndex {
    rows: Vec<(String, String, prost_types::Struct, Vec<String>)>,
    inv: HashMap<String, Vec<usize>>,
    query_field: &'static str,
}

impl InvertedIndex {
    fn build(rows: &[SchemeRow], field: &'static str, query_field: &'static str) -> Self {
        let mut packed_rows: Vec<(String, String, prost_types::Struct, Vec<String>)> =
            Vec::with_capacity(rows.len());
        let mut inv: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, row) in rows.iter().enumerate() {
            let terms = row
                .scheme_data
                .get(field)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            for term in &terms {
                inv.entry(term.clone()).or_default().push(i);
            }
            packed_rows.push((
                row.doc_id.clone(),
                row.text.clone(),
                row.metadata.clone(),
                terms,
            ));
        }
        Self {
            rows: packed_rows,
            inv,
            query_field,
        }
    }
}

impl SchemeIndex for InvertedIndex {
    fn search(
        &self,
        encrypted_query: &HashMap<String, serde_json::Value>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let q_terms = encrypted_query
            .get(self.query_field)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if q_terms.is_empty() {
            return Vec::new();
        }

        let mut counts: HashMap<usize, usize> = HashMap::new();
        for term in &q_terms {
            if let Some(postings) = self.inv.get(term) {
                for idx in postings {
                    *counts.entry(*idx).or_insert(0) += 1;
                }
            }
        }

        let q_len = q_terms.len();
        let mut scored: Vec<SearchResult> = counts
            .into_iter()
            .map(|(idx, inter)| {
                let row = &self.rows[idx];
                let union = q_len + row.3.len() - inter;
                let score = if union == 0 {
                    0.0
                } else {
                    inter as f64 / union as f64
                };
                SearchResult {
                    doc_id: row.0.clone(),
                    text: row.1.clone(),
                    metadata: row.2.clone(),
                    score,
                }
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

impl EncryptedScheme for RustSSEScheme {
    fn build_index(&self, rows: &[SchemeRow]) -> Box<dyn SchemeIndex> {
        Box::new(InvertedIndex::build(rows, "enc_terms", "enc_terms"))
    }
}

impl EncryptedScheme for RustStructuredScheme {
    fn build_index(&self, rows: &[SchemeRow]) -> Box<dyn SchemeIndex> {
        let _ = self.use_bigrams;
        Box::new(InvertedIndex::build(rows, "struct_terms", "struct_terms"))
    }
}

#[cfg(test)]
pub fn hmac_sha256_hex(token: &str, key: &str) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(key.as_bytes()).expect("valid key");
    mac.update(token.as_bytes());
    let out = mac.finalize().into_bytes();
    out.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::hmac_sha256_hex;

    #[test]
    fn hmac_hex_parity_contract() {
        assert_eq!(
            hmac_sha256_hex("tok:alpha", "abcd1234abcd1234abcd1234abcd1234"),
            "b5187317422e4f4b011ba33c1450032a763c9337170feb04932b94ecbcc21faa"
        );
    }
}
