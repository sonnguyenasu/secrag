use std::collections::HashMap;

use prost_types::Struct;

pub struct SearchResult {
    pub doc_id: String,
    pub text: String,
    pub metadata: Struct,
    pub score: f64,
}

pub trait EncryptedScheme: Send + Sync {
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
    pub doc_id: String,
    pub text: String,
    pub metadata: Struct,
    pub scheme_data: HashMap<String, serde_json::Value>,
}
