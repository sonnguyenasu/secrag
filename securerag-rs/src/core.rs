use crate::protocol::PrivacyProtocol;
use crate::types::{Chunk, Document, IndexPayload};

pub trait RetrieverCore: Send + Sync {
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<Document>;
    fn protocol(&self) -> PrivacyProtocol;
    fn index_size(&self) -> usize;
}

pub trait CorpusProcessor: Send + Sync {
    fn build(&self, chunks: Vec<Chunk>) -> IndexPayload;
    fn protocol(&self) -> PrivacyProtocol;
    fn update(&self, _id: &str, _chunk: Chunk) -> IndexPayload;
}
