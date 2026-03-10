use crate::core::CorpusProcessor;
use crate::protocol::PrivacyProtocol;
use crate::types::{Chunk, IndexPayload};

pub struct PlainIndexBuilder;

impl CorpusProcessor for PlainIndexBuilder {
    fn build(&self, chunks: Vec<Chunk>) -> IndexPayload {
        IndexPayload::Chunks(chunks)
    }

    fn protocol(&self) -> PrivacyProtocol {
        PrivacyProtocol::Baseline
    }

    fn update(&self, _id: &str, chunk: Chunk) -> IndexPayload {
        IndexPayload::Chunks(vec![chunk])
    }
}

pub struct EmbeddingIndexBuilder;

impl CorpusProcessor for EmbeddingIndexBuilder {
    fn build(&self, chunks: Vec<Chunk>) -> IndexPayload {
        IndexPayload::Chunks(chunks)
    }

    fn protocol(&self) -> PrivacyProtocol {
        PrivacyProtocol::DiffPrivacy
    }

    fn update(&self, _id: &str, chunk: Chunk) -> IndexPayload {
        IndexPayload::Chunks(vec![chunk])
    }
}

pub struct UnsupportedBuilder {
    pub protocol_value: PrivacyProtocol,
}

impl CorpusProcessor for UnsupportedBuilder {
    fn build(&self, chunks: Vec<Chunk>) -> IndexPayload {
        let _ = self.protocol_value;
        IndexPayload::Chunks(chunks)
    }

    fn protocol(&self) -> PrivacyProtocol {
        self.protocol_value
    }

    fn update(&self, _id: &str, chunk: Chunk) -> IndexPayload {
        IndexPayload::Chunks(vec![chunk])
    }
}
