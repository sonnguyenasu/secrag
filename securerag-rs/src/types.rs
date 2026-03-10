#[derive(Debug, Clone)]
pub struct Document {
    pub doc_id: String,
    pub text: String,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub doc_id: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub enum IndexPayload {
    Chunks(Vec<Chunk>),
}
