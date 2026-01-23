//! Indexing infrastructure for RLM context search.
//!
//! This module provides BM25 keyword search over chunked context.

use bm25::Document;
use bm25::SearchEngine;
use bm25::SearchEngineBuilder;
use serde::Deserialize;
use serde::Serialize;

/// Configuration for indexing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Whether indexing is enabled.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Chunk size in characters.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Overlap between chunks in characters.
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
}

fn default_enabled() -> bool {
    true
}

fn default_chunk_size() -> usize {
    1000
}

fn default_chunk_overlap() -> usize {
    100
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
        }
    }
}

/// A chunk of indexed content.
#[derive(Clone, Debug)]
pub struct Chunk {
    /// Unique chunk ID.
    pub id: u32,
    /// Start offset in the original content (bytes).
    pub start: usize,
    /// End offset in the original content (bytes).
    pub end: usize,
    /// The chunk content.
    pub content: String,
}

/// Search result from BM25 query.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// The matching chunk.
    pub chunk: Chunk,
    /// BM25 relevance score.
    pub score: f32,
}

/// Serializable search result for Python API.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResultJson {
    /// The matching text content.
    pub text: String,
    /// BM25 relevance score.
    pub score: f32,
    /// Start offset in the original content (bytes).
    pub start: usize,
    /// End offset in the original content (bytes).
    pub end: usize,
}

impl From<SearchResult> for SearchResultJson {
    fn from(result: SearchResult) -> Self {
        Self {
            text: result.chunk.content,
            score: result.score,
            start: result.chunk.start,
            end: result.chunk.end,
        }
    }
}

/// BM25 index over content chunks.
pub struct Bm25Index {
    /// The search engine.
    engine: SearchEngine<u32>,
    /// Stored chunks for retrieval.
    chunks: Vec<Chunk>,
    /// Configuration.
    config: IndexConfig,
}

impl Bm25Index {
    /// Create a new empty index.
    pub fn new(config: IndexConfig) -> Self {
        // Estimate average document length based on chunk size
        let avgdl = config.chunk_size as f32;
        Self {
            engine: SearchEngineBuilder::with_avgdl(avgdl).build(),
            chunks: Vec::new(),
            config,
        }
    }

    /// Build an index from content.
    pub fn from_content(content: &str, config: IndexConfig) -> Self {
        let chunks = chunk_content(content, config.chunk_size, config.chunk_overlap);

        // Build engine with estimated avgdl, then upsert documents
        let avgdl = config.chunk_size as f32;
        let mut engine: SearchEngine<u32> = SearchEngineBuilder::with_avgdl(avgdl).build();

        for chunk in &chunks {
            engine.upsert(Document {
                id: chunk.id,
                contents: chunk.content.clone(),
            });
        }

        Self {
            engine,
            chunks,
            config,
        }
    }

    /// Search the index for matching chunks.
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        if self.chunks.is_empty() {
            return Vec::new();
        }

        let results = self.engine.search(query, k);
        results
            .into_iter()
            .filter_map(|r| {
                self.chunks
                    .iter()
                    .find(|c| c.id == r.document.id)
                    .map(|c| SearchResult {
                        chunk: c.clone(),
                        score: r.score,
                    })
            })
            .collect()
    }

    /// Get the number of indexed chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Get a chunk by ID.
    pub fn get_chunk(&self, id: u32) -> Option<&Chunk> {
        self.chunks.iter().find(|c| c.id == id)
    }

    /// Get all chunks.
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// Get the index configuration.
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }
}

/// Split content into overlapping chunks.
fn chunk_content(content: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
    if content.is_empty() || chunk_size == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut id = 0u32;
    let mut start = 0;

    while start < len {
        // Find end position, trying to break at word boundary
        let raw_end = (start + chunk_size).min(len);
        let end = find_word_boundary(content, raw_end);

        // Extract chunk
        let chunk_str = &content[start..end];
        chunks.push(Chunk {
            id,
            start,
            end,
            content: chunk_str.to_string(),
        });

        id += 1;

        // Move start forward, accounting for overlap
        let step = if chunk_size > overlap {
            chunk_size - overlap
        } else {
            chunk_size
        };
        start += step;

        // Ensure we make progress
        if start >= end && end < len {
            start = end;
        }
    }

    chunks
}

/// Find a word boundary near the given position.
fn find_word_boundary(content: &str, pos: usize) -> usize {
    if pos >= content.len() {
        return content.len();
    }

    // Look backwards for whitespace
    let bytes = content.as_bytes();
    for i in (pos.saturating_sub(50)..=pos).rev() {
        if i < bytes.len() && bytes[i].is_ascii_whitespace() {
            return i + 1;
        }
    }

    // Look forwards for whitespace
    let end = (pos + 50).min(bytes.len());
    for (i, byte) in bytes.iter().enumerate().take(end).skip(pos) {
        if byte.is_ascii_whitespace() {
            return i;
        }
    }

    // Fall back to exact position
    pos
}

/// A chunk from a specific document.
#[derive(Clone, Debug)]
pub struct DocChunk {
    /// Unique chunk ID.
    pub id: u32,
    /// Document ID (relative path).
    pub doc_id: String,
    /// Start offset in the document (bytes).
    pub start: usize,
    /// End offset in the document (bytes).
    pub end: usize,
    /// The chunk content.
    pub content: String,
}

/// Search result from document-aware BM25 query.
#[derive(Clone, Debug)]
pub struct DocSearchResult {
    /// The matching chunk.
    pub chunk: DocChunk,
    /// BM25 relevance score.
    pub score: f32,
}

/// BM25 index over a tree of documents.
pub struct DocTreeIndex {
    /// The search engine.
    engine: SearchEngine<u32>,
    /// Stored chunks for retrieval.
    chunks: Vec<DocChunk>,
    /// Configuration.
    config: IndexConfig,
    /// Total documents indexed.
    doc_count: usize,
}

impl DocTreeIndex {
    /// Create a new empty document tree index.
    pub fn new(config: IndexConfig) -> Self {
        let avgdl = config.chunk_size as f32;
        Self {
            engine: SearchEngineBuilder::with_avgdl(avgdl).build(),
            chunks: Vec::new(),
            config,
            doc_count: 0,
        }
    }

    /// Build an index from a DocTreeStore.
    pub fn from_doc_tree(store: &crate::context::DocTreeStore, config: IndexConfig) -> Self {
        let mut index = Self::new(config.clone());

        let doc_ids = store.list_documents();
        index.doc_count = doc_ids.len();

        let mut global_id = 0u32;
        for doc_id in doc_ids {
            if let Some(doc) = store.get_document(doc_id) {
                let chunks = chunk_content(&doc.content, config.chunk_size, config.chunk_overlap);
                for chunk in chunks {
                    let doc_chunk = DocChunk {
                        id: global_id,
                        doc_id: doc_id.to_string(),
                        start: chunk.start,
                        end: chunk.end,
                        content: chunk.content.clone(),
                    };

                    index.engine.upsert(Document {
                        id: global_id,
                        contents: chunk.content,
                    });

                    index.chunks.push(doc_chunk);
                    global_id += 1;
                }
            }
        }

        index
    }

    /// Search the index for matching chunks.
    pub fn search(&self, query: &str, k: usize) -> Vec<DocSearchResult> {
        if self.chunks.is_empty() {
            return Vec::new();
        }

        let results = self.engine.search(query, k);
        results
            .into_iter()
            .filter_map(|r| {
                self.chunks
                    .iter()
                    .find(|c| c.id == r.document.id)
                    .map(|c| DocSearchResult {
                        chunk: c.clone(),
                        score: r.score,
                    })
            })
            .collect()
    }

    /// Search and group results by document.
    pub fn search_by_doc(&self, query: &str, k: usize) -> Vec<(String, Vec<DocSearchResult>)> {
        let results = self.search(query, k * 3); // Get more results for grouping

        // Group by document
        let mut by_doc: std::collections::HashMap<String, Vec<DocSearchResult>> =
            std::collections::HashMap::new();
        for result in results {
            by_doc
                .entry(result.chunk.doc_id.clone())
                .or_default()
                .push(result);
        }

        // Sort documents by their best match score
        let mut docs: Vec<_> = by_doc.into_iter().collect();
        docs.sort_by(|a, b| {
            let best_a = a.1.iter().map(|r| r.score).fold(0.0f32, f32::max);
            let best_b = b.1.iter().map(|r| r.score).fold(0.0f32, f32::max);
            best_b
                .partial_cmp(&best_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to k documents
        docs.truncate(k);
        docs
    }

    /// Get the number of indexed chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Get the number of indexed documents.
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    /// Get a chunk by ID.
    pub fn get_chunk(&self, id: u32) -> Option<&DocChunk> {
        self.chunks.iter().find(|c| c.id == id)
    }

    /// Get all chunks for a specific document.
    pub fn chunks_for_doc(&self, doc_id: &str) -> Vec<&DocChunk> {
        self.chunks.iter().filter(|c| c.doc_id == doc_id).collect()
    }

    /// Get the index configuration.
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_content_with_overlap() {
        let content = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10";
        let chunks = chunk_content(content, 20, 5);

        assert!(!chunks.is_empty());
        // Verify chunks have IDs
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.id, i as u32);
        }
    }

    #[test]
    fn empty_content_produces_no_chunks() {
        let chunks = chunk_content("", 100, 10);
        assert!(chunks.is_empty());
    }

    #[test]
    fn small_content_produces_single_chunk() {
        let content = "hello world";
        let chunks = chunk_content(content, 1000, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, content);
    }

    #[test]
    fn index_searches_content() {
        let content = "The quick brown fox jumps over the lazy dog. \
                       The dog was not amused by the fox's antics. \
                       Meanwhile, the cat watched from the window.";
        let config = IndexConfig {
            enabled: true,
            chunk_size: 50,
            chunk_overlap: 10,
        };
        let index = Bm25Index::from_content(content, config);

        assert!(!index.is_empty());

        let results = index.search("fox", 5);
        assert!(!results.is_empty());
        // At least one result should contain "fox"
        assert!(results.iter().any(|r| r.chunk.content.contains("fox")));
    }

    #[test]
    fn search_empty_index_returns_empty() {
        let config = IndexConfig::default();
        let index = Bm25Index::new(config);
        let results = index.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn search_no_match_returns_empty() {
        let content = "hello world";
        let config = IndexConfig::default();
        let index = Bm25Index::from_content(content, config);
        let results = index.search("xyznonexistent123", 10);
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn doc_tree_index_searches_multiple_docs() {
        use crate::context::ContextSource;
        use crate::context::ContextStore;
        use crate::context::DocTreeStore;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // Create test documents
        std::fs::create_dir_all(root.join("api")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "# Root\n- [API](api/AGENTS.md)\n").unwrap();
        std::fs::write(
            root.join("api/AGENTS.md"),
            "# API\n- [auth](auth.md) - Authentication docs\n",
        )
        .unwrap();
        std::fs::write(
            root.join("api/auth.md"),
            "# Authentication\n\nUse OAuth2 for authentication. Configure client credentials.",
        )
        .unwrap();
        std::fs::write(
            root.join("api/users.md"),
            "# Users API\n\nThe users endpoint returns user data. No authentication required for public profiles.",
        )
        .unwrap();

        // Load doc tree
        let mut store = DocTreeStore::new();
        store
            .load(ContextSource::DocTree(root.to_path_buf()))
            .await
            .unwrap();

        // Build index
        let config = IndexConfig {
            enabled: true,
            chunk_size: 100,
            chunk_overlap: 20,
        };
        let index = DocTreeIndex::from_doc_tree(&store, config);

        assert_eq!(index.doc_count(), 4);
        assert!(!index.is_empty());

        // Search for authentication-related content
        let results = index.search("authentication OAuth2", 5);
        assert!(!results.is_empty());
        // Auth doc should be in results
        assert!(results.iter().any(|r| r.chunk.doc_id == "api/auth.md"));

        // Search by doc
        let by_doc = index.search_by_doc("authentication", 3);
        assert!(!by_doc.is_empty());
        // First result should be auth doc (best match)
        assert!(by_doc.iter().any(|(doc_id, _)| doc_id == "api/auth.md"));
    }
}
