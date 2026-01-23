use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use ignore::WalkBuilder;
use serde::Deserialize;
use serde::Serialize;

/// Maximum file size to load (10MB).
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;

/// Maximum total context size (100MB).
const MAX_TOTAL_SIZE: u64 = 100 * 1024 * 1024;

/// Maximum number of files to load.
const MAX_FILES: usize = 10_000;

#[async_trait]
pub trait ContextStore: Send + Sync {
    async fn load(&mut self, source: ContextSource) -> Result<()>;
    fn size(&self) -> ContextSize;
    fn fetch(&self, start: usize, end: usize) -> Result<&str>;
    fn content(&self) -> Result<&str>;
    fn fetch_doc(&self, _doc_id: &str, _start: usize, _end: usize) -> Result<&str> {
        bail!("document access not supported")
    }
    fn metadata(&self) -> &ContextMetadata;
}

#[derive(Clone, Debug)]
pub enum ContextSource {
    /// Single string content.
    String(String),
    /// Single file.
    File(PathBuf),
    /// Directory tree of documents (typically with AGENTS.md routing).
    DocTree(PathBuf),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContextSize {
    pub chars: usize,
    pub tokens_estimate: usize,
    pub documents: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ContextMetadata {
    pub size: ContextSize,
    pub documents: Vec<DocumentMetadata>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document ID (relative path).
    pub id: String,
    /// Relative path to the file (same as id).
    pub path: String,
    /// Size of the document content in bytes.
    pub size: usize,
    /// Start offset in the combined content.
    pub start: usize,
    /// End offset in the combined content.
    pub end: usize,
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryStore {
    content: String,
    metadata: ContextMetadata,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn set_content(&mut self, content: String) {
        let bytes = content.len();
        let chars = content.chars().count();
        let tokens_estimate = estimate_tokens(&content);
        self.metadata = ContextMetadata {
            size: ContextSize {
                chars,
                tokens_estimate,
                documents: 1,
                bytes,
            },
            documents: Vec::new(),
        };
        self.content = content;
    }
}

#[async_trait]
impl ContextStore for InMemoryStore {
    async fn load(&mut self, source: ContextSource) -> Result<()> {
        match source {
            ContextSource::String(content) => {
                self.set_content(content);
                Ok(())
            }
            ContextSource::File(path) => {
                let content = std::fs::read_to_string(&path)?;
                self.set_content(content);
                Ok(())
            }
            ContextSource::DocTree(_) => {
                bail!(
                    "in-memory store only supports String and File sources, use DocTreeStore for directories"
                )
            }
        }
    }

    fn size(&self) -> ContextSize {
        self.metadata.size.clone()
    }

    fn fetch(&self, start: usize, end: usize) -> Result<&str> {
        if start > end || end > self.content.len() {
            bail!("invalid span {start}-{end}");
        }
        Ok(&self.content[start..end])
    }

    fn content(&self) -> Result<&str> {
        Ok(&self.content)
    }

    fn metadata(&self) -> &ContextMetadata {
        &self.metadata
    }
}

#[derive(Debug, Default)]
pub struct MemoryMappedStore {
    mmap: Option<memmap2::Mmap>,
    metadata: ContextMetadata,
}

impl MemoryMappedStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn set_mmap(&mut self, mmap: memmap2::Mmap) -> Result<()> {
        let content = std::str::from_utf8(&mmap)
            .map_err(|err| anyhow::anyhow!("context is not valid UTF-8: {err}"))?;
        let bytes = mmap.len();
        let chars = content.chars().count();
        let tokens_estimate = estimate_tokens(content);
        self.metadata = ContextMetadata {
            size: ContextSize {
                chars,
                tokens_estimate,
                documents: 1,
                bytes,
            },
            documents: Vec::new(),
        };
        self.mmap = Some(mmap);
        Ok(())
    }

    fn content_str(&self) -> Result<&str> {
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("context not loaded"))?;
        std::str::from_utf8(mmap)
            .map_err(|err| anyhow::anyhow!("context is not valid UTF-8: {err}"))
    }
}

#[async_trait]
impl ContextStore for MemoryMappedStore {
    async fn load(&mut self, source: ContextSource) -> Result<()> {
        match source {
            ContextSource::String(_) => {
                bail!("memory-mapped store only supports file sources")
            }
            ContextSource::File(path) => {
                let file = std::fs::File::open(&path)?;
                let mmap = unsafe { memmap2::Mmap::map(&file)? };
                self.set_mmap(mmap)?;
                Ok(())
            }
            ContextSource::DocTree(_) => {
                bail!(
                    "memory-mapped store only supports file sources, use DocTreeStore for directories"
                )
            }
        }
    }

    fn size(&self) -> ContextSize {
        self.metadata.size.clone()
    }

    fn fetch(&self, start: usize, end: usize) -> Result<&str> {
        let content = self.content_str()?;
        if start > end || end > content.len() {
            bail!("invalid span {start}-{end}");
        }
        Ok(&content[start..end])
    }

    fn content(&self) -> Result<&str> {
        self.content_str()
    }

    fn metadata(&self) -> &ContextMetadata {
        &self.metadata
    }
}

#[derive(Debug)]
pub enum ContextStoreKind {
    InMemory(InMemoryStore),
    MemoryMapped(MemoryMappedStore),
    DocTree(DocTreeStore),
}

impl Default for ContextStoreKind {
    fn default() -> Self {
        Self::InMemory(InMemoryStore::new())
    }
}

impl ContextStoreKind {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the DocTreeStore if this is a DocTree context.
    pub fn as_doc_tree(&self) -> Option<&DocTreeStore> {
        match self {
            Self::DocTree(store) => Some(store),
            _ => None,
        }
    }
}

#[async_trait]
impl ContextStore for ContextStoreKind {
    async fn load(&mut self, source: ContextSource) -> Result<()> {
        match source {
            ContextSource::String(content) => {
                let mut store = InMemoryStore::new();
                store.load(ContextSource::String(content)).await?;
                *self = Self::InMemory(store);
                Ok(())
            }
            ContextSource::File(path) => {
                let mut store = MemoryMappedStore::new();
                store.load(ContextSource::File(path)).await?;
                *self = Self::MemoryMapped(store);
                Ok(())
            }
            ContextSource::DocTree(path) => {
                let mut store = DocTreeStore::new();
                store.load(ContextSource::DocTree(path)).await?;
                *self = Self::DocTree(store);
                Ok(())
            }
        }
    }

    fn size(&self) -> ContextSize {
        match self {
            Self::InMemory(store) => store.size(),
            Self::MemoryMapped(store) => store.size(),
            Self::DocTree(store) => store.size(),
        }
    }

    fn fetch(&self, start: usize, end: usize) -> Result<&str> {
        match self {
            Self::InMemory(store) => store.fetch(start, end),
            Self::MemoryMapped(store) => store.fetch(start, end),
            Self::DocTree(store) => store.fetch(start, end),
        }
    }

    fn content(&self) -> Result<&str> {
        match self {
            Self::InMemory(store) => store.content(),
            Self::MemoryMapped(store) => store.content(),
            Self::DocTree(store) => store.content(),
        }
    }

    fn fetch_doc(&self, doc_id: &str, start: usize, end: usize) -> Result<&str> {
        match self {
            Self::DocTree(store) => store.fetch_doc(doc_id, start, end),
            _ => bail!("document access not supported for this store type"),
        }
    }

    fn metadata(&self) -> &ContextMetadata {
        match self {
            Self::InMemory(store) => store.metadata(),
            Self::MemoryMapped(store) => store.metadata(),
            Self::DocTree(store) => store.metadata(),
        }
    }
}

/// Approximate bytes per token for modern LLMs.
const APPROX_BYTES_PER_TOKEN: usize = 4;

/// Estimate token count from text using byte-based approximation.
fn estimate_tokens(content: &str) -> usize {
    let bytes = content.len();
    bytes.saturating_add(APPROX_BYTES_PER_TOKEN.saturating_sub(1)) / APPROX_BYTES_PER_TOKEN
}

/// Check if a file is a markdown document.
pub fn is_markdown_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext == "md")
}

/// Check if a file appears to be a text file by sampling the first chunk.
/// Returns false for binary files (containing null bytes) or non-UTF-8 files.
fn is_text_file(path: &Path) -> bool {
    // Check file size first
    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return false,
    };

    if metadata.len() > MAX_FILE_SIZE {
        return false;
    }

    // Read the first 8KB to check for binary content
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    use std::io::Read;
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = [0u8; 8192];
    let bytes_read = match reader.read(&mut buffer) {
        Ok(n) => n,
        Err(_) => return false,
    };

    // Check for null bytes (binary indicator)
    if buffer[..bytes_read].contains(&0) {
        return false;
    }

    // Check if it's valid UTF-8
    std::str::from_utf8(&buffer[..bytes_read]).is_ok()
}

/// A document in the doc tree (content accessed via DocTreeStore).
#[derive(Clone, Debug)]
pub struct Document {
    /// Relative path from the root (used as ID).
    pub id: String,
    /// Absolute path to the file.
    pub path: PathBuf,
    /// Byte offset to the start of the actual content in combined content (after separator).
    pub content_offset: usize,
    /// Size of the content in bytes.
    pub size: usize,
    /// Whether this is an AGENTS.md file (routing file).
    pub is_agents_md: bool,
}

/// Store for a tree of documents.
#[derive(Debug, Default)]
pub struct DocTreeStore {
    /// Root path of the doc tree.
    root: PathBuf,
    /// All documents indexed by relative path.
    documents: std::collections::HashMap<String, Document>,
    /// Combined content (all docs concatenated with separators).
    combined_content: String,
    /// Metadata about the context.
    metadata: ContextMetadata,
    /// List of AGENTS.md files for routing.
    agents_files: Vec<String>,
}

impl DocTreeStore {
    /// Create a new empty doc tree store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the root path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Get a document by ID (relative path).
    pub fn get_document(&self, id: &str) -> Option<&Document> {
        self.documents.get(id)
    }

    /// Get the content of a document by ID.
    ///
    /// This extracts the content from the combined_content using offsets.
    pub fn document_content(&self, id: &str) -> Option<&str> {
        let doc = self.documents.get(id)?;
        self.combined_content
            .get(doc.content_offset..doc.content_offset + doc.size)
    }

    /// List all document IDs.
    pub fn list_documents(&self) -> Vec<&str> {
        self.documents
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }

    /// List all AGENTS.md files.
    pub fn list_agents_files(&self) -> &[String] {
        &self.agents_files
    }

    /// List documents under a path prefix.
    pub fn list_documents_under(&self, prefix: &str) -> Vec<&str> {
        self.documents
            .keys()
            .filter(|k| k.starts_with(prefix))
            .map(std::string::String::as_str)
            .collect()
    }

    /// Load a directory tree of documents.
    async fn load_tree(&mut self, root: PathBuf) -> Result<()> {
        self.root = root.clone();
        self.documents.clear();
        self.agents_files.clear();

        let mut docs = Vec::new();
        let mut total_size: u64 = 0;

        // Use ignore crate for gitignore-aware walking
        let walker = WalkBuilder::new(&root)
            .hidden(false) // Don't skip hidden files by default
            .git_ignore(true) // Respect .gitignore
            .git_global(true) // Respect global gitignore
            .git_exclude(true) // Respect .git/info/exclude
            .build();

        for entry in walker.flatten() {
            let path = entry.path();

            // Skip directories
            if path.is_dir() {
                continue;
            }

            // Check file count limit
            if docs.len() >= MAX_FILES {
                tracing::warn!(
                    "Reached maximum file count ({MAX_FILES}), skipping remaining files"
                );
                break;
            }

            // Check if it's a text file (also checks file size limit internally)
            if !is_text_file(path) {
                continue;
            }

            let rel_path = path
                .strip_prefix(&root)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());

            docs.push((rel_path, path.to_path_buf()));
        }

        // Sort for deterministic order
        docs.sort_by(|a, b| a.0.cmp(&b.0));

        // Build combined content and index documents
        let mut combined = String::new();
        let mut doc_metadata = Vec::new();

        for (rel_path, abs_path) in docs {
            let content = match std::fs::read_to_string(&abs_path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("Failed to read {}: {}", abs_path.display(), e);
                    continue;
                }
            };

            // Check total size limit
            let content_size = content.len() as u64;
            if total_size + content_size > MAX_TOTAL_SIZE {
                tracing::warn!(
                    "Reached maximum total size ({MAX_TOTAL_SIZE} bytes), skipping remaining files"
                );
                break;
            }
            total_size += content_size;

            let is_agents_md = rel_path.ends_with("AGENTS.md");
            if is_agents_md {
                self.agents_files.push(rel_path.clone());
            }

            let size = content.len();

            // Add separator and content per spec: \n===== {relative_path} =====\n
            if !combined.is_empty() {
                combined.push('\n');
            }
            let separator = format!("===== {rel_path} =====\n");
            combined.push_str(&separator);
            let content_offset = combined.len(); // After separator
            combined.push_str(&content);

            let doc = Document {
                id: rel_path.clone(),
                path: abs_path,
                content_offset,
                size,
                is_agents_md,
            };

            doc_metadata.push(DocumentMetadata {
                id: rel_path.clone(),
                path: rel_path.clone(), // Use relative path per spec
                size,
                start: content_offset,
                end: content_offset + size,
            });

            self.documents.insert(rel_path, doc);
        }

        self.combined_content = combined;

        // Build metadata
        let bytes = self.combined_content.len();
        let chars = self.combined_content.chars().count();
        let tokens_estimate = estimate_tokens(&self.combined_content);

        self.metadata = ContextMetadata {
            size: ContextSize {
                chars,
                tokens_estimate,
                documents: self.documents.len(),
                bytes,
            },
            documents: doc_metadata,
        };

        Ok(())
    }
}

#[async_trait]
impl ContextStore for DocTreeStore {
    async fn load(&mut self, source: ContextSource) -> Result<()> {
        match source {
            ContextSource::DocTree(path) => self.load_tree(path).await,
            _ => bail!("DocTreeStore only supports DocTree sources"),
        }
    }

    fn size(&self) -> ContextSize {
        self.metadata.size.clone()
    }

    fn fetch(&self, start: usize, end: usize) -> Result<&str> {
        if start > end || end > self.combined_content.len() {
            bail!("invalid span {start}-{end}");
        }
        Ok(&self.combined_content[start..end])
    }

    fn content(&self) -> Result<&str> {
        Ok(&self.combined_content)
    }

    fn fetch_doc(&self, doc_id: &str, start: usize, end: usize) -> Result<&str> {
        let doc = self
            .documents
            .get(doc_id)
            .ok_or_else(|| anyhow::anyhow!("document not found: {doc_id}"))?;

        if start > end || end > doc.size {
            bail!("invalid span {start}-{end} for doc {doc_id}");
        }

        // Calculate absolute offset in combined_content
        let abs_start = doc.content_offset + start;
        let abs_end = doc.content_offset + end;
        Ok(&self.combined_content[abs_start..abs_end])
    }

    fn metadata(&self) -> &ContextMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn memory_mapped_store_loads_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("context.txt");
        std::fs::write(&path, "hello world").unwrap();

        let mut store = MemoryMappedStore::new();
        store.load(ContextSource::File(path)).await.unwrap();
        assert_eq!(store.size().chars, 11);
        assert_eq!(store.fetch(0, 5).unwrap(), "hello");
        assert_eq!(store.content().unwrap(), "hello world");
    }

    #[tokio::test]
    async fn doc_tree_store_loads_directory() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // Create a simple doc tree with various text files
        std::fs::create_dir_all(root.join("docs")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "# Root\n- [docs](docs/AGENTS.md)").unwrap();
        std::fs::write(root.join("docs/AGENTS.md"), "# Docs\n- [readme](readme.md)").unwrap();
        std::fs::write(root.join("docs/readme.md"), "# Readme\nHello world").unwrap();
        std::fs::write(root.join("docs/code.rs"), "fn main() {}").unwrap();

        let mut store = DocTreeStore::new();
        store
            .load(ContextSource::DocTree(root.to_path_buf()))
            .await
            .unwrap();

        // Should load all text files (md and rs)
        assert_eq!(store.metadata().size.documents, 4);
        assert_eq!(store.list_agents_files().len(), 2);

        // Can fetch individual doc
        let content = store.fetch_doc("docs/readme.md", 0, 10).unwrap();
        assert_eq!(content, "# Readme\nH");

        // Can list docs under path
        let docs = store.list_documents_under("docs/");
        assert_eq!(docs.len(), 3);
    }

    #[tokio::test]
    async fn doc_tree_respects_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // Initialize as a git repository (required for .gitignore to be respected)
        std::fs::create_dir(root.join(".git")).unwrap();
        // Create .gitignore to ignore node_modules
        std::fs::write(root.join(".gitignore"), "node_modules/\n").unwrap();
        std::fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "# Root").unwrap();
        std::fs::write(root.join("node_modules/pkg/readme.md"), "# Skip me").unwrap();

        let mut store = DocTreeStore::new();
        store
            .load(ContextSource::DocTree(root.to_path_buf()))
            .await
            .unwrap();

        // Should only have AGENTS.md and .gitignore (text files), not node_modules
        // Note: .gitignore itself is also a text file
        let doc_ids: Vec<_> = store.list_documents();
        assert!(
            !doc_ids.contains(&"node_modules/pkg/readme.md"),
            "should not include gitignored files"
        );
        assert!(doc_ids.contains(&"AGENTS.md"));
    }

    #[tokio::test]
    async fn doc_tree_skips_binary_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        std::fs::write(root.join("readme.md"), "# Hello").unwrap();
        // Create a binary file (containing null bytes)
        std::fs::write(root.join("binary.bin"), b"\x00\x01\x02\x03").unwrap();

        let mut store = DocTreeStore::new();
        store
            .load(ContextSource::DocTree(root.to_path_buf()))
            .await
            .unwrap();

        let doc_ids: Vec<_> = store.list_documents();
        assert!(doc_ids.contains(&"readme.md"));
        assert!(
            !doc_ids.contains(&"binary.bin"),
            "should not include binary files"
        );
    }

    #[tokio::test]
    async fn doc_tree_uses_spec_separator() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        std::fs::write(root.join("a.txt"), "content a").unwrap();
        std::fs::write(root.join("b.txt"), "content b").unwrap();

        let mut store = DocTreeStore::new();
        store
            .load(ContextSource::DocTree(root.to_path_buf()))
            .await
            .unwrap();

        let content = store.content().unwrap();
        // Spec requires: \n===== {relative_path} =====\n
        assert!(
            content.contains("===== a.txt ====="),
            "should use 5-equals separator"
        );
        assert!(
            content.contains("===== b.txt ====="),
            "should use 5-equals separator"
        );
    }
}
