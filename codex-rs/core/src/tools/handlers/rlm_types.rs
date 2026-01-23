use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use serde_json::Value;

use crate::protocol::SandboxPolicy;
use crate::tools::context::ToolOutput;

pub(crate) fn json_tool_output(value: Value, success: bool) -> ToolOutput {
    let content = serde_json::to_string(&value).unwrap_or_else(|err| {
        format!(
            "{{\"success\":false,\"error_code\":\"serialization_error\",\"error_message\":\"{err}\"}}"
        )
    });
    ToolOutput::Function {
        content,
        content_items: None,
        success: Some(success),
    }
}

pub(crate) fn error_value(
    code: &str,
    message: impl Into<String>,
    suggestion: Option<&str>,
) -> Value {
    let mut map = serde_json::Map::new();
    map.insert("success".to_string(), Value::Bool(false));
    map.insert("error_code".to_string(), Value::String(code.to_string()));
    map.insert("error_message".to_string(), Value::String(message.into()));
    if let Some(suggestion) = suggestion {
        map.insert(
            "suggestion".to_string(),
            Value::String(suggestion.to_string()),
        );
    }
    Value::Object(map)
}

pub(crate) fn validate_existing_absolute_path(path: &Path) -> Result<(), Value> {
    if !path.is_absolute() {
        return Err(error_value(
            "path_outside_sandbox",
            "path must be absolute",
            Some("Use an absolute path within the workspace"),
        ));
    }

    if !path.exists() {
        let path_display = path.display();
        return Err(error_value(
            "path_not_found",
            format!("path does not exist: {path_display}"),
            Some("Check the path and try again"),
        ));
    }

    Ok(())
}

/// Validate that a path exists, is absolute, and is within the sandbox roots.
///
/// This performs security-critical validation to ensure RLM tools only access
/// files within the allowed sandbox boundaries.
pub(crate) fn validate_path_in_sandbox(
    path: &Path,
    sandbox_policy: &SandboxPolicy,
    cwd: &Path,
) -> Result<(), Value> {
    // First check basic path requirements
    validate_existing_absolute_path(path)?;

    // Resolve symlinks and canonicalize the path to prevent symlink escapes
    let canonical_path = match path.canonicalize() {
        Ok(p) => p,
        Err(err) => {
            return Err(error_value(
                "path_resolution_error",
                format!("failed to resolve path: {err}"),
                Some("Ensure the path exists and is accessible"),
            ));
        }
    };

    // Check sandbox policy
    match sandbox_policy {
        SandboxPolicy::DangerFullAccess | SandboxPolicy::ExternalSandbox { .. } => {
            // Full access mode - no sandbox restrictions
            Ok(())
        }
        SandboxPolicy::ReadOnly => {
            // RLM tools need read access, which ReadOnly allows
            // The sandbox will enforce this at the OS level
            Ok(())
        }
        SandboxPolicy::WorkspaceWrite { .. } => {
            // Get the writable roots for this policy
            let writable_roots = sandbox_policy.get_writable_roots_with_cwd(cwd);

            // Normalize the canonical path
            let normalized = normalize_path(&canonical_path);

            // Check if path is within any writable root (which also implies readable)
            // For RLM, we're loading content (reading), but we check writable roots
            // because those define the workspace boundaries.
            // We canonicalize the writable roots to handle symlinks like /var -> /private/var on macOS.
            let is_within_workspace = writable_roots.iter().any(|root| {
                let canonical_root = root
                    .root
                    .as_path()
                    .canonicalize()
                    .unwrap_or_else(|_| root.root.to_path_buf());
                let canonical_root_normalized = normalize_path(&canonical_root);
                normalized.starts_with(&canonical_root_normalized)
            });

            if is_within_workspace {
                Ok(())
            } else {
                let path_display = path.display();
                Err(error_value(
                    "path_outside_sandbox",
                    format!("path is outside sandbox boundaries: {path_display}"),
                    Some("Use a path within the workspace or configured writable roots"),
                ))
            }
        }
    }
}

/// Normalize a path by removing `.` and resolving `..` without touching the filesystem.
fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => { /* skip */ }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_utils_absolute_path::AbsolutePathBuf;
    use pretty_assertions::assert_eq;
    use std::fs;

    #[test]
    fn validate_existing_absolute_path_accepts_existing_absolute() {
        let file = tempfile::NamedTempFile::new().unwrap();
        assert!(validate_existing_absolute_path(file.path()).is_ok());
    }

    #[test]
    fn validate_existing_absolute_path_rejects_relative_path() {
        let path = Path::new("relative/path");
        let err = validate_existing_absolute_path(path).unwrap_err();
        assert_eq!(err["error_code"], "path_outside_sandbox");
    }

    #[test]
    fn validate_existing_absolute_path_rejects_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.txt");
        let err = validate_existing_absolute_path(&path).unwrap_err();
        assert_eq!(err["error_code"], "path_not_found");
    }

    // Sandbox validation tests

    #[test]
    fn sandbox_allows_path_within_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path();
        let file_path = cwd.join("test.txt");
        fs::write(&file_path, "content").unwrap();

        let policy = SandboxPolicy::WorkspaceWrite {
            writable_roots: vec![],
            network_access: false,
            exclude_tmpdir_env_var: true,
            exclude_slash_tmp: true,
        };

        assert!(validate_path_in_sandbox(&file_path, &policy, cwd).is_ok());
    }

    #[test]
    fn sandbox_allows_path_in_writable_root() {
        let workspace = tempfile::tempdir().unwrap();
        let extra_root = tempfile::tempdir().unwrap();
        let file_path = extra_root.path().join("allowed.txt");
        fs::write(&file_path, "content").unwrap();

        let policy = SandboxPolicy::WorkspaceWrite {
            writable_roots: vec![AbsolutePathBuf::try_from(extra_root.path()).unwrap()],
            network_access: false,
            exclude_tmpdir_env_var: true,
            exclude_slash_tmp: true,
        };

        assert!(validate_path_in_sandbox(&file_path, &policy, workspace.path()).is_ok());
    }

    #[test]
    fn sandbox_rejects_path_outside_workspace() {
        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let file_path = outside.path().join("outside.txt");
        fs::write(&file_path, "content").unwrap();

        let policy = SandboxPolicy::WorkspaceWrite {
            writable_roots: vec![],
            network_access: false,
            exclude_tmpdir_env_var: true,
            exclude_slash_tmp: true,
        };

        let err = validate_path_in_sandbox(&file_path, &policy, workspace.path()).unwrap_err();
        assert_eq!(err["error_code"], "path_outside_sandbox");
    }

    #[test]
    fn sandbox_danger_full_access_allows_anything() {
        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let file_path = outside.path().join("outside.txt");
        fs::write(&file_path, "content").unwrap();

        let policy = SandboxPolicy::DangerFullAccess;

        // Should allow paths outside workspace
        assert!(validate_path_in_sandbox(&file_path, &policy, workspace.path()).is_ok());
    }

    #[test]
    fn sandbox_read_only_allows_reads() {
        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let file_path = outside.path().join("outside.txt");
        fs::write(&file_path, "content").unwrap();

        let policy = SandboxPolicy::ReadOnly;

        // ReadOnly policy allows reads anywhere (OS sandbox enforces)
        assert!(validate_path_in_sandbox(&file_path, &policy, workspace.path()).is_ok());
    }

    #[test]
    #[cfg(unix)]
    fn sandbox_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();

        // Create a file outside the workspace
        let secret_file = outside.path().join("secret.txt");
        fs::write(&secret_file, "secret content").unwrap();

        // Create a symlink inside workspace that points outside
        let symlink_path = workspace.path().join("sneaky_link");
        symlink(&secret_file, &symlink_path).unwrap();

        let policy = SandboxPolicy::WorkspaceWrite {
            writable_roots: vec![],
            network_access: false,
            exclude_tmpdir_env_var: true,
            exclude_slash_tmp: true,
        };

        // The symlink target is outside the workspace, so this should fail
        let err = validate_path_in_sandbox(&symlink_path, &policy, workspace.path()).unwrap_err();
        assert_eq!(err["error_code"], "path_outside_sandbox");
    }

    #[test]
    fn sandbox_rejects_path_traversal_attempt() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().join("workspace");
        let sibling = tmp.path().join("sibling");
        fs::create_dir_all(&workspace).unwrap();
        fs::create_dir_all(&sibling).unwrap();

        let file_path = sibling.join("file.txt");
        fs::write(&file_path, "content").unwrap();

        let policy = SandboxPolicy::WorkspaceWrite {
            writable_roots: vec![],
            network_access: false,
            exclude_tmpdir_env_var: true,
            exclude_slash_tmp: true,
        };

        // Sibling directory is outside workspace
        let err = validate_path_in_sandbox(&file_path, &policy, &workspace).unwrap_err();
        assert_eq!(err["error_code"], "path_outside_sandbox");
    }

    #[test]
    fn normalize_path_handles_dot_components() {
        let path = PathBuf::from("/a/b/./c/../d");
        let normalized = normalize_path(&path);
        assert_eq!(normalized, PathBuf::from("/a/b/d"));
    }

    #[test]
    fn normalize_path_handles_multiple_parent_refs() {
        let path = PathBuf::from("/a/b/c/../../d");
        let normalized = normalize_path(&path);
        assert_eq!(normalized, PathBuf::from("/a/d"));
    }
}
