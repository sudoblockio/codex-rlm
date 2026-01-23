use std::path::Path;

use serde_json::Value;

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

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

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
}
