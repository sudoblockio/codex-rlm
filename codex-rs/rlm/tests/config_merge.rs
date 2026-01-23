use codex_rlm::RlmConfig;
use pretty_assertions::assert_eq;
use tempfile::TempDir;

#[test]
fn codex_config_merges_model_providers_and_rlm_section() {
    let temp = TempDir::new().expect("tempdir");
    let config_path = temp.path().join("config.toml");
    let toml = r#"
model_provider = "mock"

[model_providers.mock]
name = "mock"
base_url = "http://localhost:11434/v1"
env_key = "MOCK_API_KEY"
wire_api = "chat"

[rlm.gateway]
model = "gpt-test"
max_output_tokens = 42

[rlm.prompt]
template = "Q: {query}\nR: {routing_summary}"
"#;
    std::fs::write(&config_path, toml).expect("write config");

    let mut config = RlmConfig::default();
    config
        .apply_codex_config_from_path(&config_path, true, true)
        .expect("apply codex config");

    assert_eq!(config.gateway.provider, "mock");
    assert_eq!(
        config.gateway.base_url.as_deref(),
        Some("http://localhost:11434/v1")
    );
    assert_eq!(config.gateway.model, "gpt-test");
    assert_eq!(config.gateway.max_output_tokens, 42);
    assert_eq!(config.prompt.template, "Q: {query}\nR: {routing_summary}");
}
