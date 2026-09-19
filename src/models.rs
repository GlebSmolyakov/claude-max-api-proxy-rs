//! Which model names the proxy accepts and what it passes to `claude --model`.

/// CLI aliases. The CLI resolves each one to the newest model of that family.
pub const ALIASES: [&str; 4] = ["fable", "opus", "sonnet", "haiku"];

/// Used when a request names no model.
pub const DEFAULT_MODEL: &str = "opus";

/// Map a requested model name to a `--model` value.
///
/// Aliases pass through, as do full Claude model ids such as
/// `claude-sonnet-5` or `claude-opus-5[1m]`; the CLI checks those itself.
/// Anything else is refused instead of silently falling back to a model the
/// client did not ask for.
pub fn resolve(requested: Option<&str>) -> Result<String, String> {
    let Some(name) = requested.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(DEFAULT_MODEL.to_string());
    };
    let name = name.strip_prefix("claude-code-cli/").unwrap_or(name);

    if ALIASES.contains(&name) {
        return Ok(name.to_string());
    }

    let plausible_id = name.starts_with("claude-")
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '[' | ']'));
    if plausible_id {
        return Ok(name.to_string());
    }

    Err(format!(
        "Unknown model '{name}'. Use an alias ({}) or a full Claude model id such as claude-sonnet-5.",
        ALIASES.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_pass_through() {
        for alias in ALIASES {
            assert_eq!(resolve(Some(alias)).unwrap(), alias);
        }
    }

    #[test]
    fn missing_or_blank_model_uses_default() {
        assert_eq!(resolve(None).unwrap(), DEFAULT_MODEL);
        assert_eq!(resolve(Some("  ")).unwrap(), DEFAULT_MODEL);
    }

    #[test]
    fn provider_prefix_is_stripped() {
        assert_eq!(resolve(Some("claude-code-cli/sonnet")).unwrap(), "sonnet");
        assert_eq!(resolve(Some("claude-code-cli/claude-sonnet-5")).unwrap(), "claude-sonnet-5");
    }

    #[test]
    fn full_model_ids_pass_through_unchanged() {
        for id in ["claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-opus-5[1m]", "claude-fable-5-1"] {
            assert_eq!(resolve(Some(id)).unwrap(), id);
        }
    }

    #[test]
    fn unknown_models_are_refused() {
        for name in ["gpt-4o", "claude-x --verbose", "llama3"] {
            let err = resolve(Some(name)).unwrap_err();
            assert!(err.contains("Unknown model"), "{name}: {err}");
        }
    }
}
