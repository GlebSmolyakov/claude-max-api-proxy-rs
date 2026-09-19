//! Maps conversation prefixes to saved CLI sessions.
//!
//! After every successful turn the store remembers which CLI session now
//! holds the conversation including the reply (`Conversation::key_after_reply`).
//! The client's next request carries that same history, which hashes to the
//! same key, so the turn resumes the session and sends only the new message.
//! Every resume forks (`--fork-session`), so regenerating an earlier reply
//! branches off cleanly instead of appending to a session that moved on.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, warn};

const SESSION_TTL_SECS: u64 = 24 * 60 * 60;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    session_id: String,
    created_at: u64,
    last_used_at: u64,
}

#[derive(Clone)]
pub struct SessionStore {
    entries: Arc<RwLock<HashMap<String, Entry>>>,
    file_path: PathBuf,
    /// Where the CLI keeps transcripts for the proxy's working directory.
    transcripts_dir: PathBuf,
    save_lock: Arc<Mutex<()>>,
}

impl SessionStore {
    pub async fn open(file_path: PathBuf, transcripts_dir: PathBuf) -> Self {
        let entries = match tokio::fs::read_to_string(&file_path).await {
            Ok(data) => serde_json::from_str(&data).unwrap_or_else(|e| {
                warn!("Ignoring unreadable sessions file {}: {e}", file_path.display());
                HashMap::new()
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
            Err(e) => {
                warn!("Could not read sessions file {}: {e}", file_path.display());
                HashMap::new()
            }
        };
        if !entries.is_empty() {
            info!("Loaded {} saved sessions from {}", entries.len(), file_path.display());
        }
        Self {
            entries: Arc::new(RwLock::new(entries)),
            file_path,
            transcripts_dir,
            save_lock: Arc::new(Mutex::new(())),
        }
    }

    /// The session that holds the conversation up to `key`, if any.
    pub async fn lookup(&self, key: &str) -> Option<String> {
        let mut entries = self.entries.write().await;
        let entry = entries.get_mut(key)?;
        entry.last_used_at = crate::status::unix_now();
        Some(entry.session_id.clone())
    }

    pub async fn remember(&self, key: String, session_id: String) {
        let now = crate::status::unix_now();
        self.entries.write().await.insert(
            key,
            Entry {
                session_id,
                created_at: now,
                last_used_at: now,
            },
        );
        self.save().await;
    }

    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    /// Drop entries unused for a day and delete the CLI transcripts that no
    /// remaining entry points to. Returns how many entries were dropped.
    pub async fn cleanup_expired(&self) -> usize {
        self.cleanup_older_than(crate::status::unix_now().saturating_sub(SESSION_TTL_SECS))
            .await
    }

    async fn cleanup_older_than(&self, cutoff: u64) -> usize {
        let orphaned: Vec<String> = {
            let mut entries = self.entries.write().await;
            let mut dropped = Vec::new();
            entries.retain(|_, e| {
                let keep = e.last_used_at >= cutoff;
                if !keep {
                    dropped.push(e.session_id.clone());
                }
                keep
            });
            let alive: HashSet<&String> = entries.values().map(|e| &e.session_id).collect();
            dropped.retain(|id| !alive.contains(id));
            dropped
        };
        if orphaned.is_empty() {
            return 0;
        }

        let mut deleted = 0;
        for id in &orphaned {
            // Session ids come from our own file; only ever delete `<uuid>.jsonl`.
            if uuid::Uuid::parse_str(id).is_err() {
                continue;
            }
            match tokio::fs::remove_file(self.transcripts_dir.join(format!("{id}.jsonl"))).await {
                Ok(()) => deleted += 1,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => warn!("Could not delete transcript {id}: {e}"),
            }
        }
        info!(
            "Expired {} saved sessions, deleted {deleted} transcripts",
            orphaned.len()
        );
        self.save().await;
        orphaned.len()
    }

    pub fn spawn_cleanup_task(&self) {
        let store = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));
            loop {
                interval.tick().await;
                store.cleanup_expired().await;
            }
        });
    }

    /// Write the file atomically: a temp file renamed over the old one.
    async fn save(&self) {
        let _guard = self.save_lock.lock().await;
        let data = {
            let entries = self.entries.read().await;
            match serde_json::to_string_pretty(&*entries) {
                Ok(data) => data,
                Err(e) => {
                    error!("Failed to serialize sessions: {e}");
                    return;
                }
            }
        };
        let tmp = self.file_path.with_extension("json.tmp");
        if let Err(e) = tokio::fs::write(&tmp, data).await {
            error!("Failed to write {}: {e}", tmp.display());
            return;
        }
        if let Err(e) = tokio::fs::rename(&tmp, &self.file_path).await {
            error!("Failed to replace {}: {e}", self.file_path.display());
        }
    }
}

/// The directory where the CLI saves sessions started in `cwd`:
/// `<config dir>/projects/<cwd with every character other than an ASCII
/// letter, digit or '-' replaced by '-'>`. The config dir is
/// `$CLAUDE_CONFIG_DIR`, or `~/.claude` by default.
pub fn transcripts_dir_for(cwd: &Path) -> PathBuf {
    let config_dir = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join(".claude")
        });
    let slug: String = cwd
        .to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' })
        .collect();
    config_dir.join("projects").join(slug)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("claude-max-api-test-{name}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    async fn store(dir: &Path) -> SessionStore {
        SessionStore::open(dir.join("sessions.json"), dir.join("transcripts")).await
    }

    #[tokio::test]
    async fn remembers_and_finds_sessions() {
        let dir = temp_dir("lookup");
        let s = store(&dir).await;
        assert_eq!(s.lookup("k").await, None);
        s.remember("k".into(), "sid-1".into()).await;
        assert_eq!(s.lookup("k").await.as_deref(), Some("sid-1"));
        assert_eq!(s.len().await, 1);
    }

    #[tokio::test]
    async fn survives_a_restart() {
        let dir = temp_dir("persist");
        store(&dir).await.remember("k".into(), "sid-1".into()).await;
        let reopened = store(&dir).await;
        assert_eq!(reopened.lookup("k").await.as_deref(), Some("sid-1"));
    }

    #[tokio::test]
    async fn unreadable_file_starts_empty() {
        let dir = temp_dir("corrupt");
        std::fs::write(dir.join("sessions.json"), "not json").unwrap();
        assert_eq!(store(&dir).await.len().await, 0);
    }

    #[tokio::test]
    async fn cleanup_deletes_only_unreferenced_uuid_transcripts() {
        let dir = temp_dir("cleanup");
        let transcripts = dir.join("transcripts");
        std::fs::create_dir_all(&transcripts).unwrap();
        let old = uuid::Uuid::new_v4().to_string();
        let shared = uuid::Uuid::new_v4().to_string();
        for id in [&old, &shared] {
            std::fs::write(transcripts.join(format!("{id}.jsonl")), "{}").unwrap();
        }

        let s = store(&dir).await;
        s.remember("old".into(), old.clone()).await;
        s.remember("shared-old".into(), shared.clone()).await;
        s.remember("shared-new".into(), shared.clone()).await;
        s.remember("bogus".into(), "../escape".into()).await;
        {
            let mut entries = s.entries.write().await;
            for key in ["old", "shared-old", "bogus"] {
                entries.get_mut(key).unwrap().last_used_at = 0;
            }
        }

        assert_eq!(s.cleanup_older_than(1).await, 2, "old and bogus expire; shared is still referenced");
        assert!(!transcripts.join(format!("{old}.jsonl")).exists());
        assert!(transcripts.join(format!("{shared}.jsonl")).exists());
        assert_eq!(s.lookup("shared-new").await, Some(shared));
        assert_eq!(s.lookup("old").await, None);
    }

    #[test]
    fn transcripts_dir_uses_the_cli_slug() {
        let dir = transcripts_dir_for(Path::new("/Users/me/.claude-max-api/workdir"));
        assert!(dir.ends_with("projects/-Users-me--claude-max-api-workdir"), "{}", dir.display());
    }
}
