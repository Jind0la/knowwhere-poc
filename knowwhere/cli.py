"""knowwhere.cli — One-command install and management.

    pip install knowwhere
    knowwhere init          # Set up everything
    knowwhere health        # Check pipeline status
    knowwhere doctor        # Diagnose common issues
    knowwhere config show   # Print current config
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional


def _get_config_dir() -> Path:
    return Path.home() / ".knowwhere"


def _get_config_path() -> Path:
    return _get_config_dir() / "config.toml"


def _get_hermes_plugins_dir() -> Path:
    return Path.home() / ".hermes" / "plugins"


def _get_repo_root() -> Path:
    """Repo root is the parent of this knowwhere/ package."""
    return Path(__file__).resolve().parent.parent


def _get_package_dir() -> Path:
    """Package directory containing schema, plugin, and other data files."""
    return Path(__file__).resolve().parent


def _get_schema_path() -> Path:
    """Find schema.sql — package-bundled copy preferred, repo-root fallback."""
    pkg_path = _get_package_dir() / "schema.sql"
    if pkg_path.exists():
        return pkg_path
    repo_path = _get_repo_root() / "knowwhere-schema.sql"
    if repo_path.exists():
        return repo_path
    return pkg_path  # best effort


def _get_plugin_src() -> Path | None:
    """Find hermes_plugin source — package-bundled copy preferred, repo-root fallback."""
    pkg_path = _get_package_dir() / "hermes_plugin"
    if pkg_path.exists():
        return pkg_path
    repo_path = _get_repo_root() / "hermes-plugin" / "knowwhere"
    if repo_path.exists():
        return repo_path
    return None


def _load_config() -> dict | None:
    """Load config.toml if it exists, return parsed dict or None."""
    config_path = _get_config_path()
    if not config_path.exists():
        return None
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            # Fallback: basic line-based parsing
            return _parse_config_fallback(config_path)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _parse_config_fallback(path: Path) -> dict:
    """Very basic TOML fallback for when tomllib/tomli is unavailable."""
    result: dict = {}
    current_section: dict | None = None
    section_key = ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section_key = line[1:-1]
            current_section = {}
            result[section_key] = current_section
        elif "=" in line and current_section is not None:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"')
            current_section[key] = val
    return result


def _read_zshrc() -> str:
    zshrc = Path.home() / ".zshrc"
    if zshrc.exists():
        return zshrc.read_text()
    return ""


def _write_zshrc(env_lines: list[str]) -> None:
    """Append env vars to .zshrc if not present."""
    zshrc = Path.home() / ".zshrc"
    existing = _read_zshrc()
    new_lines = []
    for line in env_lines:
        key = line.split("=")[0].replace("export ", "")
        if key not in existing:
            new_lines.append(line)
    if new_lines:
        with open(zshrc, "a") as f:
            f.write("\n# KnowWhere\n")
            for line in new_lines:
                f.write(line + "\n")
        print(f"✓ Added {len(new_lines)} env var(s) to ~/.zshrc")


def cmd_init(args) -> int:
    """Initialize KnowWhere: config, DB, plugin, cron."""
    config_dir = _get_config_dir()
    config_path = _get_config_path()
    repo_root = _get_repo_root()

    print("⚡ KnowWhere init\n")

    # 1. Create config directory
    config_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Config directory: {config_dir}")

    # 2. Database setup
    db_url = os.environ.get("KNOWWHERE_DB_URL", "")
    if not db_url:
        # Try .zshrc
        import re
        zshrc_text = _read_zshrc()
        m = re.search(r'export KNOWWHERE_DB_URL="([^"]+)"', zshrc_text)
        if m:
            db_url = m.group(1)

    if not db_url:
        print("\n⚠️  No KNOWWHERE_DB_URL found.")
        print("   KnowWhere needs a PostgreSQL database with pgvector.")
        print()
        print("   Option A: Use an existing PostgreSQL (recommended)")
        print("     export KNOWWHERE_DB_URL='postgresql://user:pass@host:5432/dbname'")
        print()
        print("   Option B: Local PostgreSQL via Homebrew")
        print("     brew install postgresql@16 && brew services start postgresql@16")
        print("     createdb knowwhere")
        print("     psql knowwhere -c 'CREATE EXTENSION vector;'")
        print("     export KNOWWHERE_DB_URL='postgresql://localhost:5432/knowwhere'")
        print()
        print("   Option C: Railway (free tier)")
        print("     railway.app → New Project → Provision PostgreSQL")
        print("     Copy the connection string")
        print()
        db_url = input("   Paste your KNOWWHERE_DB_URL: ").strip()

    if db_url:
        _write_zshrc([f'export KNOWWHERE_DB_URL="{db_url}"'])
        os.environ["KNOWWHERE_DB_URL"] = db_url

    # 3. Write config.toml
    embedding_provider = args.embedding or "ollama"
    llm_provider = args.llm or "deepseek"
    config_content = textwrap.dedent(f"""\
    # KnowWhere configuration — generated by `knowwhere init`
    [storage]
    url = "{db_url}"

    [embedding]
    provider = "{embedding_provider}"
    ollama_model = "nomic-embed-text"

    [summarization]
    provider = "{llm_provider}"

    [injection]
    max_chars = 3000
    min_score = 0.30
    ucb_c = 1.5
    """)
    config_path.write_text(config_content)
    print(f"✓ Config: {config_path}")

    # 4. Initialize database schema
    print("\n📦 Setting up database...")
    schema_path = _get_schema_path()
    if schema_path.exists():
        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(schema_path.read_text())
                conn.commit()
            conn.close()
            print("✓ Database schema applied")
        except Exception as e:
            print(f"⚠️  Database setup failed: {e}")
            print("   You can retry later with: knowwhere db init")
    else:
        print(f"⚠️  Schema file not found: {schema_path}")
        print("   Run: knowwhere db init --schema-path <path>")

    # 5. Install Hermes plugin
    print("\n🔌 Installing Hermes plugin...")
    plugin_src = _get_plugin_src()
    plugin_dst = _get_hermes_plugins_dir() / "knowwhere"

    if plugin_src and plugin_src.exists():
        # Backup existing
        if plugin_dst.exists() or plugin_dst.is_symlink():
            backup_dir = Path.home() / ".hermes" / "plugin-backups" / "knowwhere"
            backup_dir.mkdir(parents=True, exist_ok=True)
            if plugin_dst.is_dir() and not plugin_dst.is_symlink():
                shutil.copytree(plugin_dst, backup_dir, dirs_exist_ok=True)
                shutil.rmtree(plugin_dst)
            elif plugin_dst.is_symlink():
                plugin_dst.unlink()
            print(f"  Backed up existing plugin to {backup_dir}")

        # Create symlink
        plugin_dst.symlink_to(plugin_src.resolve())
        print(f"✓ Plugin installed: {plugin_dst} → {plugin_src}")

        # Enable plugin
        try:
            result = subprocess.run(
                ["hermes", "plugins", "enable", "knowwhere"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print("✓ Plugin enabled in Hermes")
            else:
                print(f"  Note: Run 'hermes plugins enable knowwhere' manually")
                print(f"  (hermes CLI not available: {result.stderr.strip()})")
        except FileNotFoundError:
            print("  Note: Run 'hermes plugins enable knowwhere' after restarting Hermes")
    else:
        print(f"⚠️  Plugin source not found: {plugin_src}")

    # 6. Set up cron jobs
    print("\n⏰ Setting up nightly pipeline cron...")
    try:
        # Check if hermes CLI is available
        result = subprocess.run(
            ["hermes", "cron", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("  (Run 'knowwhere cron install' to install the nightly pipeline)")
            print("   Or manually: hermes cron create ...")
        else:
            print("  Install manually via Hermes CLI when available")
    except FileNotFoundError:
        print("  Hermes CLI not found — install cron jobs after Hermes is set up:")
        print("  $ knowwhere cron install")

    # 7. Summary
    print("\n" + "=" * 50)
    print("✅ KnowWhere initialized!")
    print()
    print("Next steps:")
    print(f"  1. Source your env:  source ~/.zshrc")
    print(f"  2. Restart Hermes:    hermes gateway restart")
    print(f"  3. Verify:            knowwhere health")
    print()
    print("Optional:")
    print(f"  • Install local embeddings (no Ollama): pip install knowwhere[embeddings]")
    print(f"  • Set up nightly cron:                  knowwhere cron install")
    print("=" * 50)

    return 0


def cmd_health(args) -> int:
    """Check KnowWhere pipeline health."""
    config = _load_config()
    if not config:
        print("❌ KnowWhere not initialized. Run: knowwhere init")
        return 1

    db_url = config.get("storage", {}).get("url", "")
    if not db_url:
        db_url = os.environ.get("KNOWWHERE_DB_URL", "")

    print("🩺 KnowWhere Health\n")

    # Plugin check
    plugin_path = _get_hermes_plugins_dir() / "knowwhere"
    if plugin_path.exists():
        print("✓ Hermes plugin: installed")
    else:
        print("⚠️  Hermes plugin: not installed")

    # DB check
    if db_url:
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            conn = psycopg2.connect(db_url, connect_timeout=5)
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM summaries")
                summary_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM summaries WHERE embedding IS NOT NULL")
                embed_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM summaries WHERE debut_seen = FALSE")
                debut_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM sources")
                source_count = cur.fetchone()[0]
            conn.close()

            print(f"✓ Database: connected")
            print(f"  Summaries: {summary_count}")
            print(f"  Embeddings: {embed_count}/{summary_count}")
            print(f"  Debuts pending: {debut_count}")
            print(f"  Sources: {source_count}")

            if embed_count < summary_count:
                null_count = summary_count - embed_count
                print(f"  ⚠️  {null_count} summaries without embeddings — run: knowwhere embed")
        except Exception as e:
            print(f"⚠️  Database: unreachable ({e})")
    else:
        print("⚠️  Database: not configured")

    # Ollama check
    try:
        import urllib.request, json
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        if "nomic-embed-text" in models:
            print("✓ Ollama: running (nomic-embed-text available)")
        else:
            print(f"⚠️  Ollama: running but nomic-embed-text not found")
            print(f"   Install: ollama pull nomic-embed-text")
    except Exception:
        print("⚠️  Ollama: not running (embeddings won't work)")
        print("   Install: brew install ollama && ollama serve")

    # Cron check
    print("\n  Cron jobs: check with 'hermes cron list'")

    return 0


def cmd_doctor(args) -> int:
    """Diagnose common KnowWhere issues."""
    print("🔍 KnowWhere Doctor\n")
    issues = 0

    # Check config
    config = _load_config()
    if not config:
        print("❌ No config found — run: knowwhere init")
        return 1

    db_url = config.get("storage", {}).get("url", "")
    if not db_url:
        db_url = os.environ.get("KNOWWHERE_DB_URL", "")

    # 1. DB connectivity
    print("1. Database...")
    if db_url:
        try:
            import psycopg2
            conn = psycopg2.connect(db_url, connect_timeout=5)
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            print("   ✅ Connected")
        except Exception as e:
            print(f"   ❌ Connection failed: {e}")
            print("   → Check KNOWWHERE_DB_URL in ~/.zshrc")
            print("   → Verify database is running and accessible")
            issues += 1
    else:
        print("   ❌ No database URL configured")
        issues += 1

    # 2. pgvector extension
    if db_url:
        try:
            import psycopg2
            conn = psycopg2.connect(db_url, connect_timeout=5)
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
                if cur.fetchone():
                    print("   ✅ pgvector extension installed")
                else:
                    print("   ❌ pgvector extension missing")
                    print("   → Run: psql $KNOWWHERE_DB_URL -c 'CREATE EXTENSION vector;'")
                    issues += 1
            conn.close()
        except Exception:
            pass

    # 3. Embedding NULL check
    if db_url:
        try:
            import psycopg2
            conn = psycopg2.connect(db_url, connect_timeout=5)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM summaries WHERE embedding IS NULL")
                null_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM summaries")
                total = cur.fetchone()[0]
            conn.close()
            if null_count == 0:
                print(f"   ✅ All {total} summaries have embeddings")
            else:
                print(f"   ⚠️  {null_count}/{total} summaries have NULL embeddings")
                print(f"   → Run: knowwhere embed")
        except Exception:
            pass

    # 4. Ollama check
    print("\n2. Embedding provider (Ollama)...")
    try:
        import urllib.request, json
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        if "nomic-embed-text" in models:
            print("   ✅ Ollama + nomic-embed-text ready")
        else:
            print("   ⚠️  Ollama running but nomic-embed-text missing")
            print("   → Run: ollama pull nomic-embed-text")
            issues += 1
    except Exception:
        print("   ❌ Ollama not running on localhost:11434")
        print("   → Start: ollama serve")
        print("   → Or install local embeddings: pip install knowwhere[embeddings]")
        issues += 1

    # 5. Plugin check
    print("\n3. Hermes plugin...")
    plugin_path = _get_hermes_plugins_dir() / "knowwhere"
    if plugin_path.exists():
        print("   ✅ Plugin installed")
        # Check if it's a symlink
        if plugin_path.is_symlink():
            target = plugin_path.resolve()
            print(f"   → Linked to: {target}")
            if not target.exists():
                print(f"   ❌ Symlink target missing!")
                print(f"   → Run: knowwhere init")
                issues += 1
    else:
        print("   ❌ Plugin not installed")
        print("   → Run: knowwhere init")
        issues += 1

    # 6. Pipeline status (hook-driven, no cron required)
    print("\n4. Pipeline...")
    plugin_path = _get_hermes_plugins_dir() / "knowwhere"
    if plugin_path.exists() and (plugin_path / "plugin.yaml").exists():
        print("   ✅ Hook-driven pipeline active (post_llm_call + on_session_reset)")
        print("   → Summaries are generated live — no nightly cron needed")
    else:
        print("   ⚠️  Plugin installed but hooks may not be active")

    # Optional: Debut Injection cron (auto-injects unseen summaries nightly)
    print("\n5. Debut injection cron (optional)...")
    try:
        result = subprocess.run(
            ["hermes", "cron", "list"],
            capture_output=True, text=True, timeout=10
        )
        if "knowwhere" in result.stdout.lower() or "debut" in result.stdout.lower():
            print("   ✅ Debut injection cron found")
        else:
            print("   ℹ️  No Debut injection cron (optional — debuts inject at session start)")
            print("   → For nightly auto-injection: knowwhere cron install")
    except FileNotFoundError:
        print("   ℹ️  Cannot check — hermes CLI not available")

    # Summary
    print(f"\n{'=' * 40}")
    if issues == 0:
        print("✅ All checks passed!")
    else:
        print(f"⚠️  Found {issues} issue(s) — see above for fixes")
    return min(issues, 1)


def cmd_cron_install(args) -> int:
    """Print instructions for installing cron jobs."""
    print("⏰ KnowWhere Cron Installation\n")
    print("The nightly pipeline runs three jobs via Hermes cron:\n")
    print("1. Nightly Summarize + Embed (23:00):")
    print("   hermes cron create \\")
    print("     --name 'KnowWhere Nightly Pipeline' \\")
    print("     --schedule '0 23 * * *' \\")
    print("     --prompt 'Run the KnowWhere nightly pipeline:'")
    print("     --model deepseek-v4-flash")
    print()
    print("2. Debut Injection (23:15):")
    print("   hermes cron create \\")
    print("     --name 'KnowWhere Debut Injection' \\")
    print("     --schedule '15 23 * * *' \\")
    print("     --prompt 'Run debut injection for KnowWhere'")
    print()
    print("3. DB Health Check (every 30 min):")
    print("   hermes cron create \\")
    print("     --name 'KnowWhere DB Health' \\")
    print("     --schedule '*/30 * * * *' \\")
    print("     --prompt 'Check KnowWhere database health'")
    print()
    print("Or use the Hermes TUI: hermes cron")
    return 0


def cmd_embed(args) -> int:
    """Run embedding backfill."""
    print("📐 Embedding backfill...")
    repo_root = _get_repo_root()
    script = repo_root / "embed_summaries.py"
    if script.exists():
        result = subprocess.run([sys.executable, str(script)], cwd=str(repo_root))
        return result.returncode
    else:
        print(f"❌ Script not found: {script}")
        return 1


def cmd_config(args) -> int:
    """Show or edit config."""
    action = args.config_action
    config_path = _get_config_path()

    if action == "show":
        if config_path.exists():
            print(config_path.read_text())
        else:
            print("No config found. Run: knowwhere init")
            return 1
    elif action == "path":
        print(config_path)
    return 0


def cmd_db(args) -> int:
    """Database operations."""
    if args.db_action == "init":
        config = _load_config()
        db_url = (config or {}).get("storage", {}).get("url", "") if config else ""
        if not db_url:
            db_url = os.environ.get("KNOWWHERE_DB_URL", "")
        if not db_url:
            print("❌ No database URL configured")
            return 1

        schema_path = args.schema_path or str(_get_schema_path())
        if not Path(schema_path).exists():
            print(f"❌ Schema not found: {schema_path}")
            return 1

        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(Path(schema_path).read_text())
                conn.commit()
            conn.close()
            print("✓ Schema applied")
            return 0
        except Exception as e:
            print(f"❌ Failed: {e}")
            return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="knowwhere",
        description="KnowWhere — Subconscious Memory for AI Agents",
    )
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Initialize KnowWhere (config, DB, plugin, cron)")
    p_init.add_argument("--embedding", choices=["ollama", "openai", "local"],
                        default="ollama", help="Embedding provider (default: ollama)")
    p_init.add_argument("--llm", choices=["deepseek", "openai", "ollama"],
                        default="deepseek", help="LLM provider for summarization")

    # health
    sub.add_parser("health", help="Check KnowWhere pipeline health")

    # doctor
    sub.add_parser("doctor", help="Diagnose common issues")

    # cron
    p_cron = sub.add_parser("cron", help="Cron job management")
    p_cron.add_argument("cron_action", choices=["install", "list"],
                        default="install", nargs="?",
                        help="Action (default: install)")

    # embed
    sub.add_parser("embed", help="Run embedding backfill for NULL embeddings")

    # config
    p_config = sub.add_parser("config", help="Config management")
    p_config.add_argument("config_action", choices=["show", "path"],
                          default="show", nargs="?",
                          help="Action (default: show)")

    # db
    p_db = sub.add_parser("db", help="Database operations")
    p_db.add_argument("db_action", choices=["init"],
                      default="init", nargs="?",
                      help="Action (default: init)")
    p_db.add_argument("--schema-path", help="Path to knowwhere-schema.sql")

    args = parser.parse_args()

    if args.command == "init":
        return cmd_init(args)
    elif args.command == "health":
        return cmd_health(args)
    elif args.command == "doctor":
        return cmd_doctor(args)
    elif args.command == "cron":
        return cmd_cron_install(args)
    elif args.command == "embed":
        return cmd_embed(args)
    elif args.command == "config":
        return cmd_config(args)
    elif args.command == "db":
        return cmd_db(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
