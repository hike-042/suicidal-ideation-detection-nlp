#!/usr/bin/env python3
"""
Suicidal Ideation Detection in Social Media
Launch script for the web application.
"""
import os
import sys
import subprocess
import webbrowser
import time
import argparse


def load_local_env(env_path: str = ".env"):
    """Load KEY=VALUE pairs from a local .env file if present."""
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"WARNING: Could not load {env_path}: {exc}")


def check_dependencies():
    """Check if required packages are installed."""
    required = ["fastapi", "uvicorn", "jinja2"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def check_api_keys():
    """Check if at least one LLM API path is configured."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if openrouter_key:
        print("OpenRouter API key loaded.")
    if anthropic_key:
        print("Anthropic API key loaded.")

    if not anthropic_key and not openrouter_key:
        print("WARNING: No API key configured.")
        print("   The app will only work via fallback mode unless you set OPENROUTER_API_KEY or ANTHROPIC_API_KEY.")
        print()

    return bool(anthropic_key or openrouter_key)


def find_free_port(preferred: int) -> int:
    """Return preferred port if free, otherwise scan 8001-8099."""
    import socket
    for port in [preferred] + list(range(8001, 8100)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    return preferred  # give up, let uvicorn raise the error


def main():
    load_local_env()
    parser = argparse.ArgumentParser(description="Launch MindGuard — AI Mental Wellness Analyser")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Preferred port")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload (development)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MindGuard — AI Mental Wellness Risk Analyser")
    print("  Multi-Agent Suicidal Ideation Detection")
    print("=" * 60)
    print()

    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Run: pip install {' '.join(missing)}")
        sys.exit(1)

    # Check API keys
    check_api_keys()

    # Find a free port
    port = find_free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} is in use — using port {port} instead.")

    url = f"http://localhost:{port}"
    print(f"Starting server at {url}")
    print("Press Ctrl+C to stop")
    print()

    # Open browser after a short delay
    if not args.no_browser:
        def open_browser():
            time.sleep(2)
            webbrowser.open(url)
        import threading
        t = threading.Thread(target=open_browser, daemon=True)
        t.start()

    # Run uvicorn
    reload_flag = ["--reload"] if args.reload else []
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", args.host,
        "--port", str(port),
    ] + reload_flag)


if __name__ == "__main__":
    main()
