"""InterviewIQ Server Launcher with Automatic Local Network IP Detection.

Binds to 0.0.0.0 so that other laptops, desktops, and mobile phones on the same
Wi-Fi / Ethernet local network can connect directly to InterviewIQ.
"""
import socket
import sys
import uvicorn


def get_local_ip() -> str:
    """Detect the active local IPv4 address on Wi-Fi or Ethernet."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to a public DNS IP (doesn't send any traffic) to determine the routing IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def main():
    port = 8001
    local_ip = get_local_ip()

    print("\n" + "=" * 74)
    print("  🚀  InterviewIQ — GenAI Virtual Interviewer Platform is LIVE!")
    print("=" * 74)
    print(f"  💻  Local Access (This Computer):")
    print(f"      http://localhost:{port}")
    print(f"      http://127.0.0.1:{port}")
    print()
    if local_ip not in ("127.0.0.1", "localhost"):
        print(f"  🌐  Network / Wi-Fi Broadcast URL (Other devices on same network):")
        print(f"      http://{local_ip}:{port}")
        print(f"      👉 Share this clickable URL with students or interviewers on your Wi-Fi")
    print()
    print(f"  📖  API Documentation (Swagger UI):")
    print(f"      http://localhost:{port}/docs")
    print("=" * 74)
    print("  Press Ctrl+C to stop the server.\n")

    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
