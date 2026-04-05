"""Entry point for the trading bot dashboard server."""
import sys
import os

# Ensure project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.dashboard_server import app, bot_loop
import threading

if __name__ == "__main__":
    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()
    print("Dashboard: http://localhost:5555")
    app.run(host="0.0.0.0", port=5555, debug=False, threaded=True)
