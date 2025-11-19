#!/usr/bin/env python3
"""
Load Balancer for Distributed Monolith Architecture
Routes requests to backend monolith instances using round-robin
"""

import os
import requests
import threading
from flask import Flask, request, jsonify

# Read environment variables
NODE_0_LB_IP = os.environ["NODE_0_LB_IP"]
NODE_0_IP = os.environ["NODE_0_IP"]
NODE_1_IP = os.environ["NODE_1_IP"]
NODE_2_IP = os.environ["NODE_2_IP"]

# Backend port for Node 0 (runs on same machine as load balancer)

app = Flask(__name__)


class RoundRobinLoadBalancer:
    """Simple round-robin load balancer"""

    def __init__(self, backend_urls):
        self.backends = backend_urls
        self.current_index = 0
        # NOTE: Flask app is threaded=True, need lock per request.
        self.lock = threading.Lock()
        print(f"Load balancer initialized with backends: {self.backends}")

    def get_next_backend(self):
        """Get next backend URL using round-robin"""
        with self.lock:
            backend = self.backends[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.backends)
            return backend


# Initialize load balancer with backend URLs
backend_urls = ["http://" + node_ip for node_ip in [NODE_0_IP, NODE_1_IP, NODE_2_IP]]
load_balancer = RoundRobinLoadBalancer(backend_urls)


@app.route("/query", methods=["POST"])
def handle_query():
    """Forward query to a backend using round-robin"""
    try:
        # Get request data
        data = request.json
        request_id = data.get("request_id")
        query = data.get("query")

        if not request_id or not query:
            return jsonify({"error": "Missing request_id or query"}), 400

        # Get next backend
        backend_url = load_balancer.get_next_backend()
        print(f"[Load Balancer] Routing request {request_id} to {backend_url}")

        # Forward request to backend
        response = requests.post(
            f"{backend_url}/query",
            json=data,
            timeout=300,  # NOTE: 5 minute timeout -- match hardcoded handle_query in pipeline.py
        )

        # Return backend response
        return jsonify(response.json()), response.status_code

    except requests.exceptions.Timeout:
        return jsonify({"error": "Backend timeout"}), 504
    except requests.exceptions.RequestException as e:
        print(f"[Load Balancer] Error forwarding request: {e}")
        return jsonify({"error": f"Backend error: {str(e)}"}), 502
    except Exception as e:
        print(f"[Load Balancer] Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "role": "load_balancer",
            "backends": load_balancer.backends,
        }
    ), 200


def main():
    """Start the load balancer"""
    print("=" * 60)
    print("LOAD BALANCER - Round Robin")
    print("=" * 60)
    print(f"\nListening on: {NODE_0_LB_IP}")
    print(f"Backends:")
    for i, backend in enumerate(backend_urls):
        print(f"  {i}: {backend}")
    print("\n" + "=" * 60)

    # Parse hostname and port from NODE_0_IP
    hostname, port = NODE_0_LB_IP.split(":")
    port = int(port)

    # Start Flask server
    app.run(host=hostname, port=port, threaded=True)


if __name__ == "__main__":
    main()
