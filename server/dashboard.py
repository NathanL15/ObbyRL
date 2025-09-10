#!/usr/bin/env python3
"""
Simple web dashboard for monitoring RL training progress.
Provides real-time metrics visualization without heavy dependencies.
"""

from flask import Flask, render_template_string, jsonify
import requests
import json
import time
from threading import Thread
import os

# Simple HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>RL Training Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }
        .metric-value { font-size: 24px; color: #27ae60; margin: 5px 0; }
        .metric-subtitle { font-size: 14px; color: #7f8c8d; }
        .status-good { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-error { color: #e74c3c; }
        .config-section { background: #ecf0f1; padding: 15px; border-radius: 4px; margin: 10px 0; }
        .error { color: #e74c3c; padding: 20px; background: #fadbd8; border-radius: 8px; }
        .timestamp { font-size: 12px; color: #95a5a6; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 RL Training Dashboard</h1>
            <p>Real-time monitoring for ObbyRL training system</p>
            <div class="timestamp">Last updated: {{ timestamp }}</div>
        </div>
        
        {% if error %}
        <div class="error">
            <strong>Connection Error:</strong> {{ error }}
            <br><small>Make sure the RL server is running on {{ server_url }}</small>
        </div>
        {% else %}
        
        <div class="metrics-grid">
            <!-- Performance Metrics -->
            <div class="metric-card">
                <div class="metric-title">⚡ Performance</div>
                <div class="metric-value {{ 'status-good' if stats.avg_request_time_ms < 120 else 'status-warning' if stats.avg_request_time_ms < 200 else 'status-error' }}">
                    {{ "%.1f" | format(stats.avg_request_time_ms) }}ms
                </div>
                <div class="metric-subtitle">Average request time</div>
                <div class="metric-subtitle">Target: <120ms</div>
                <div class="metric-subtitle">Requests processed: {{ stats.request_count }}</div>
            </div>
            
            <!-- Training Progress -->
            <div class="metric-card">
                <div class="metric-title">📈 Training Progress</div>
                <div class="metric-value">{{ stats.step_count }}</div>
                <div class="metric-subtitle">Total steps</div>
                <div class="metric-subtitle">Episode: {{ stats.current_episode }}</div>
                <div class="metric-subtitle">Best return: {{ "%.2f" | format(stats.best_return) }}</div>
            </div>
            
            <!-- Exploration -->
            <div class="metric-card">
                <div class="metric-title">🎯 Exploration</div>
                <div class="metric-value {{ 'status-good' if stats.eps < 0.3 else 'status-warning' if stats.eps < 0.7 else 'status-error' }}">
                    {{ "%.3f" | format(stats.eps) }}
                </div>
                <div class="metric-subtitle">Epsilon (exploration rate)</div>
                <div class="metric-subtitle">Lower = more exploitation</div>
            </div>
            
            <!-- Episode Metrics -->
            <div class="metric-card">
                <div class="metric-title">🏆 Episodes</div>
                <div class="metric-value">{{ "%.1f" | format(stats.avg_episode_reward) }}</div>
                <div class="metric-subtitle">Average reward</div>
                <div class="metric-subtitle">Length: {{ "%.1f" | format(stats.avg_episode_length) }} steps</div>
                <div class="metric-subtitle">Recent episodes: {{ stats.recent_episodes }}</div>
            </div>
            
            <!-- Buffer Status -->
            <div class="metric-card">
                <div class="metric-title">💾 Memory</div>
                <div class="metric-value">{{ stats.buffer_size }}</div>
                <div class="metric-subtitle">Replay buffer size</div>
                <div class="metric-subtitle">Elite buffer: {{ stats.elite_buffer_size }}</div>
                <div class="metric-subtitle">Training loss: {{ "%.4f" | format(stats.last_loss) }}</div>
            </div>
            
            <!-- Optimizations -->
            <div class="metric-card">
                <div class="metric-title">⚙️ Optimizations</div>
                <div class="metric-subtitle">
                    Action Cache: 
                    <span class="{{ 'status-good' if stats.action_cache.enabled else 'status-warning' }}">
                        {{ 'Enabled' if stats.action_cache.enabled else 'Disabled' }}
                    </span>
                </div>
                <div class="metric-subtitle">Cache size: {{ stats.action_cache.size }}/{{ stats.action_cache.max_size }}</div>
                <div class="metric-subtitle">
                    Request batching: 
                    <span class="{{ 'status-good' if stats.request_batcher.enabled else 'status-warning' }}">
                        {{ 'Enabled' if stats.request_batcher.enabled else 'Disabled' }}
                    </span>
                </div>
            </div>
        </div>
        
        <!-- Action Distribution -->
        {% if stats.action_distribution %}
        <div class="metric-card" style="margin-top: 20px;">
            <div class="metric-title">🎮 Action Distribution</div>
            {% for action, count in stats.action_distribution.items() %}
            <div class="metric-subtitle">
                Action {{ action }}: {{ count }} 
                ({{ "%.1f" | format(count * 100 / stats.request_count) if stats.request_count > 0 else 0 }}%)
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <!-- Configuration Summary -->
        <div class="metric-card" style="margin-top: 20px;">
            <div class="metric-title">🔧 Configuration</div>
            {% if model_config %}
            <div class="config-section">
                <strong>Model:</strong> {{ model_config.model.hidden_size }} hidden units, 
                {{ model_config.model.n_actions }} actions
            </div>
            <div class="config-section">
                <strong>Training:</strong> LR={{ model_config.training.learning_rate }}, 
                Batch={{ model_config.training.batch_size }}, 
                Gamma={{ model_config.training.gamma }}
            </div>
            <div class="config-section">
                <strong>Optimizations:</strong> 
                Compile={{ model_config.optimization.compile_model }}, 
                Cache={{ model_config.optimization.get('enable_action_cache', False) }}
            </div>
            {% endif %}
        </div>
        
        {% endif %}
    </div>
</body>
</html>
"""

class Dashboard:
    def __init__(self, rl_server_url="http://127.0.0.1:5000", port=5001):
        self.rl_server_url = rl_server_url
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/')
        def dashboard():
            try:
                # Get stats from RL server
                stats_response = requests.get(f"{self.rl_server_url}/stats", timeout=5)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                else:
                    raise Exception(f"Stats API returned {stats_response.status_code}")
                
                # Get model config
                config_response = requests.get(f"{self.rl_server_url}/config/model", timeout=5)
                model_config = config_response.json() if config_response.status_code == 200 else None
                
                return render_template_string(
                    DASHBOARD_HTML,
                    stats=stats,
                    model_config=model_config,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    server_url=self.rl_server_url,
                    error=None
                )
                
            except Exception as e:
                return render_template_string(
                    DASHBOARD_HTML,
                    stats={},
                    model_config=None,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    server_url=self.rl_server_url,
                    error=str(e)
                )
        
        @self.app.route('/api/stats')
        def api_stats():
            """JSON API for stats (for external tools)"""
            try:
                response = requests.get(f"{self.rl_server_url}/stats", timeout=5)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"RL server returned {response.status_code}"}, 500
            except Exception as e:
                return {"error": str(e)}, 500
    
    def run(self, debug=False):
        """Start the dashboard server"""
        print(f"🚀 Starting RL Training Dashboard on http://127.0.0.1:{self.port}")
        print(f"📊 Monitoring RL server at {self.rl_server_url}")
        print(f"🔄 Auto-refresh every 5 seconds")
        self.app.run(host='127.0.0.1', port=self.port, debug=debug)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RL Training Dashboard")
    parser.add_argument("--rl-url", default="http://127.0.0.1:5000", 
                       help="RL server URL")
    parser.add_argument("--port", type=int, default=5001,
                       help="Dashboard port")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode")
    
    args = parser.parse_args()
    
    dashboard = Dashboard(args.rl_url, args.port)
    dashboard.run(args.debug)

if __name__ == "__main__":
    main()