# Makefile for ObbyRL development
# Provides convenient commands for common operations

.PHONY: help install server dashboard test-server test-offline config clean

# Default target
help:
	@echo "ObbyRL Development Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make clean         - Clean generated files"
	@echo ""
	@echo "Running Services:"
	@echo "  make server        - Start RL training server"
	@echo "  make dashboard     - Start monitoring dashboard"
	@echo ""
	@echo "Testing & Development:"
	@echo "  make test-server   - Test server functionality"
	@echo "  make test-offline  - Test offline training"
	@echo "  make config        - View current configuration"
	@echo ""
	@echo "Configuration Management:"
	@echo "  make config-update - Update reward configuration"
	@echo "  make config-reload - Reload server configuration"
	@echo ""
	@echo "Training & Experiments:"
	@echo "  make train-offline - Run offline training (20 episodes)"
	@echo "  make performance   - Run performance test"

# Installation and setup
install:
	@echo "Installing Python dependencies..."
	cd server && pip install -r requirements.txt
	@echo "✅ Installation complete"

# Start RL training server
server:
	@echo "🚀 Starting RL training server..."
	cd server && python rl_server.py

# Start monitoring dashboard
dashboard:
	@echo "📊 Starting monitoring dashboard..."
	@echo "Dashboard will be available at: http://127.0.0.1:5001"
	cd server && python dashboard.py

# Test server functionality
test-server:
	@echo "🧪 Testing server functionality..."
	cd server && python test_performance.py

# Test offline training
test-offline:
	@echo "🔬 Testing offline training environment..."
	cd server && python train_offline.py --test

# View current configuration
config:
	@echo "⚙️ Current Reward Configuration:"
	cd server && python config_tool.py --action view --config-type reward
	@echo ""
	@echo "⚙️ Current Model Configuration:"
	cd server && python config_tool.py --action view --config-type model

# Update reward configuration with optimized values
config-update:
	@echo "🔧 Updating reward configuration..."
	cd server && python config_tool.py --action update-rewards

# Reload server configuration
config-reload:
	@echo "🔄 Reloading server configuration..."
	cd server && python config_tool.py --action reload

# Run offline training
train-offline:
	@echo "🏃 Running offline training (20 episodes)..."
	cd server && python train_offline.py --episodes 20

# Run extended offline training
train-offline-long:
	@echo "🏃 Running extended offline training (100 episodes)..."
	cd server && python train_offline.py --episodes 100

# Run performance test
performance:
	@echo "⚡ Running performance test..."
	cd server && python test_performance.py

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f server/training_metrics.csv
	rm -f server/offline_training_results.png
	rm -f checkpoints/offline_trained.pt
	@echo "✅ Cleanup complete"

# Development shortcuts
dev-setup: install
	@echo "🛠️ Setting up development environment..."
	@echo "Run 'make server' to start the RL server"
	@echo "Run 'make dashboard' to start the monitoring dashboard"
	@echo "Open Roblox Studio and load the place file to begin training"

# Quick test suite
test-all: test-server test-offline
	@echo "✅ All tests completed"

# Monitor logs in real-time (requires server to be running)
logs:
	@echo "📋 Monitoring training logs..."
	@echo "Make sure server is running first with 'make server'"
	tail -f server/training_metrics.csv 2>/dev/null || echo "CSV log file not found. Start training first."

# Performance benchmark
benchmark:
	@echo "🏁 Running performance benchmark..."
	cd server && python train_offline.py --episodes 50
	@echo "Benchmark complete. Check checkpoints/offline_trained.pt for results."

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "Current README.md contains comprehensive documentation"
	@echo "API endpoints:"
	@echo "  GET  /stats          - Performance statistics"
	@echo "  GET  /config/reward  - Reward configuration"  
	@echo "  GET  /config/model   - Model configuration"
	@echo "  POST /config/reload  - Reload configuration"
	@echo "  POST /step           - Training step (used by Roblox client)"

# Show system status
status:
	@echo "🔍 System Status:"
	@echo ""
	@echo "RL Server:"
	@curl -s http://127.0.0.1:5000/stats >/dev/null 2>&1 && echo "  ✅ Running (port 5000)" || echo "  ❌ Not running"
	@echo ""
	@echo "Dashboard:"
	@curl -s http://127.0.0.1:5001/ >/dev/null 2>&1 && echo "  ✅ Running (port 5001)" || echo "  ❌ Not running" 
	@echo ""
	@echo "Configuration files:"
	@test -f config/model_config.yaml && echo "  ✅ model_config.yaml" || echo "  ❌ model_config.yaml missing"
	@test -f config/reward_shaping.conf && echo "  ✅ reward_shaping.conf" || echo "  ❌ reward_shaping.conf missing"
	@echo ""
	@echo "Python dependencies:"
	@cd server && python -c "import torch, flask, yaml; print('  ✅ All dependencies installed')" 2>/dev/null || echo "  ❌ Missing dependencies - run 'make install'"