#!/usr/bin/env python3
"""
Test script to demonstrate the new performance logging and metrics capabilities.
This simulates some training steps to show the logging output.
"""

import time
import json
import requests
import numpy as np

def test_server_performance():
    base_url = "http://127.0.0.1:5000"
    
    print("Testing RL server performance tracking...")
    
    # Test stats endpoint
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        print(f"Stats endpoint: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Server not running or not accessible: {e}")
        return
    
    # Simulate some training steps
    print("\nSimulating training steps...")
    
    # Sample observation matching the expected format
    sample_obs = {
        "dx": 10.5, "dy": 2.1, "dz": -5.3,
        "vx": 1.2, "vy": 0.1, "vz": -0.8,
        "down": 3.2, "forward": 15.6,
        "angle": 0.8, "grounded": 1, "speed": 12.5, "tJump": 0.1,
        "r0": 20, "r1": 18, "r2": 25, "r3": 30,
        "r4": 22, "r5": 19, "r6": 24, "r7": 28,
        "dropF": 5.0, "dropR": 4.8, "dropL": 5.2,
        "hazardDist": 0.7, "lastDeathType": 0
    }
    
    for step in range(20):
        # Add some variation
        obs = sample_obs.copy()
        obs["dx"] += np.random.normal(0, 2)
        obs["dy"] += np.random.normal(0, 1)
        
        payload = {
            "obs": obs,
            "reward": np.random.normal(0.1, 0.5),  # Small random rewards
            "done": (step % 15 == 14)  # End episode every 15 steps
        }
        
        start_time = time.time()
        try:
            response = requests.post(f"{base_url}/step", json=payload, timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                action = response.json().get("action", 0)
                print(f"Step {step}: action={action}, duration={duration*1000:.1f}ms")
            else:
                print(f"Step {step}: Error {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Step {step}: Request failed: {e}")
            break
            
        time.sleep(0.1)  # Small delay between requests
    
    # Get final stats
    print("\nFinal performance stats:")
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"Average request time: {stats['avg_request_time_ms']:.1f}ms")
            print(f"Total steps processed: {stats['step_count']}")
            print(f"Episodes completed: {stats['recent_episodes']}")
            print(f"Action distribution: {stats['action_distribution']}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to get final stats: {e}")

if __name__ == "__main__":
    test_server_performance()