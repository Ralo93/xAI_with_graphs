import json
import subprocess

# Generate test data
test_data = {
    "node_features": [[i % 2 for i in range(1433)] for _ in range(3)],
    "edge_index": [[0, 1], [1, 2], [2, 0]],
}

# Save to file
with open("test_input.json", "w") as f:
    json.dump(test_data, f)

# Execute curl command
curl_command = "curl -X POST http://localhost:8080/api/v1/predict/ -H 'Content-Type: application/json' -d @test_input.json"
subprocess.run(curl_command, shell=True)
