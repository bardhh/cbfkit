import os
import json
import datetime

TRACKING_FILE = ".jules/examples_tutorials_tracking.md"
OUTPUT_FILE = ".jules/relay_tasks.json"

def load_exclusions():
    exclusions = set()
    if not os.path.exists(TRACKING_FILE):
        return exclusions

    with open(TRACKING_FILE, 'r') as f:
        content = f.read()

    # Simple parsing: look for filenames in the table or list
    # Assuming the file format described in the prompt
    lines = content.split('\n')
    for line in lines:
        if "|" in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) > 2:
                filename = parts[1]
                if filename and filename != "File" and "---" not in filename:
                    exclusions.add(filename)
    return exclusions

def analyze_file(filepath):
    complexity = "low"
    reason = []
    loc = 0
    content = ""
    is_ipynb = filepath.endswith('.ipynb')

    try:
        if is_ipynb:
            with open(filepath, 'r') as f:
                nb = json.load(f)
                code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
                for cell in code_cells:
                    source = cell.get('source', [])
                    loc += len(source)
                    content += "".join(source) + "\n"
        else:
            with open(filepath, 'r') as f:
                content = f.read()
                loc = len(content.split('\n'))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Static Analysis Indicators
    imports_jax = "jax" in content
    imports_ros = "rclpy" in content or "rospy" in content
    imports_cvxpy = "cvxpy" in content

    patterns_complex_jax = "scan(" in content or "custom_vjp" in content or "lax.scan" in content
    patterns_basic_jax = "jit(" in content or "vmap(" in content or "grad(" in content or "@jit" in content
    patterns_solver = "Solver" in content or "Controller" in content or "solve(" in content
    patterns_viz = "matplotlib" in content or "plt." in content

    # Determine Complexity
    score = 0
    reasons_list = []

    if loc < 100:
        score += 1
        reasons_list.append("<100 LOC")
    elif loc < 300:
        score += 2
        reasons_list.append("100-300 LOC")
    else:
        score += 3
        reasons_list.append(">300 LOC")

    if imports_ros:
        score += 10 # Force High
        reasons_list.append("ROS integration")

    if patterns_complex_jax:
        score += 3
        reasons_list.append("Complex JAX")

    if imports_cvxpy or "qpsolvers" in content:
        score += 2
        reasons_list.append("External Solver")

    if patterns_solver:
        score += 1
        reasons_list.append("Solver/Controller")

    if patterns_basic_jax:
        score += 1
        reasons_list.append("Basic JAX")

    if patterns_viz:
        reasons_list.append("Viz")

    # Classification
    if score >= 3 or loc > 300 or imports_ros or patterns_complex_jax:
        complexity = "high"
    elif score >= 2 or loc >= 100 or patterns_basic_jax or patterns_solver or patterns_viz:
        complexity = "medium"
    else:
        complexity = "low"

    return {
        "filepath": filepath,
        "complexity": complexity,
        "reason": f"{', '.join(reasons_list)}",
        "loc": loc
    }

def discover_files(exclusions):
    files_to_process = []
    dirs_to_scan = ['examples', 'tutorials']

    for d in dirs_to_scan:
        for root, dirs, files in os.walk(d):
            # Exclude visualizations and pycache
            if 'visualizations' in root or '__pycache__' in root:
                continue

            for file in files:
                if file == "__init__.py":
                    continue
                if not (file.endswith('.py') or file.endswith('.ipynb')):
                    continue

                filepath = os.path.join(root, file)
                if filepath in exclusions:
                    continue

                analysis = analyze_file(filepath)
                if analysis:
                    files_to_process.append(analysis)

    return files_to_process

def create_batches(analyzed_files):
    high = [f for f in analyzed_files if f['complexity'] == 'high']
    medium = [f for f in analyzed_files if f['complexity'] == 'medium']
    low = [f for f in analyzed_files if f['complexity'] == 'low']

    print(f"Distribution: High={len(high)}, Medium={len(medium)}, Low={len(low)}")

    batches = []
    batch_id = 1

    # Fill batches with High complexity files (max 2 per batch)
    while high:
        batch_files = []
        count = min(len(high), 2)
        for _ in range(count):
            batch_files.append(high.pop(0))

        batches.append({
            "batch_id": batch_id,
            "files": [f['filepath'] for f in batch_files],
            "complexity": "high",
            "reason": "; ".join([f"{os.path.basename(f['filepath'])}: {f['reason']}" for f in batch_files])
        })
        batch_id += 1

    # Fill batches with Medium complexity files (max 3 per batch)
    while medium:
        batch_files = []
        count = min(len(medium), 3)
        for _ in range(count):
            batch_files.append(medium.pop(0))

        batches.append({
            "batch_id": batch_id,
            "files": [f['filepath'] for f in batch_files],
            "complexity": "medium",
            "reason": "; ".join([f"{os.path.basename(f['filepath'])}: {f['reason']}" for f in batch_files])
        })
        batch_id += 1

    # Fill batches with Low complexity files (max 6 per batch)
    while low:
        batch_files = []
        count = min(len(low), 6)
        for _ in range(count):
            batch_files.append(low.pop(0))

        batches.append({
            "batch_id": batch_id,
            "files": [f['filepath'] for f in batch_files],
            "complexity": "low",
            "reason": "; ".join([f"{os.path.basename(f['filepath'])}: {f['reason']}" for f in batch_files])
        })
        batch_id += 1

    return batches

def main():
    exclusions = load_exclusions()
    print(f"Loaded {len(exclusions)} exclusions.")

    analyzed_files = discover_files(exclusions)
    print(f"Discovered {len(analyzed_files)} files.")

    batches = create_batches(analyzed_files)
    print(f"Created {len(batches)} batches.")

    output = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_files": len(analyzed_files),
        "batches": batches
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
