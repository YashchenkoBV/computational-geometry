### crust.py

**Purpose:** `crust.py` reconstructs planar curves from a 2D point cloud using the Crust algorithm (via Delaunay triangulation on lifted points).

**How to run:**

```bash
pip install -r requirements.txt
python crust.py input_points.txt output_edges.txt
```
* input_points.txt — text file with one sample point per line: x y
* output_edges.txt — text file where each line is an edge as two 0-based vertex indices: i j
