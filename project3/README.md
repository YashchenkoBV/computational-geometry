### fix_normals.py

**Purpose:** `fix_normals.py` fixes inconsistent triangle orientations (flipped facets) in an OFF surface mesh and computes smooth per-vertex normals for correct shading.

**How to run:**

```bash
pip install -r requirements.txt

cd project3

python fix_normals.py input.off output.off
```
* input.off — input mesh in OFF or NOFF format (triangles only).
* output.off — output mesh in NOFF format, with reoriented faces and per-vertex normals.