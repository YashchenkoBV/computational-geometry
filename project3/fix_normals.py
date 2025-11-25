#!/usr/bin/env python3
"""
fix_normals.py

Read a triangle surface mesh in OFF / NOFF format,
re-orient all triangles consistently (using BFS over
manifold edges), compute smooth per-vertex normals,
and write the result as NOFF.

Usage:
    python fix_normals.py input.off output.off
"""

import sys
import numpy as np


def read_off(path):
    """Read OFF or NOFF; ignore any normals in the input."""
    with open(path, "r") as f:
        # Header
        line = f.readline()
        while line.strip().startswith("#") or not line.strip():
            line = f.readline()
        header = line.strip()
        if header not in ("OFF", "NOFF"):
            raise ValueError(f"Expected OFF or NOFF, got {header!r}")

        # Counts
        line = f.readline().strip()
        while line.startswith("#") or not line:
            line = f.readline().strip()
        parts = line.split()
        if len(parts) < 2:
            raise ValueError("Invalid OFF counts line.")
        n_verts, n_faces = int(parts[0]), int(parts[1])

        # Vertices (x y z ...)
        verts = []
        while len(verts) < n_verts:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            x, y, z = map(float, vals[:3])
            verts.append((x, y, z))
        vertices = np.array(verts, dtype=float)

        # Faces (assume triangles)
        faces = []
        while len(faces) < n_faces:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            k = int(vals[0])
            idx = list(map(int, vals[1:1 + k]))
            if k != 3:
                raise ValueError("This tool assumes triangular faces only.")
            faces.append(tuple(idx))

    return vertices, faces


def write_noff(path, vertices, normals, faces):
    """Write NOFF: vertices with normals + triangular faces."""
    n_verts = vertices.shape[0]
    n_faces = len(faces)
    with open(path, "w") as f:
        f.write("NOFF\n")
        f.write(f"{n_verts} {n_faces} 0\n")
        for v, n in zip(vertices, normals):
            f.write(f"{v[0]} {v[1]} {v[2]} {n[0]} {n[1]} {n[2]}\n")
        for (i, j, k) in faces:
            f.write(f"3 {i} {j} {k}\n")


# ----------------------------------------------------------------------
# Orientation logic (with BFS)
# ----------------------------------------------------------------------

def edge_orientation(tri, a, b):
    """
    For triangle tri = (v0, v1, v2) and edge {a,b},
    return +1 if a->b appears in tri,
           -1 if b->a appears in tri,
            0 if tri does not contain edge {a,b}.
    """
    v0, v1, v2 = tri
    edges = [(v0, v1), (v1, v2), (v2, v0)]
    for x, y in edges:
        if x == a and y == b:
            return 1
        if x == b and y == a:
            return -1
    return 0


def build_manifold_adjacency(faces):
    """
    Build face adjacency *only across manifold edges*:
    edges that are incident to exactly two faces.

    neighbors[i] is a list of (j, edge) pairs.
    """
    n_faces = len(faces)
    edge_to_faces = {}
    for fi, (a, b, c) in enumerate(faces):
        for e in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(e))
            edge_to_faces.setdefault(key, []).append(fi)

    neighbors = [[] for _ in range(n_faces)]
    for edge, flist in edge_to_faces.items():
        if len(flist) == 2:  # manifold edge
            f1, f2 = flist
            neighbors[f1].append((f2, edge))
            neighbors[f2].append((f1, edge))
    return neighbors


def orient_faces_locally(faces):
    """
    Use BFS to make triangle orientations locally consistent
    across manifold edges.
    """
    n_faces = len(faces)
    neighbors = build_manifold_adjacency(faces)
    faces_oriented = [list(t) for t in faces]
    visited = [False] * n_faces
    comp_id = [-1] * n_faces

    from collections import deque
    comp = 0

    for start in range(n_faces):
        if visited[start]:
            continue

        visited[start] = True
        comp_id[start] = comp
        queue = deque([start])

        while queue:
            f = queue.popleft()
            tri_f = faces_oriented[f]

            for g, edge in neighbors[f]:
                if visited[g]:
                    continue
                tri_g = faces_oriented[g]
                a, b = edge

                of = edge_orientation(tri_f, a, b)
                og = edge_orientation(tri_g, a, b)

                # If both faces traverse the edge and in the same direction,
                # flip the neighbor so they are opposite along that edge.
                if of != 0 and og != 0 and of == og:
                    tri_g = [tri_g[0], tri_g[2], tri_g[1]]
                    faces_oriented[g] = tri_g

                visited[g] = True
                comp_id[g] = comp
                queue.append(g)

        comp += 1

    return faces_oriented, comp_id


def signed_volume(vertices, faces_idx, faces):
    """
    Signed volume contributed by faces in faces_idx.

    V = (1/6) * sum ( (v0 x v1) · v2 )
    """
    vol = 0.0
    for fi in faces_idx:
        i, j, k = faces[fi]
        v0, v1, v2 = vertices[i], vertices[j], vertices[k]
        vol += np.dot(np.cross(v0, v1), v2)
    return vol / 6.0


def orient_faces_globally(vertices, faces):
    """
    1) BFS-based local consistency across manifold edges.
    2) For each connected component separately, flip the whole
       component if its signed volume is negative so normals
       point roughly outward.
    """
    faces_oriented, comp_id = orient_faces_locally(faces)
    n_faces = len(faces_oriented)

    # Group faces by component
    comps = {}
    for f in range(n_faces):
        cid = comp_id[f]
        comps.setdefault(cid, []).append(f)

    # For each component, apply a global flip if needed
    for cid, flist in comps.items():
        if not flist:
            continue
        vol = signed_volume(vertices, flist, faces_oriented)
        if vol < 0.0:
            for f in flist:
                i, j, k = faces_oriented[f]
                faces_oriented[f] = [i, k, j]

    return faces_oriented


# ----------------------------------------------------------------------
# Vertex normals
# ----------------------------------------------------------------------

def compute_vertex_normals(vertices, faces):
    """
    Smooth per-vertex normals as normalized sum of incident face normals.
    """
    n_verts = vertices.shape[0]
    normals = np.zeros((n_verts, 3), dtype=float)

    for i, j, k in faces:
        v0, v1, v2 = vertices[i], vertices[j], vertices[k]
        n = np.cross(v1 - v0, v2 - v0)  # area-weighted face normal
        normals[i] += n
        normals[j] += n
        normals[k] += n

    for idx in range(n_verts):
        l = np.linalg.norm(normals[idx])
        if l > 0.0:
            normals[idx] /= l
        else:
            normals[idx] = np.array([0.0, 0.0, 1.0])

    return normals


def main(argv):
    if len(argv) != 3:
        print("Usage: python fix_normals.py input.off output.off")
        return

    in_path, out_path = argv[1], argv[2]
    vertices, faces = read_off(in_path)

    # 1. Orient faces coherently (BFS + per-component volume)
    faces_oriented = orient_faces_globally(vertices, faces)

    # 2. Compute smooth vertex normals
    normals = compute_vertex_normals(vertices, faces_oriented)

    # 3. Write NOFF
    write_noff(out_path, vertices, normals, faces_oriented)
    print(f"Wrote NOFF with fixed normals to {out_path}")


if __name__ == "__main__":
    main(sys.argv)
