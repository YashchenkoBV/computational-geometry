#!/usr/bin/env python3
"""
fix_normals.py

Read a triangle surface mesh in OFF / NOFF format,
re-orient all triangles consistently, compute smooth
per-vertex normals, and write the result as NOFF.

Usage:
    python fix_normals.py input.off output.off
"""

import sys
import numpy as np


# ----------------------------------------------------------------------
# OFF / NOFF I/O
# ----------------------------------------------------------------------

def read_off(path):
    """
    Read OFF or NOFF file.

    Returns
    -------
    vertices : (n, 3) float array
    faces    : list[(i, j, k)]  triangle indices
    """
    with open(path, "r") as f:
        # Read header: OFF or NOFF
        line = f.readline()
        while line.strip().startswith("#") or not line.strip():
            line = f.readline()
        header = line.strip()
        if header not in ("OFF", "NOFF"):
            raise ValueError("Unsupported header: expected OFF or NOFF")

        # Read counts
        line = f.readline().strip()
        while line.startswith("#") or not line:
            line = f.readline().strip()
        parts = line.split()
        if len(parts) < 2:
            raise ValueError("Invalid OFF counts line.")
        n_verts, n_faces = int(parts[0]), int(parts[1])

        # Read vertices (ignore any normals/colors in input)
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

        # Read faces (assume triangles)
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
            idx = list(map(int, vals[1:1+k]))
            if k != 3:
                raise ValueError("This tool assumes triangular faces only.")
            faces.append(tuple(idx))

    return vertices, faces


def write_noff(path, vertices, normals, faces):
    """
    Write NOFF (OFF with per-vertex normals).

    First line: NOFF
    Then: n_verts n_faces 0
    Then: x y z nx ny nz  per vertex
    Then: 3 i j k         per face
    """
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
# Orientation helpers
# ----------------------------------------------------------------------

def edge_orientation(tri, a, b):
    """
    For triangle tri = (v0, v1, v2) and edge {a,b},
    return +1 if edge appears as a -> b,
           -1 if edge appears as b -> a,
            0 if edge not in triangle.
    """
    v0, v1, v2 = tri
    edges = [(v0, v1), (v1, v2), (v2, v0)]
    for x, y in edges:
        if x == a and y == b:
            return 1
        if x == b and y == a:
            return -1
    return 0


def build_adjacency(faces):
    """
    Build face adjacency via shared edges.

    neighbors[i] is a list of (j, edge) where j is
    index of neighbor face and edge is the unordered
    pair (a,b) that they share.
    """
    n_faces = len(faces)
    edge_to_faces = {}
    for fi, (a, b, c) in enumerate(faces):
        for e in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(e))
            edge_to_faces.setdefault(key, []).append(fi)

    neighbors = [[] for _ in range(n_faces)]
    for edge, flist in edge_to_faces.items():
        if len(flist) == 2:
            f1, f2 = flist
            neighbors[f1].append((f2, edge))
            neighbors[f2].append((f1, edge))
    return neighbors


def orient_faces_locally(vertices, faces):
    """
    Make triangle orientations locally consistent using BFS.

    Returns list of oriented (i, j, k) triples.
    """
    n_faces = len(faces)
    neighbors = build_adjacency(faces)
    faces_oriented = [list(t) for t in faces]
    visited = [False] * n_faces

    from collections import deque

    for start in range(n_faces):
        if visited[start]:
            continue
        visited[start] = True
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
                if of != 0 and og != 0 and of == og:
                    # same direction -> flip neighbor
                    tri_g = [tri_g[0], tri_g[2], tri_g[1]]
                    faces_oriented[g] = tri_g
                visited[g] = True
                queue.append(g)

    return faces_oriented


def signed_volume(vertices, faces):
    """
    Signed volume of closed triangulated surface:

        V = (1/6) * sum ( (v0 x v1) · v2 )
    """
    vol = 0.0
    for i, j, k in faces:
        v0 = vertices[i]
        v1 = vertices[j]
        v2 = vertices[k]
        vol += np.dot(np.cross(v0, v1), v2)
    return vol / 6.0


def orient_faces_globally(vertices, faces):
    """
    First enforce local consistency, then choose the global
    orientation so that the enclosed volume is positive
    (normals pointing outward).
    """
    faces_oriented = orient_faces_locally(vertices, faces)
    vol = signed_volume(vertices, faces_oriented)
    if vol < 0.0:
        faces_oriented = [(i, k, j) for (i, j, k) in faces_oriented]
    return faces_oriented


# ----------------------------------------------------------------------
# Normal computation
# ----------------------------------------------------------------------

def compute_vertex_normals(vertices, faces):
    """
    Compute smooth per-vertex normals as the normalized sum
    of incident face normals.

    For triangle (i,j,k) with oriented vertices:
        n_f = normalize( (v_j - v_i) x (v_k - v_i) )

    Then vertex normal:
        N_i = normalize( sum_{faces incident to i} n_f )
    """
    n_verts = vertices.shape[0]
    normals = np.zeros((n_verts, 3), dtype=float)

    # accumulate area-weighted face normals
    for i, j, k in faces:
        v0 = vertices[i]
        v1 = vertices[j]
        v2 = vertices[k]
        n = np.cross(v1 - v0, v2 - v0)
        normals[i] += n
        normals[j] += n
        normals[k] += n

    # normalize
    for idx in range(n_verts):
        length = np.linalg.norm(normals[idx])
        if length > 0.0:
            normals[idx] /= length
        else:
            normals[idx] = np.array([0.0, 0.0, 1.0])

    return normals


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv):
    if len(argv) != 3:
        print("Usage: python fix_normals.py input.off output.off")
        return

    in_path, out_path = argv[1], argv[2]
    vertices, faces = read_off(in_path)

    # 1. Make triangle orientations coherent and outward
    faces_oriented = orient_faces_globally(vertices, faces)

    # 2. Compute smooth vertex normals from these oriented faces
    normals = compute_vertex_normals(vertices, faces_oriented)

    # 3. Write NOFF so viewer uses our normals for shading
    write_noff(out_path, vertices, normals, faces_oriented)
    print(f"Wrote NOFF with fixed normals to {out_path}")


if __name__ == "__main__":
    main(sys.argv)
