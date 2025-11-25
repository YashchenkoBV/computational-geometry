import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def delaunay(points):
    """
    Delaunay triangulation of a 2D point set using
    lifting to the paraboloid and taking the lower
    part of the 3D convex hull.
    """
    points = np.asarray(points, dtype=float)

    # add z = x^2 + y^2 as a third coordinate
    z = np.sum(points ** 2, axis=1)
    points_3d = np.c_[points, z]

    # find convex hull with scipy
    hull = ConvexHull(points_3d)

    # As per the lecture, each facet has equation a*x + b*y + c*z + d <= 0 for hull interior.
    # The algorithm asks to filter out the upper hull
    # Thus, keep only facets with c < 0
    lower_facets = np.where(hull.equations[:, 2] < 0)[0]
    triangles = hull.simplices[lower_facets]
    return triangles


def circumcenter(a, b, c, eps=1e-12):
    """
    Finding circumcenter of triangle ABC in 2D.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    # skip (almost) collinear points
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < eps:
        return None

    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy

    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d

    return np.array([ux, uy])


def crust(points):
    """
    Crust algorithm for 2D point sets.

    Input:
        points : array of shape (n, 2)

    Output:
        edges : sorted list of unique edges (i, j) with i < j,
                where i and j are indices into points2d.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)

    # 1. Delaunay triangulation of P - lower convex hull of lifted points
    tri1 = delaunay(points)

    # 2. Find set C - circumcenters of the Delaunay triangles
    centers = []
    for ia, ib, ic in tri1:
        center = circumcenter(points[ia],
                              points[ib],
                              points[ic])
        if center is not None and np.all(np.isfinite(center)):
            centers.append(center)
    centers = np.asarray(centers)

    # 3. Delaunay triangulation of P ∪ C
    if centers.size == 0:
        tri2 = np.zeros((0, 3), dtype=int)
    else:
        union_points = np.vstack([points, centers])
        tri2 = delaunay(union_points)

    # 4. Filter out all the edges that have one or both endpoints in C
    edges = set()
    for ia, ib, ic in tri2:
        for u, v in ((ia, ib), (ib, ic), (ic, ia)):
            i, j = sorted((u, v))
            if i < n and j < n:
                edges.add((i, j))

    return sorted(edges)


def main():
    points = np.loadtxt("points.txt")

    # Output
    edges = crust(points)
    for i, j in edges:
        print(i, j)

    # Visualization
    plt.figure(figsize=(5, 5))
    plt.scatter(points[:, 0], points[:, 1], s=1, c='red', marker='.')
    for i, j in edges:
        plt.plot([points[i, 0], points[j, 0]],
                 [points[i, 1], points[j, 1]], c='red', linewidth=1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Crust reconstruction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
