import numpy as np

C3D4_FACE = {
    "S1": (0, 1, 2),
    "S2": (0, 3, 1),
    "S3": (1, 3, 2),
    "S4": (2, 3, 0),
}


def read_nodes(path):
    ids, xs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            ids.append(int(p[0]))
            xs.append([float(p[1]), float(p[2]), float(p[3])])
    ids = np.asarray(ids, dtype=np.int64)
    xs = np.asarray(xs, dtype=np.float64)
    n = ids.max()
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[ids - 1] = xs
    return coords


def read_elems(path):
    cells = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = [int(x) for x in line.split(",")]
            cells.append([p[1] - 1, p[2] - 1, p[3] - 1, p[4] - 1])
    return np.asarray(cells, dtype=np.int64)


def _expand(tokens, generate):
    if generate:
        a, b, s = (int(t) for t in tokens[:3])
        return list(range(a, b + 1, s))
    return [int(t) for t in tokens if t != ""]


def read_nsets(path):
    sets, cur, gen = {}, None, False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("*NSET"):
                cur = s.split("NSET=")[1].split(",")[0].strip()
                gen = "GENERATE" in s.upper()
                sets[cur] = []
                continue
            if s.startswith("*"):
                cur = None
                continue
            if cur is not None:
                sets[cur].extend(_expand(s.rstrip(",").split(","), gen))
    return {k: np.asarray(v, dtype=np.int64) - 1 for k, v in sets.items()}


def read_surfaces(path, cells):
    surfs, cur = {}, None
    raw = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("*SURFACE"):
                cur = s.split("NAME=")[1].split(",")[0].strip()
                raw[cur] = []
                continue
            if s.startswith("*"):
                cur = None
                continue
            if cur is not None:
                p = s.split(",")
                raw[cur].append((int(p[0]) - 1, p[1].strip().upper()))
    for name, faces in raw.items():
        tris = []
        for elem0, face in faces:
            loc = C3D4_FACE[face]
            tet = cells[elem0]
            tris.append([tet[loc[0]], tet[loc[1]], tet[loc[2]]])
        surfs[name] = np.asarray(tris, dtype=np.int64)
    return surfs


if __name__ == "__main__":
    import sys
    g = sys.argv[1] if len(sys.argv) > 1 else "0"
    base = f"mesh/mesh_files/{g}_"
    coords = read_nodes(base + "node.txt")
    cells = read_elems(base + "elem.txt")
    nsets = read_nsets(base + "set.txt")
    surfs = read_surfaces(base + "surf.txt", cells)
    print(f"geom {g}: {coords.shape[0]} nodes, {cells.shape[0]} tets")
    print("bbox min", coords.min(0), "max", coords.max(0))
    print("nsets:", {k: len(v) for k, v in nsets.items()})
    print("surfs:", {k: len(v) for k, v in surfs.items()})
