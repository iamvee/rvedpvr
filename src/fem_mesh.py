import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

import basix.ufl
from dolfinx import mesh as dmesh

TAG = {"EPI": 1, "TOP": 2, "RV": 3, "LV": 4}


def log(*a):
    if MPI.COMM_WORLD.rank == 0:
        print(*a, flush=True)


def ipoints(V):
    ip = V.element.interpolation_points
    return ip() if callable(ip) else ip


def build_mesh(coords, cells, comm):
    el = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    c, x = cells.astype(np.int64), coords.astype(np.float64)
    try:
        return dmesh.create_mesh(comm, c, el, x)
    except (IndexError, TypeError, AttributeError):
        return dmesh.create_mesh(comm, c, x, el)


def tag_facets(msh, coords, surfs):
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, tdim)
    ext = dmesh.exterior_facet_indices(msh.topology)
    mid = dmesh.compute_midpoints(msh, fdim, ext)
    tree = cKDTree(mid)
    facets, values = [], []
    for key, sname in [("EPI", "epi_surf"), ("TOP", "top_surf"),
                       ("RV", "RV_surf"), ("LV", "LV_surf")]:
        cen = coords[surfs[sname]].mean(axis=1)
        _, idx = tree.query(cen)
        facets.append(ext[idx])
        values.append(np.full(len(idx), TAG[key], np.int32))
    facets = np.concatenate(facets)
    values = np.concatenate(values)
    fu, ui = np.unique(facets, return_index=True)
    return dmesh.meshtags(msh, fdim, fu.astype(np.int32), values[ui])


def vertex_coords(msh):
    msh.topology.create_connectivity(0, msh.topology.dim)
    imap = msh.topology.index_map(0)
    allv = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
    return dmesh.compute_midpoints(msh, 0, allv), allv
