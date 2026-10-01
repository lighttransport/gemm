"""Differentiable geometry/radiance shared by appearance fitting and inference."""


def deform_torch(vertices, triangles, arrays, controls=None, policy="eigen-v1"):
    import torch as t
    a = arrays
    points = vertices[triangles[a["triangle"]]]
    e1, e2 = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    normal = t.linalg.cross(e1, e2)
    length = t.linalg.vector_norm(normal, dim=1)
    normal = normal / length.clamp_min(1e-12)[:, None]
    basis = t.stack((e1, e2, normal), -1)
    centers = (points * a["barycentric"][..., None]).sum(1) + normal * a["normal_offset"][:, None]
    cov = basis @ a["covariance_local"] @ basis.transpose(1, 2)
    if policy == "trace-v1":
        # SPD floor without unstable eigenvector gradients at repeated eigenvalues.
        cov = cov + t.eye(3, device=cov.device, dtype=cov.dtype) * 1e-10
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
        cov = cov * (1e-4 / trace.clamp_min(1e-10)).clamp_max(1)[:, None, None]
    elif policy == "eigen-v1":
        eig, rot = t.linalg.eigh(cov)
        cov = (rot * eig.clamp(1e-10, 1e-4)[:, None, :]) @ rot.transpose(1, 2)
    else:
        raise ValueError("unknown covariance policy")
    rgb = a["rgb"]
    if controls is not None:
        coeff = t.tanh(controls @ a["expression_matrix"])
        rgb = rgb + t.einsum("nkr,k->nr", a["color_basis"], coeff)
    rgb = rgb.clamp(0, 1) if policy == "trace-v1" else rgb.clamp_min(0)
    return centers, cov, a["opacity"] * (length > 1e-10), rgb
