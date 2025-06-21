import numpy as np

def reprojection_errors(
    points_3d: np.ndarray,
    camera_mats: np.ndarray,
    K: np.ndarray,
    obs_uv: np.ndarray,
    cam_idxs: np.ndarray,
    pt_idxs: np.ndarray
) -> np.ndarray:
    """
    计算所有观测的重投影误差 (N_obs,)。
    """
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # 按观测索引批量投影
    Rs = camera_mats[cam_idxs, :3, :3]    # (N_obs,3,3)
    ts = camera_mats[cam_idxs, :3,  3]    # (N_obs,3)
    Xs = points_3d[pt_idxs]               # (N_obs,3)

    # 相机坐标系下
    Xc = (Rs @ Xs[...,None]).squeeze(-1) + ts  # (N_obs,3)
    us = fx * (Xc[:,0] / Xc[:,2]) + cx
    vs = fy * (Xc[:,1] / Xc[:,2]) + cy

    proj = np.stack([us, vs], axis=1)      # (N_obs,2)
    errs = np.linalg.norm(proj - obs_uv, axis=1)
    return errs

def reprojection_stats(errs: np.ndarray) -> dict:
    """
    输入 errs = reprojection_errors(...)，
    返回平均/中位/最大 重投影误差。
    """
    return {
        'mean':  np.mean(errs),
        'median':np.median(errs),
        'max':   np.max(errs),
        'std':   np.std(errs),
    }

def track_length_stats(
    point_observations: dict
) -> dict:
    """
    输入 point_observations: pid -> {cam_idx: (u,v), ...}
    返回每点观测的相机数的分布统计。
    """
    lengths = np.array([len(obs) for obs in point_observations.values()])
    return {
        'mean':   lengths.mean(),
        'median': lengths.mean() if len(lengths)==0 else np.median(lengths),
        'max':    int(lengths.max()) if len(lengths)>0 else 0,
        'hist':   np.bincount(lengths)
    }

def image_coverage_stats(
    point_observations: dict,
    n_cameras: int
) -> dict:
    """
    计算每张图像“能看到”的三维点数分布。
    """
    counts = np.zeros(n_cameras, dtype=int)
    for obs in point_observations.values():
        for c in obs.keys():
            counts[c] += 1
    return {
        'per_image': counts,
        'mean':      counts.mean(),
        'median':    np.median(counts),
        'min':       counts.min(),
        'max':       counts.max()
    }

def points_per_camera_ratio(
    points_3d: np.ndarray,
    camera_mats: np.ndarray
) -> float:
    """
    稀疏点总数 / 相机总数
    """
    return float(points_3d.shape[0]) / float(camera_mats.shape[0])
