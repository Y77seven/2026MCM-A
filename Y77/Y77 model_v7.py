import numpy as np

# 缩放定律应用
def apply_scaling_law(Y_ref, X_ref, X_tar, beta):
    return Y_ref * (X_tar / X_ref) ** beta

# 逻辑增长模型右侧函数
def logistic_rhs(X, t, r, K):
    return r * X * (1 - X / K)

# Sigmoid 过渡函数
def sigmoid_transition(x, x0, k, low, high):
    # Smoothly transitions from low to high
    return low + (high-low)/(1+np.exp(-k*(x-x0)))

# 数值雅可比矩阵计算
def numeric_jacobian(f, x, eps=1e-6):
    x = np.array(x, dtype=float)
    n = x.size
    J = np.zeros((n,n))
    fx = np.array(f(x))
    for i in range(n):
        x2 = x.copy(); x2[i] += eps
        J[:,i] = (np.array(f(x2)) - fx) / eps
    return J

# 蒙特卡洛置信带计算
def monte_carlo_band(simulate_once, runs=1000, quantiles=(2.5,50,97.5), seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(runs):
        results.append(simulate_once(rng))
    arr = np.array(results)
    q = np.percentile(arr, quantiles, axis=0)
    return q  # shape: (len(quantiles), T)

# 一维显式扩散方程求解
def solve_diffusion_1d(u0, D, dt, dx, steps, bc='neumann', u_left=0.0, u_right=0.0):
    """Explicit finite difference for 1D diffusion.
    Stability (minimum statement for paper): r = D*dt/dx^2 <= 0.5.
    bc: 'neumann' (no-flux) or 'dirichlet' (fixed value).
    """
    r = D*dt/dx**2
    if r > 0.5:
        raise ValueError(f'Unstable explicit scheme: r={r:.3f} > 0.5. Reduce dt or increase dx.')
    u = u0.astype(float).copy()
    for _ in range(steps):
        un = u.copy()
        u[1:-1] = un[1:-1] + r*(un[2:] - 2*un[1:-1] + un[:-2])
        if bc == 'neumann':
            # no-flux: du/dn=0 -> mirror boundary
            u[0] = u[1]
            u[-1] = u[-2]
        elif bc == 'dirichlet':
            u[0] = u_left
            u[-1] = u_right
        else:
            raise ValueError('bc must be neumann or dirichlet')
    return u

# 图拉普拉斯矩阵及图上扩散计算
def graph_laplacian(A):
    # A: adjacency matrix
    D = np.diag(A.sum(axis=1))
    return D - A

def diffuse_on_graph(u, A, Dcoef, dt, steps, reaction=None):
    L = graph_laplacian(A)
    u = u.astype(float).copy()
    for _ in range(steps):
        du = -Dcoef * (L @ u)
        if reaction is not None:
            du = du + reaction(u)
        u = u + dt*du
    return u

# 帕累托非支配点筛选
def pareto_nondominated(points):
    # points: list of (J1,J2) to minimize
    nd = []
    for i,p in enumerate(points):
        dominated = False
        for j,q in enumerate(points):
            if j==i: continue
            if (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break
        if not dominated:
            nd.append(p)
    return nd

# 帕累托前沿拐点计算
def knee_point(front):
    """Pick a knee point on a 2D Pareto front (both objectives minimized).
    Uses max distance to the line connecting the two extreme points.
    front: list of (J1, J2)
    returns: (J1, J2)
    """
    pts = np.array(front, dtype=float)
    if len(pts) == 0:
        return None
    # sort by J1
    pts = pts[np.argsort(pts[:,0])]
    A = pts[0]; B = pts[-1]
    AB = B - A
    denom = np.linalg.norm(AB)
    if denom < 1e-12:
        return tuple(pts[len(pts)//2])
    # distance from each point to line AB
    d = np.abs(np.cross(AB, pts - A) / denom)
    idx = int(np.argmax(d))
    return tuple(pts[idx])