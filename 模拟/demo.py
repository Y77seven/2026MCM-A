import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. 物理常数与环境
g0 = 9.80665
R_e = 6371000
rho0 = 1.225
H_scale = 8500
T_ambient_surface = 288.15  # 海平面温度 (K)
gamma = 1.4  # 空气比热比

# 2. 高级火箭参数
# 比冲模型：随高度变化
Isp_sea = 365.0  # 海平面比冲
Isp_vac = 460.0  # 真空比冲 (稍微优化以确保逃逸)
H_isp_scale = 5000  # 比冲过渡高度


def get_isp(h):
    # 使用简单的指数过渡模型
    return Isp_vac - (Isp_vac - Isp_sea) * np.exp(-h / H_isp_scale)

#epsilon = 0.04 第一次
epsilon = 0.033  # 激进的结构系数 (碳纤维)
m_payload = 200.0  # 微型探测器
m_total = 100000.0

m_structure = m_total * epsilon
m_propellant = m_total - m_structure - m_payload
m_final = m_structure + m_payload

# 推重比设置 (按海平面推力计算)
TWR = 1.5
Thrust_sea = m_total * g0 * TWR

# 气动参数
Cd = 0.3
Diameter = 3.5
Area = np.pi * (Diameter / 2) ** 2


# 3. 动力学模型 (含热力学计算)
def rocket_ode(t, y):
    h = y[0]
    v = y[1]
    m = y[2]

    if m > m_final:
        # 当前高度的比冲
        current_isp = get_isp(h)
        # 推力 (假设燃料流率恒定，或者推力恒定，这里假设推力恒定更简单)
        Thrust = Thrust_sea
        ve = current_isp * g0
        dm_dt = -Thrust / ve
    else:
        Thrust = 0
        dm_dt = 0
        m = m_final

    # 环境计算
    r = R_e + h
    g = g0 * (R_e / r) ** 2
    rho = rho0 * np.exp(-h / H_scale)

    # 气动阻力
    Drag = 0.5 * rho * v ** 2 * Cd * Area

    # 动力学方程
    dv_dt = (Thrust - Drag) / m - g
    dh_dt = v

    return [dh_dt, dv_dt, dm_dt]


# 求解事件：燃料耗尽
def burnout_event(t, y):
    return y[2] - m_final


burnout_event.terminal = True
burnout_event.direction = -1

# 求解 (燃烧阶段)
y0 = [0.0, 0.0, m_total]
t_span = (0, 1000)

sol = solve_ivp(rocket_ode, t_span, y0, events=burnout_event, rtol=1e-6, max_step=0.5)

# 滑行段模拟 (Burnout -> Final)
y_burnout = sol.y[:, -1]
t_coast = np.linspace(sol.t[-1], sol.t[-1] + 200, 200)
sol_coast = solve_ivp(rocket_ode, [sol.t[-1], sol.t[-1] + 300], y_burnout, t_eval=t_coast)

# 合并数据
t_all = np.concatenate((sol.t, sol_coast.t))
h_all = np.concatenate((sol.y[0], sol_coast.y[0]))
v_all = np.concatenate((sol.y[1], sol_coast.y[1]))

# 4. 后处理：计算热与动压
rho_all = rho0 * np.exp(-h_all / H_scale)
# 声速 a = sqrt(gamma * R * T)，这里简化假设温度随高度线性下降然后恒定
T_ambient = np.maximum(T_ambient_surface - 0.0065 * h_all, 216.65)
c_sound = np.sqrt(1.4 * 287 * T_ambient)
Mach_all = v_all / c_sound

# 动压 q
q_all = 0.5 * rho_all * v_all ** 2

# 驻点温度 (Stagnation Temp) - 气动加热指标
T_stag = T_ambient * (1 + 0.2 * Mach_all ** 2)

# 计算关键指标
v_burnout = np.max(v_all)
escape_velocity = np.sqrt(2 * g0 * R_e)

plt.figure(figsize=(14, 10))

# 图1: 速度
plt.subplot(2, 2, 1)
plt.plot(t_all, v_all, 'b', linewidth=2)
plt.axhline(11200, color='r', linestyle='--', label='Escape Vel (11.2 km/s)')
plt.title('Velocity Profile')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()

# 图2: 动压 (Max Q)
plt.subplot(2, 2, 2)
plt.plot(t_all, q_all / 1000, 'orange', linewidth=2)
plt.axhline(35, color='r', linestyle=':', label='Limit (35 kPa)')
plt.title('Dynamic Pressure (Max Q)')
plt.ylabel('Dynamic Pressure (kPa)')
plt.legend()
plt.grid(True)

# 图3: 气动加热
plt.subplot(2, 2, 3)
plt.plot(t_all, T_stag, 'r', linewidth=2)
plt.title('Aerodynamic Heating (Stagnation Temp)')
plt.ylabel('Temperature (K)')
plt.xlabel('Time (s)')
plt.grid(True)

# 图4: 高度
plt.subplot(2, 2, 4)
plt.plot(t_all, h_all / 1000, 'g', linewidth=2)
plt.title('Altitude Profile')
plt.ylabel('Altitude (km)')
plt.xlabel('Time (s)')
plt.grid(True)

plt.tight_layout()
plt.savefig('rocket_simulation_results.png')
plt.show()


print(f"--- 火箭参数 ---")
print(f"总质量: {m_total / 1000:.1f} 吨")
print(f"干重(结构+载荷): {m_final / 1000:.2f} 吨")
print(f"质量比 (R): {m_total / m_final:.2f}")
# print(f"燃烧时间: {burn_time:.2f} 秒") # burn_time 变量在函数外可能不准，直接用 sol.t[-1]
print(f"燃烧时间: {sol.t[-1]:.2f} 秒")
# print(f"排气速度: {ve:.2f} m/s") # ve 随高度变化，这里不打印定值

print(f"最终速度: {v_all[-1]:.2f} m/s")
print(f"最大动压: {np.max(q_all) / 1000:.2f} kPa")
print(f"最大表面温度估计: {np.max(T_stag):.0f} K")

print(f"\n--- 模拟结果 ---")
print(f"最大速度 (Burnout Velocity): {v_burnout:.2f} m/s")
print(f"地球逃逸速度: {escape_velocity:.2f} m/s")

if v_burnout > escape_velocity:
    print("结论: 成功达到逃逸速度！")
else:
    print(f"结论: 未达到逃逸速度 (差 {escape_velocity - v_burnout:.2f} m/s)")
    print("建议: 尝试减小结构系数 epsilon 或 增加比冲 Isp")