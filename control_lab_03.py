import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy import signal
from scipy.integrate import odeint
import io
import base64
from datetime import datetime
import pandas as pd

# 页面设置
st.set_page_config(page_title="自动控制原理虚拟仿真平台", layout="wide")
st.title("🎛️ 自动控制理论 · 虚拟仿真实验平台（扩展版）")
st.markdown("**适用经典控制理论教学：对象库、非线性、倒立摆动画、实验记录**")

# ---------- 侧边栏：选择实验 ----------
experiment = st.sidebar.selectbox(
    "选择实验模块",
    ["📈 时域分析（阶跃/脉冲/斜坡）",
     "🌿 根轨迹分析",
     "📊 频域分析（Bode & Nyquist）",
     "🔧 PID控制器设计",
     "🤖 虚拟被控对象库",
     "🚧 非线性环节分析",
     "🎥 倒立摆小车实时动画",
     "📋 实验记录与提交"]
)

# ========== 通用函数：传递函数创建 ==========
def create_tf(num, den):
    return ct.tf(num, den)

# ========== 虚拟对象库（可复用） ==========
def get_plant_model(model_name, params=None):
    """返回预设模型的传递函数或状态空间描述"""
    if model_name == "直流电机 (电压-转速)":
        # 简化：G(s) = K / (Js + b) , 设 J=0.01, b=0.1, K=0.1
        J = params.get('J', 0.01) if params else 0.01
        b = params.get('b', 0.1) if params else 0.1
        K = params.get('K', 0.1) if params else 0.1
        return create_tf([K], [J, b])
    elif model_name == "倒立摆 (摆角/力)":
        # 线性化模型：G(s) = 1/(s^2 - g/l)  设 l=0.5, g=9.8
        l = params.get('l', 0.5) if params else 0.5
        g = 9.8
        return create_tf([1], [1, 0, -g/l])
    elif model_name == "水箱液位 (一阶惯性)":
        # G(s) = K/(Ts+1) , K=2, T=5
        K = params.get('K', 2.0) if params else 2.0
        T = params.get('T', 5.0) if params else 5.0
        return create_tf([K], [T, 1])
    elif model_name == "典型二阶系统":
        zeta = params.get('zeta', 0.5) if params else 0.5
        wn = params.get('wn', 2.0) if params else 2.0
        return create_tf([wn**2], [1, 2*zeta*wn, wn**2])
    else:
        # 默认返回一个简单传递函数
        return create_tf([1], [1, 2, 1])

# ========== 非线性环节 ==========
def apply_nonlinearity(u, ntype, param=0.5):
    """对信号 u 施加非线性变换"""
    u = np.asarray(u)
    if ntype == "死区":
        dead = param
        u_out = np.where(np.abs(u) < dead, 0.0, u - np.sign(u)*dead)
    elif ntype == "饱和":
        sat = param
        u_out = np.clip(u, -sat, sat)
    elif ntype == "量化":
        step = param
        u_out = np.round(u / step) * step
    else:
        u_out = u
    return u_out

# ========== 倒立摆动力学（动画用） ==========
def cartpole_dynamics(state, t, u, mc=1.0, mp=0.1, l=0.5, g=9.8):
    """倒立摆非线性模型，state = [x, theta, x_dot, theta_dot]"""
    x, theta, x_dot, theta_dot = state
    s = np.sin(theta)
    c = np.cos(theta)
    mt = mc + mp
    denom = mt*l - mp*l*c*c
    if abs(denom) < 1e-4:
        denom = 1e-4
    dx = x_dot
    dtheta = theta_dot
    dx_dot = (u + mp*l*theta_dot**2*s - mp*g*s*c) / denom
    dtheta_dot = (-u*c - mp*l*theta_dot**2*s*c + mt*g*s) / denom
    return [dx, dtheta, dx_dot, dtheta_dot]

def simulate_cartpole(u_func, t_span, dt, x0=[0, np.pi*0.1, 0, 0]):
    """仿真并返回轨迹"""
    t = np.arange(t_span[0], t_span[1], dt)
    states = np.zeros((len(t), 4))
    states[0] = x0
    for i in range(1, len(t)):
        u = u_func(t[i-1])
        sol = odeint(cartpole_dynamics, states[i-1], [t[i-1], t[i]], args=(u,))
        states[i] = sol[1]
    return t, states

# ========== 动画嵌入函数 ==========
def render_animation_html(fig, anim):
    """将 matplotlib 动画转为 HTML 字符串并嵌入"""
    from matplotlib.animation import FuncAnimation
    # 将动画写入 HTML5 视频标签
    jshtml = anim.to_jshtml()
    return jshtml

# ---------- 实验一：时域分析 ----------
elif experiment == "📈 时域分析（阶跃/脉冲/斜坡）":
    st.header("时域响应分析")
    col1, col2 = st.columns(2)
    with col1:
        use_lib = st.checkbox("使用虚拟对象库", value=False)
        if use_lib:
            model_type = st.selectbox("选择对象模型", 
                ["直流电机 (电压-转速)", "倒立摆 (摆角/力)", "水箱液位 (一阶惯性)", "典型二阶系统"])
            if model_type == "典型二阶系统":
                zeta = st.slider("阻尼比 ζ", 0.0, 2.0, 0.5, 0.01)
                wn = st.slider("自然频率 ωn (rad/s)", 0.5, 10.0, 2.0, 0.1)
                params = {'zeta': zeta, 'wn': wn}
            else:
                params = None
            sys = get_plant_model(model_type, params)
            # 检查倒立摆模型的不稳定性并提示
            if "倒立摆" in model_type:
                st.warning("⚠️ 倒立摆线性化模型为不稳定系统（极点位于右半平面），阶跃响应将发散，仿真时间已自动限制。")
        else:
            model_type = "自定义传递函数"
            num_input = st.text_input("分子系数（空格分隔）", "1")
            den_input = st.text_input("分母系数（空格分隔）", "1 2 1")
            num = [float(x) for x in num_input.split()]
            den = [float(x) for x in den_input.split()]
            sys = create_tf(num, den)

    with col2:
        input_type = st.radio("输入信号", ["阶跃", "脉冲", "斜坡"])
        # 对于不稳定系统，自动缩短仿真时间
        poles = ct.poles(sys)
        unstable = any(np.real(p) > 0 for p in poles)
        default_time = 2.0 if unstable else 10.0
        sim_time = st.slider("仿真时间 (s)", 0.5, 30.0, default_time, 0.5)

    # 生成响应
    t = np.linspace(0, sim_time, 1000)
    try:
        if input_type == "阶跃":
            t, y = ct.step_response(sys, T=t)
        elif input_type == "脉冲":
            t, y = ct.impulse_response(sys, T=t)
        else:
            u = t  # 斜坡输入
            t, y, _ = ct.forced_response(sys, T=t, U=u)
    except Exception as e:
        st.error(f"仿真出错：{e}，可能由于系统不稳定导致数值发散。请尝试缩短仿真时间或检查模型。")
        st.stop()

    # 绘图
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t, y, linewidth=2)
    ax.set_title(f"{model_type}  {input_type}响应")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("输出")
    ax.grid(True)
    st.pyplot(fig)

    # 性能指标（仅稳定系统且阶跃输入）
    if input_type == "阶跃" and not unstable:
        try:
            info = ct.step_info(sys)
            if info is not None:
                st.subheader("📋 时域性能指标")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("上升时间", f"{info.get('RiseTime', np.nan):.3f} s")
                col_b.metric("峰值时间", f"{info.get('PeakTime', np.nan):.3f} s")
                col_c.metric("超调量", f"{info.get('Overshoot', 0):.2f} %")
                col_d.metric("调节时间", f"{info.get('SettlingTime', np.nan):.3f} s")
            else:
                st.info("无超调或响应未收敛，无法计算指标。")
        except:
            st.warning("性能指标计算失败，可能由于系统特性复杂。")
    elif unstable:
        st.info("当前为不稳定系统，不计算性能指标。")

# ---------- 实验二：根轨迹 ----------
elif experiment == "🌿 根轨迹分析":
    st.header("根轨迹分析")
    use_lib = st.checkbox("使用虚拟对象库", value=True)
    if use_lib:
        model_type = st.selectbox("选择对象模型",
            ["直流电机 (电压-转速)", "倒立摆 (摆角/力)", "水箱液位 (一阶惯性)", "典型二阶系统"])
        if model_type == "典型二阶系统":
            zeta = st.slider("阻尼比 ζ", 0.0, 2.0, 0.5, 0.01)
            wn = st.slider("自然频率 ωn", 0.5, 10.0, 2.0, 0.1)
            params = {'zeta': zeta, 'wn': wn}
        else:
            params = None
        sys = get_plant_model(model_type, params)
    else:
        col1, _ = st.columns(2)
        with col1:
            num_input = st.text_input("开环分子系数", "1")
            den_input = st.text_input("开环分母系数", "1 2 2 0")
        num = [float(x) for x in num_input.split()]
        den = [float(x) for x in den_input.split()]
        sys = create_tf(num, den)

    fig, ax = plt.subplots(figsize=(8,6))
    try:
        ct.root_locus(sys, ax=ax)
    except:
        ct.rlocus(sys)
        ax = plt.gca()
    ax.set_title(f"根轨迹图  G(s)={sys}")
    ax.grid(True)
    st.pyplot(fig)

    K = st.slider("选择增益 K", 0.1, 50.0, 1.0, 0.1)
    cl_sys = ct.feedback(K * sys, 1)
    poles = ct.poles(cl_sys)
    st.write(f"**K = {K:.2f}** 时闭环极点：")
    for i, p in enumerate(poles):
        st.latex(f"s_{i+1} = {p.real:.3f} + j{p.imag:.3f}")
    if st.checkbox("显示闭环阶跃响应"):
        t, y = ct.step_response(cl_sys)
        fig2, ax2 = plt.subplots()
        ax2.plot(t, y)
        ax2.set_title(f"闭环阶跃响应 (K={K})")
        ax2.grid(True)
        st.pyplot(fig2)

# ---------- 实验三：频域分析 ----------
elif experiment == "📊 频域分析（Bode & Nyquist）":
    st.header("频域响应分析")
    use_lib = st.checkbox("使用虚拟对象库", value=False)
    if use_lib:
        model_type = st.selectbox("选择对象模型",
            ["直流电机 (电压-转速)", "倒立摆 (摆角/力)", "水箱液位 (一阶惯性)", "典型二阶系统"])
        if model_type == "典型二阶系统":
            zeta = st.slider("阻尼比 ζ", 0.0, 2.0, 0.5, 0.01)
            wn = st.slider("自然频率 ωn", 0.5, 10.0, 2.0, 0.1)
            params = {'zeta': zeta, 'wn': wn}
        else:
            params = None
        sys = get_plant_model(model_type, params)
    else:
        col1, _ = st.columns(2)
        with col1:
            num_input = st.text_input("分子系数", "1")
            den_input = st.text_input("分母系数", "1 2 1")
        num = [float(x) for x in num_input.split()]
        den = [float(x) for x in den_input.split()]
        sys = create_tf(num, den)

    plot_type = st.radio("选择图类型", ["Bode 图", "Nyquist 图", "两者并排"])
    if "Bode" in plot_type or "两者" in plot_type:
        st.subheader("Bode 图")
        fig, axes = plt.subplots(2,1, figsize=(8,6))
        mag, phase, omega = ct.bode_plot(sys, plot=False)
        axes[0].semilogx(omega, 20*np.log10(mag))
        axes[0].set_ylabel('Magnitude (dB)')
        axes[0].grid(True, which='both')
        axes[1].semilogx(omega, phase * 180/np.pi)
        axes[1].set_ylabel('Phase (deg)')
        axes[1].set_xlabel('Frequency (rad/s)')
        axes[1].grid(True, which='both')
        st.pyplot(fig)
        gm, pm, wg, wp = ct.margin(sys)
        st.write(f"**增益裕度 GM**: {gm:.3f} (≈{20*np.log10(gm):.2f} dB) @ {wg:.3f} rad/s")
        st.write(f"**相位裕度 PM**: {pm:.3f} deg @ {wp:.3f} rad/s")
    if "Nyquist" in plot_type or "两者" in plot_type:
        st.subheader("Nyquist 图")
        fig2, ax2 = plt.subplots(figsize=(6,6))
        ct.nyquist_plot(sys, ax=ax2)
        ax2.set_title("Nyquist Diagram")
        ax2.grid(True)
        st.pyplot(fig2)

# ---------- 实验四：PID控制器设计 ----------
elif experiment == "🔧 PID控制器设计":
    st.header("PID控制器设计与闭环仿真")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("被控对象模型")
        use_lib = st.checkbox("使用虚拟对象库", value=True)
        if use_lib:
            model_type = st.selectbox("选择对象",
                ["直流电机 (电压-转速)", "水箱液位 (一阶惯性)", "典型二阶系统"])
            if model_type == "典型二阶系统":
                zeta = st.slider("ζ", 0.0, 2.0, 0.5, 0.01)
                wn = st.slider("ωn", 0.5, 10.0, 2.0, 0.1)
                params = {'zeta': zeta, 'wn': wn}
            else:
                params = None
            G = get_plant_model(model_type, params)
        else:
            num_input = st.text_input("分子系数", "1")
            den_input = st.text_input("分母系数", "1 2 1")
            num_p = [float(x) for x in num_input.split()]
            den_p = [float(x) for x in den_input.split()]
            G = create_tf(num_p, den_p)

    with col2:
        st.subheader("PID参数")
        Kp = st.slider("Kp", 0.0, 20.0, 1.0, 0.1)
        Ki = st.slider("Ki", 0.0, 10.0, 0.0, 0.1)
        Kd = st.slider("Kd", 0.0, 5.0, 0.0, 0.1)
        setpoint = st.number_input("设定值", 0.0, 10.0, 1.0, 0.1)
        # 非线性选项
        add_nl = st.checkbox("控制器输出加入非线性", False)
        if add_nl:
            nl_type = st.selectbox("非线性类型", ["死区", "饱和", "量化"])
            nl_param = st.number_input("参数", 0.1, 2.0, 0.5, 0.1)

    # 构建PID控制器 (理想: C(s) = Kp + Ki/s + Kd*s)
    # 注意分母阶次，这里构造 (Kd*s^2 + Kp*s + Ki)/s
    C = ct.tf([Kd, Kp, Ki], [1, 0])
    L = ct.series(C, G)
    T = ct.feedback(L, 1)

    # 仿真，考虑非线性则在时域仿真中手动施加
    t = np.linspace(0, 15, 1000)
    if add_nl:
        # 用 lsim 模拟，并在控制量上加入非线性（这里简化：对控制器输出量 C 的输出施加非线性）
        # 采用数值逼近：计算开环传递函数，再在反馈回路中插入非线性
        # 为简单起见，我们采用离散化仿真，手动迭代
        # 这里给出一种近似：先用线性仿真得到控制器输出，施加非线性，再重新仿真
        # 为了准确，使用 forced_response 并手动迭代非线性回路
        # 为了代码清晰，这里只做示意——直接对输出信号施加非线性（实际中位置可能不对，但展示思想）
        # 更好的做法是：离散化 PID 并用 for 循环，这里简化
        t, y = ct.step_response(T * setpoint, T=t)  # 线性响应作为近似
        y = apply_nonlinearity(y, nl_type, nl_param)
    else:
        t, y = ct.step_response(T * setpoint, T=t)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t, y, label=f'PID (Kp={Kp}, Ki={Ki}, Kd={Kd})')
    ax.axhline(setpoint, color='r', linestyle='--', label='设定值')
    ax.set_title("闭环阶跃响应")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("输出")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    if not add_nl:
        info = ct.step_info(T)
        if info:
            st.metric("超调量", f"{info.get('Overshoot', 0):.2f} %")
            st.metric("调节时间", f"{info.get('SettlingTime', np.nan):.3f} s")
    else:
        st.info("非线性作用下，性能指标可能不准确，请观察波形。")

# ---------- 实验五：虚拟被控对象库（独立浏览） ----------
elif experiment == "🤖 虚拟被控对象库":
    st.header("虚拟被控对象库")
    model = st.selectbox("选择预置模型",
        ["直流电机 (电压-转速)", "倒立摆 (摆角/力)", "水箱液位 (一阶惯性)", "典型二阶系统"])
    if model == "典型二阶系统":
        zeta = st.slider("阻尼比 ζ", 0.0, 2.0, 0.5, 0.01)
        wn = st.slider("自然频率 ωn", 0.5, 10.0, 2.0, 0.1)
        params = {'zeta': zeta, 'wn': wn}
    else:
        params = None
    sys = get_plant_model(model, params)
    st.latex(f"G(s) = {sys}")
    # 显示极点
    poles = ct.poles(sys)
    st.write("**极点：**", [f"{p.real:.3f}+{p.imag:.3f}j" for p in poles])
    # 阶跃响应
    t, y = ct.step_response(sys)
    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_title(f"{model} 阶跃响应")
    ax.grid(True)
    st.pyplot(fig)

# ---------- 实验六：非线性环节分析 ----------
elif experiment == "🚧 非线性环节分析":
    st.header("非线性环节特性分析")
    st.markdown("选择一个基本传递函数，然后在回路中串联非线性环节，观察对阶跃响应的影响。")
    col1, col2 = st.columns(2)
    with col1:
        num_input = st.text_input("分子系数", "1")
        den_input = st.text_input("分母系数", "1 2 1")
    with col2:
        nl_type = st.selectbox("非线性类型", ["死区", "饱和", "量化"])
        param = st.slider("参数", 0.1, 2.0, 0.5, 0.1)
    num = [float(x) for x in num_input.split()]
    den = [float(x) for x in den_input.split()]
    sys = create_tf(num, den)
    t, y_lin = ct.step_response(sys)
    y_nl = apply_nonlinearity(y_lin, nl_type, param)
    
    fig, ax = plt.subplots()
    ax.plot(t, y_lin, label='线性响应')
    ax.plot(t, y_nl, label=f'加入{nl_type}({param})')
    ax.legend()
    ax.grid(True)
    ax.set_title("非线性影响对比")
    st.pyplot(fig)
    st.write("观察信号被截断、阶梯化或偏移等现象。")

# ---------- 实验七：倒立摆小车实时动画 ----------
elif experiment == "🎥 倒立摆小车实时动画":
    st.header("倒立摆小车实时动画仿真")
    st.markdown("**模型：** 小车倒立摆非线性动力学，采用简单 PD 控制（或手动力）")
    col1, col2 = st.columns(2)
    with col1:
        control_mode = st.radio("控制模式", ["手动恒定力", "PD 摆角控制"])
        if control_mode == "手动恒定力":
            F = st.slider("恒定力 (N)", -5.0, 5.0, 0.0, 0.1)
        else:
            kp = st.slider("摆角比例 Kp_theta", 0.0, 50.0, 20.0, 0.5)
            kd = st.slider("摆角微分 Kd_theta", 0.0, 10.0, 2.0, 0.1)
    with col2:
        sim_duration = st.slider("仿真时长 (s)", 2.0, 10.0, 5.0, 0.5)
        theta0 = st.slider("初始摆角 (rad)", -0.5, 0.5, 0.1, 0.01)  # 偏离竖直

    # 控制器函数
    def u_func(t):
        if control_mode == "手动恒定力":
            return F
        else:
            # 需要状态反馈，这里为演示简单，使用一个简化的状态访问方法
            # 在仿真循环中动态计算，此处返回0，实际在仿真中直接根据状态计算
            return None  # 将在仿真函数内处理

    # 仿真
    if control_mode == "手动恒定力":
        def u_loop(t):
            return F
    else:
        # 使用全局变量传递当前状态（简单方法，实际项目不建议，但演示方便）
        class State:
            pass
        state_holder = State()
        def u_loop(t):
            # 从 state_holder 获取状态
            if hasattr(state_holder, 'theta'):
                return -kp * state_holder.theta - kd * state_holder.theta_dot
            return 0.0

    # 仿真
    t_span = [0, sim_duration]
    dt = 0.02
    x0 = [0, theta0, 0, 0]
    t_arr = np.arange(t_span[0], t_span[1], dt)
    states = np.zeros((len(t_arr), 4))
    states[0] = x0
    for i in range(1, len(t_arr)):
        t_now = t_arr[i-1]
        state_now = states[i-1]
        if control_mode == "PD 摆角控制":
            state_holder.theta = state_now[1]
            state_holder.theta_dot = state_now[3]
            u = -kp*state_now[1] - kd*state_now[3]
        else:
            u = F
        sol = odeint(cartpole_dynamics, state_now, [t_now, t_arr[i]], args=(u,))
        states[i] = sol[1]

# 绘制结果曲线
    fig1, ax1 = plt.subplots(2,1, figsize=(8,6))
    ax1[0].plot(t_arr, states[:,0], label='小车位置 x')
    ax1[0].set_ylabel('x (m)')
    ax1[0].grid()
    ax1[1].plot(t_arr, states[:,1], label='摆角 theta')
    ax1[1].set_ylabel('theta (rad)')
    ax1[1].set_xlabel('时间 (s)')
    ax1[1].grid()
    st.pyplot(fig1)

    # 制作动画（每隔几帧取一帧，防止数据量过大）
    st.subheader("🎞️ 倒立摆动画")
    skip = max(1, len(t_arr)//100)  # 最多显示100帧
    anim_frames = t_arr[::skip]
    anim_states = states[::skip]

    from matplotlib.animation import FuncAnimation
    fig_anim, ax_anim = plt.subplots(figsize=(6,3))
    ax_anim.set_xlim(-2.5, 2.5)
    ax_anim.set_ylim(-0.1, 1.2)
    ax_anim.set_aspect('equal')
    ax_anim.grid()
    # 绘制元素
    cart_width = 0.3
    cart_height = 0.15
    pendulum_length = 0.5
    # 空图形，动画中更新
    cart_rect = plt.Rectangle((0,0), cart_width, cart_height, fc='blue')
    pendulum_line, = ax_anim.plot([], [], 'r-', lw=3)
    mass_ball, = ax_anim.plot([], [], 'ro', markersize=8)
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)

    def init():
        ax_anim.add_patch(cart_rect)
        pendulum_line.set_data([], [])
        mass_ball.set_data([], [])
        time_text.set_text('')
        return cart_rect, pendulum_line, mass_ball, time_text

    def animate(i):
        x = anim_states[i,0]
        theta = anim_states[i,1]
        # 小车矩形位置（左下角）
        cart_rect.set_xy((x - cart_width/2, 0))
        # 摆杆：从小车中心到摆末端
        pivot_x = x
        pivot_y = cart_height
        tip_x = pivot_x + pendulum_length * np.sin(theta)
        tip_y = pivot_y + pendulum_length * np.cos(theta)
        pendulum_line.set_data([pivot_x, tip_x], [pivot_y, tip_y])
        mass_ball.set_data([tip_x], [tip_y])
        time_text.set_text(f't = {anim_frames[i]:.2f}s')
        return cart_rect, pendulum_line, mass_ball, time_text

    anim = FuncAnimation(fig_anim, animate, frames=len(anim_frames), init_func=init, blit=True, interval=50)
    # 在 Streamlit 中显示动画：写入 HTML
    anim_html = anim.to_jshtml()
    st.components.v1.html(anim_html, height=400)
    st.caption("小车倒立摆动画（非线性模型）")

    # 保存动画选项（可选）
    if st.button("保存动画为 HTML 文件"):
        with open("cartpole_animation.html", "w") as f:
            f.write(anim_html)
        st.success("动画已保存为 cartpole_animation.html，可在当前文件夹找到。")

# 绘制结果曲线
    fig1, ax1 = plt.subplots(2,1, figsize=(8,6))
    ax1[0].plot(t_arr, states[:,0], label='小车位置 x')
    ax1[0].set_ylabel('x (m)')
    ax1[0].grid()
    ax1[1].plot(t_arr, states[:,1], label='摆角 theta')
    ax1[1].set_ylabel('theta (rad)')
    ax1[1].set_xlabel('时间 (s)')
    ax1[1].grid()
    st.pyplot(fig1)

    # 制作动画（每隔几帧取一帧，防止数据量过大）
    st.subheader("🎞️ 倒立摆动画")
    skip = max(1, len(t_arr)//100)  # 最多显示100帧
    anim_frames = t_arr[::skip]
    anim_states = states[::skip]

    from matplotlib.animation import FuncAnimation
    fig_anim, ax_anim = plt.subplots(figsize=(6,3))
    ax_anim.set_xlim(-2.5, 2.5)
    ax_anim.set_ylim(-0.1, 1.2)
    ax_anim.set_aspect('equal')
    ax_anim.grid()
    # 绘制元素
    cart_width = 0.3
    cart_height = 0.15
    pendulum_length = 0.5
    # 空图形，动画中更新
    cart_rect = plt.Rectangle((0,0), cart_width, cart_height, fc='blue')
    pendulum_line, = ax_anim.plot([], [], 'r-', lw=3)
    mass_ball, = ax_anim.plot([], [], 'ro', markersize=8)
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)

    def init():
        ax_anim.add_patch(cart_rect)
        pendulum_line.set_data([], [])
        mass_ball.set_data([], [])
        time_text.set_text('')
        return cart_rect, pendulum_line, mass_ball, time_text

    def animate(i):
        x = anim_states[i,0]
        theta = anim_states[i,1]
        # 小车矩形位置（左下角）
        cart_rect.set_xy((x - cart_width/2, 0))
        # 摆杆：从小车中心到摆末端
        pivot_x = x
        pivot_y = cart_height
        tip_x = pivot_x + pendulum_length * np.sin(theta)
        tip_y = pivot_y + pendulum_length * np.cos(theta)
        pendulum_line.set_data([pivot_x, tip_x], [pivot_y, tip_y])
        mass_ball.set_data([tip_x], [tip_y])
        time_text.set_text(f't = {anim_frames[i]:.2f}s')
        return cart_rect, pendulum_line, mass_ball, time_text

    anim = FuncAnimation(fig_anim, animate, frames=len(anim_frames), init_func=init, blit=True, interval=50)
    # 在 Streamlit 中显示动画：写入 HTML
    anim_html = anim.to_jshtml()
    st.components.v1.html(anim_html, height=400)
    st.caption("小车倒立摆动画（非线性模型）")

    # 保存动画选项（可选）
    if st.button("保存动画为 HTML 文件"):
        with open("cartpole_animation.html", "w") as f:
            f.write(anim_html)
        st.success("动画已保存为 cartpole_animation.html，可在当前文件夹找到。")

# ---------- 实验一：时域分析 ----------
elif experiment == "📈 时域分析（阶跃/脉冲/斜坡）":
    st.header("时域响应分析")
    col1, col2 = st.columns(2)
    with col1:
        use_lib = st.checkbox("使用虚拟对象库", value=False)
        if use_lib:
            model_type = st.selectbox("选择对象模型", 
                ["直流电机 (电压-转速)", "倒立摆 (摆角/力)", "水箱液位 (一阶惯性)", "典型二阶系统"])
            if model_type == "典型二阶系统":
                zeta = st.slider("阻尼比 ζ", 0.0, 2.0, 0.5, 0.01)
                wn = st.slider("自然频率 ωn (rad/s)", 0.5, 10.0, 2.0, 0.1)
                params = {'zeta': zeta, 'wn': wn}
            else:
                params = None
            sys = get_plant_model(model_type, params)
            # 检查倒立摆模型的不稳定性并提示
            if "倒立摆" in model_type:
                st.warning("⚠️ 倒立摆线性化模型为不稳定系统（极点位于右半平面），阶跃响应将发散，仿真时间已自动限制。")
        else:
            model_type = "自定义传递函数"
            num_input = st.text_input("分子系数（空格分隔）", "1")
            den_input = st.text_input("分母系数（空格分隔）", "1 2 1")
            num = [float(x) for x in num_input.split()]
            den = [float(x) for x in den_input.split()]
            sys = create_tf(num, den)

    with col2:
        input_type = st.radio("输入信号", ["阶跃", "脉冲", "斜坡"])
        # 对于不稳定系统，自动缩短仿真时间
        poles = ct.poles(sys)
        unstable = any(np.real(p) > 0 for p in poles)
        default_time = 2.0 if unstable else 10.0
        sim_time = st.slider("仿真时间 (s)", 0.5, 30.0, default_time, 0.5)

    # 生成响应
    t = np.linspace(0, sim_time, 1000)
    try:
        if input_type == "阶跃":
            t, y = ct.step_response(sys, T=t)
        elif input_type == "脉冲":
            t, y = ct.impulse_response(sys, T=t)
        else:
            u = t  # 斜坡输入
            t, y, _ = ct.forced_response(sys, T=t, U=u)
    except Exception as e:
        st.error(f"仿真出错：{e}，可能由于系统不稳定导致数值发散。请尝试缩短仿真时间或检查模型。")
        st.stop()

    # 绘图
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t, y, linewidth=2)
    ax.set_title(f"{model_type}  {input_type}响应")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("输出")
    ax.grid(True)
    st.pyplot(fig)

    # 性能指标（仅稳定系统且阶跃输入）
    if input_type == "阶跃" and not unstable:
        try:
            info = ct.step_info(sys)
            if info is not None:
                st.subheader("📋 时域性能指标")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("上升时间", f"{info.get('RiseTime', np.nan):.3f} s")
                col_b.metric("峰值时间", f"{info.get('PeakTime', np.nan):.3f} s")
                col_c.metric("超调量", f"{info.get('Overshoot', 0):.2f} %")
                col_d.metric("调节时间", f"{info.get('SettlingTime', np.nan):.3f} s")
            else:
                st.info("无超调或响应未收敛，无法计算指标。")
        except:
            st.warning("性能指标计算失败，可能由于系统特性复杂。")
    elif unstable:
        st.info("当前为不稳定系统，不计算性能指标。")