import math
import threading
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process, Manager, Value
from tqdm import tqdm


# 计算phi3角度的函数
def calculate_phi(l1, l2, l3, s1, phi1_deg, theta1_deg):
    phi1 = np.deg2rad(phi1_deg)  # 将角度转换为弧度
    theta1 = np.deg2rad(theta1_deg)  # 将角度转换为弧度

    phi2, phi3 = sp.symbols('phi2 phi3')  # 定义符号变量

    # 建立方程
    eq1 = l1 * sp.cos(phi1) + l2 * sp.cos(phi2) - l3 * sp.cos(phi3) - s1 * sp.cos(theta1)
    eq2 = l1 * sp.sin(phi1) + l2 * sp.sin(phi2) - l3 * sp.sin(phi3) - s1 * sp.sin(theta1)

    # 解方程
    solutions = sp.solve([eq1, eq2], (phi2, phi3))
    phi2_solutions = [sol[0] for sol in solutions]
    phi3_solutions = [sol[1] for sol in solutions]

    # 计算角度值
    phi2_solutions_deg = [(float(sol.evalf()) * 180 / np.pi) % 360 for sol in phi2_solutions]
    phi3_solutions_deg = [(float(sol.evalf()) * 180 / np.pi) % 360 for sol in phi3_solutions]

    # 筛选合适的角度范围
    filtered_phi3_solutions = [sol for sol in phi3_solutions_deg if 103 <= sol <= 208]

    return phi2_solutions_deg, filtered_phi3_solutions


# 摇杆滑块机构
def calculate_phi4_and_s2(l3r, l4, phi3_degrees, e):
    # Convert degrees to radians
    phi3_radians = math.radians(phi3_degrees)

    # Calculate cos(phi4) and the possible values for sin(phi4) using the trigonometric identity
    cos_phi4 = (l3r * math.cos(phi3_radians) - e) / l4
    sin_phi4_squared = 1 - cos_phi4 ** 2
    results = []

    if sin_phi4_squared >= 0:
        sin_phi4_positive = math.sqrt(sin_phi4_squared)
        # Calculate phi4 in radians for both positive and negative sin(phi4)
        phi4_positive = math.atan2(sin_phi4_positive, cos_phi4)
        # Calculate s2 for both solutions
        s2_positive = l4 * sin_phi4_positive - l3r * math.sin(phi3_radians)
        # Append results in degrees for clarity
        results.append(("Positive Solution", math.degrees(phi4_positive), s2_positive))
    else:
        results.append(("No Real Solution", "N/A", "N/A"))

    return results


# 运动学计算
def compute_kinematics(data_matrix, set_speed):
    # 计算角速度
    for i in range(1, data_matrix.shape[0]):  # 从第二个时间点开始计算，因为第一个没有前一个状态
        for j in range(0, data_matrix.shape[2]):  # 从第二列开始计算，第一列存放原始角度
            # 当前状态和前一个状态的角度差除以时间步长得到角速度
            data_matrix[i, 1, j] = (data_matrix[i, 0, j] - data_matrix[i - 1, 0, j]) / set_speed

    # 计算角加速度
    for i in range(1, data_matrix.shape[0]):  # 从第二个时间点开始计算，因为第一个没有前一个状态
        for j in range(0, data_matrix.shape[2]):  # 从第二列开始计算，第一列存放原始角度
            # 当前状态和前一个状态的角速度差除以时间步长得到角加速度
            data_matrix[i, 2, j] = (data_matrix[i, 1, j] - data_matrix[i - 1, 1, j]) / set_speed


# 多进程处理范围计算
def process_range(l1, l2, l3, s1, theta1_deg, start_deg, end_deg, output_list, progress_counter):
    for phi1_deg in range(start_deg, end_deg + 1):
        phi2_solutions, phi3_solutions = calculate_phi(l1, l2, l3, s1, phi1_deg, theta1_deg)
        output_list.append((phi1_deg, phi2_solutions, phi3_solutions))
        with progress_counter.get_lock():
            progress_counter.value += 1


# 计算信息矩阵 InfoMat
def InfoMat_calculation(data_matrix, l1, l3_left, l3_right, l4, l5, s1, theta1_deg):
    """
    计算机械连杆系统中各点的位置，并将每个phi1角度的结果存入一个三维矩阵,我们称之为信息矩阵InfoMat。

    参数:
    - phi1_vals, phi3_vals: 角度数组（度），其中phi1_vals[frame]是一个数组。
    - data_matrix: 包含额外角度数据的矩阵。
    - frame: 当前帧的索引。
    - l1, l3_left, l3_right, l4, l5, s1: 各杆的长度。
    - theta1_deg: 初始位置的固定角度。

    返回:
    一个三维矩阵，每个二维子矩阵包含对应phi1角度计算得到的所有点坐标。

    矩阵：
    - (phi1) * (x, y) * (A, B, C, D, E, F, G)

    """

    points_matrixs = []  # 初始化三维矩阵来存储所有点的坐标
    frames = list(range(data_matrix.shape[0]))

    # 遍历phi1_vals中的每个phi1角度
    for frame in frames:
        # TotalAngleData = data_matrix[frame, 0, :]
        phi1 = np.deg2rad(data_matrix[frame, 0, 0])
        phi3 = np.deg2rad(data_matrix[frame, 0, 2])
        # phi4 = np.deg2rad(data_matrix[frame, 0, 4])
        phi4_deg = data_matrix[frame, 0, 4]
        phi4 = np.deg2rad(180 - phi4_deg)

        # 计算各点坐标
        xb, yb = l1 * np.cos(phi1), l1 * np.sin(phi1)
        xd, yd = s1 * np.cos(np.deg2rad(theta1_deg)), s1 * np.sin(np.deg2rad(theta1_deg))
        xc, yc = xd + l3_left * np.cos(phi3), yd + l3_left * np.sin(phi3)

        direction_x = xd - xc
        direction_y = yd - yc
        length_CD = np.sqrt(direction_x ** 2 + direction_y ** 2)
        unit_x = direction_x / length_CD
        unit_y = direction_y / length_CD

        xe, ye = xd + l3_right * unit_x, yd + l3_right * unit_y

        xf = xe + l4 * np.cos(phi4)
        yf = ye - l4 * np.sin(phi4)

        xg = xf
        yg = yf - l5

        # 构建当前phi1角度的points_matrix
        points_matrix = np.array([
            [0, 0],
            [xb, yb],
            [xc, yc],
            [xd, yd],
            [xe, ye],
            [xf, yf],
            [xg, yg]
        ])

        # 将当前points_matrix添加到三维数组中,再变化为矩阵
        points_matrixs.append(points_matrix)
        InfoMat = np.array(points_matrixs)

    # 将列表转换为信息矩阵
    return InfoMat


# 速度和加速度曲线
def plot_velocity_acceleration(InfoMat, dt=1.0, point_index=6):
    """
    绘制指定点的速度和加速度曲线。

    参数:
    - InfoMat: 三维数组，包含多帧中各点的坐标。
    - dt: 每帧之间的时间间隔，默认为1.0秒。
    - point_index: 指定点的索引，默认为6，对应G点。
    """
    # 提取指定点的坐标
    point_coordinates = InfoMat[:, point_index, :]

    # 计算速度
    # 通过在数组末尾附加第一个点的方式“闭环”
    wrapped_point_coordinates = np.vstack([point_coordinates, point_coordinates[0]])
    velocities = (wrapped_point_coordinates[0:-1] - wrapped_point_coordinates[1:]) / dt

    # 计算加速度
    # 同样通过在速度数组末尾附加第一个速度的方式“闭环”
    wrapped_velocities = np.vstack([velocities, velocities[0]])
    accelerations = (wrapped_velocities[0:-1] - wrapped_velocities[1:]) / dt

    return velocities, accelerations


def frictionless_moment(InfoMat, data_matrix, m, l, dt):

    l1 = 300
    l2 = 650
    l3_left, l3_right = 450, 670
    l4 = 300
    l5 = 800
    # phi3_degrees = -30  # phi3 in degrees
    e = 430
    s1 = 550  # 支座D相据于支座A的距离
    theta1_deg = 67  # 支座D相对于支座A的角度
    process_num = 10  # 线程数
    radian_value = 1.72  # 可以调整速度
    m = [40, 150, 40, 80, 80]
    l = [l1, l2, l3_left+l3_right, l4, l5]

    l = np.array(l)
    m = np.array(m)
    # 数据矩阵处理后(三维): (phi1) * (角度, 角速度, 角加速度) * (phi1, phi2, phi3, phi4)
    data = np.delete(data_matrix, 3, axis=2)

    s1_mat = (InfoMat[:, 0, :] + InfoMat[:, 1, :]) / 2
    s2_mat = (InfoMat[:, 1, :] + InfoMat[:, 2, :]) / 2
    s3_mat = (InfoMat[:, 3, :] + InfoMat[:, 4, :]) / 2
    s4_mat = (InfoMat[:, 4, :] + InfoMat[:, 5, :]) / 2
    s5_mat = (InfoMat[:, 5, :] + InfoMat[:, 6, :]) / 2

    velocities_s1 = (s1_mat[0:-1] - s1_mat[1:]) / dt  # 计算s1速度
    velocities_s2 = (s2_mat[0:-1] - s2_mat[1:]) / dt  # 计算s2速度
    velocities_s3 = (s3_mat[0:-1] - s3_mat[1:]) / dt  # 计算s3速度
    velocities_s4 = (s4_mat[0:-1] - s4_mat[1:]) / dt  # 计算s4速度
    velocities_s5 = (s5_mat[0:-1] - s5_mat[1:]) / dt  # 计算s5速度

    # 速度数据转换,计算加速度
    wrapped_velocities_s1 = np.vstack([velocities_s1, velocities_s1[0]])
    wrapped_velocities_s2 = np.vstack([velocities_s2, velocities_s2[0]])
    wrapped_velocities_s3 = np.vstack([velocities_s3, velocities_s3[0]])
    wrapped_velocities_s4 = np.vstack([velocities_s4, velocities_s4[0]])
    wrapped_velocities_s5 = np.vstack([velocities_s5, velocities_s5[0]])

    accelerations_s1 = (wrapped_velocities_s1[0:-1] - wrapped_velocities_s1[1:]) / dt  # 计算s1加速度
    accelerations_s2 = (wrapped_velocities_s2[0:-1] - wrapped_velocities_s2[1:]) / dt  # 计算s2加速度
    accelerations_s3 = (wrapped_velocities_s3[0:-1] - wrapped_velocities_s3[1:]) / dt  # 计算s3加速度
    accelerations_s4 = (wrapped_velocities_s4[0:-1] - wrapped_velocities_s4[1:]) / dt  # 计算s4加速度
    accelerations_s5 = (wrapped_velocities_s5[0:-1] - wrapped_velocities_s5[1:]) / dt  # 计算s5加速度

    sv_matrix = np.array([velocities_s1, velocities_s2, velocities_s3, velocities_s4, velocities_s5])
    sa_matrix = np.array([accelerations_s1, accelerations_s2,  accelerations_s3, accelerations_s4, accelerations_s5])

    # 使用 transpose 调整维度顺序
    sv_matrix = np.transpose(sv_matrix, (1, 0, 2))
    sa_matrix = np.transpose(sa_matrix, (1, 0, 2))

    sv_x = np.array(sv_matrix[:, :, 0])
    sa_x = np.array(sa_matrix[:, :, 1])
    sv_y = np.array(sv_matrix[:, :, 0])
    sa_y = np.array(sa_matrix[:, :, 1])
    w = np.array(data[:, 1, :])

    alpha = np.array(data[:, 2, :])
    w1 = np.array(data[1:, 1, 0])
    w1 = np.tile(w1, (5, 1)).T

    # 删除最后一行，创建一个新的360x4矩阵
    reduced_matrix = w[:-1]
    # 创建一个新的360x5的矩阵，所有元素初始化为0
    final_w = np.zeros((360, 5))
    # 将缩减后的矩阵内容复制到新矩阵的前4列
    final_w[:, :4] = reduced_matrix

    # 删除最后一行，创建一个新的360x4矩阵
    reduced_matrix = alpha[:-1]
    # 创建一个新的360x5的矩阵，所有元素初始化为0
    final_alpha = np.zeros((360, 5))
    # 将缩减后的矩阵内容复制到新矩阵的前4列
    final_alpha[:, :4] = reduced_matrix

    # print(sv_x.shape)
    # print(sv_y.shape)
    # print(sa_x.shape)
    # print(sa_y.shape)
    # print(data.shape)
    # print(final_w.shape)
    # print(final_alpha.shape)
    # print(w1.shape)

    moment_x = ((- m * sv_x * sa_x) - (1/3 * m * l ** 2 * final_w * final_alpha)) / w1
    moment_y = ((- m * sv_y * sa_y) - (1 / 3 * m * l ** 2 * final_w * final_alpha)) / w1
    print("不考虑摩擦的力矩为：")

    # 按行求和
    row_sums_x = np.sum(moment_x, axis=1)
    row_sums_y = np.sum(moment_y, axis=1)
    moment_modulus = np.sqrt(row_sums_x ** 2 + row_sums_y ** 2)
    moment_modulus = moment_modulus * (0.001**2)
    # filter_moment = [num for num in moment_modulus if num <= 1]
    filter_moment = moment_modulus[moment_modulus <= 1]
    print(filter_moment)
    # filter_moment = moment_modulus[moment_modulus <= 80000]
    # print(row_sums_x)
    # print(row_sums_y)
    # print(moment_modulus)
    # print(filter_moment)
    # print(filter_moment)

    return filter_moment


# 定义一个函数，用于在线程中运行 frictionless_moment
def run_frictionless_moment(InfoMat, data_matrix, m, l, dt=1):
    frictionless_moments = frictionless_moment(InfoMat, data_matrix, m, l, dt=dt)
    # 绘制曲线图
    plt.plot(frictionless_moments)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Filtered Values Plot')
    plt.show()


# 控制动画速度
def set_speeds(degree_value):
    return 1000 / degree_value


def simulation(l1, l2, l3_left, l3_right, l4, l5, e, s1, theta1_deg, radian_value):
    # 初始化参数
    l1 = 3
    l2 = 6.5
    l3_left, l3_right = 4.5, 6.7
    l4 = 3
    l5 = 8
    # phi3_degrees = -30  # phi3 in degrees
    e = 4.3
    s1 = 5.5  # 支座D相据于支座A的距离
    theta1_deg = 67  # 支座D相对于支座A的角度
    process_num = 10  # 线程数
    radian_value = 1.72  # 可以调整速度
    m = [40, 150, 40, 80, 80]
    l = [l1, l2, l3_left+l3_right, l4, l5]
    degree_value = np.degrees(radian_value)  # 化为角度
    set_speed = set_speeds(degree_value)  # 转化为每个矩阵的间隔时间
    # set_speed = 1  # 矩阵输出验证
    print("角速度:", degree_value)

    manager = Manager()
    result_list = manager.list()
    progress_counter = Value('i', 0)

    angle_ranges = [(i * 360 / process_num, (i + 1) * 360 / process_num - 1) for i in range(process_num)]

    processes = []
    total_tasks = 361

    pbar = tqdm(total=total_tasks)  # 初始化进度条

    # 创建多进程
    for angle_range in angle_ranges:
        p = Process(target=process_range, args=(
            l1, l2, l3_left, s1, theta1_deg, int(angle_range[0]), int(angle_range[1]), result_list, progress_counter))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    while any(p.is_alive() for p in processes):
        pbar.n = progress_counter.value
        pbar.refresh()

    for p in processes:
        p.join()

    pbar.close()  # 关闭进度条

    result_list = sorted(result_list, key=lambda x: x[0])

    phi1_vals = []
    phi3_vals = []

    # 数据解包并填充角度值
    for phi1_deg, phi2_solutions, phi3_solutions in result_list:
        for phi3 in phi3_solutions:
            phi1_vals.append(phi1_deg)
            phi3_vals.append(phi3)

    # 创建三维数据矩阵
    phi1_range = 361  # 0到360度

    # 数据矩阵(三维): (phi1) * (角度, 角速度, 角加速度) * (phi1, phi2, phi3, phi3-180, phi4)
    global data_matrix
    data_matrix = np.zeros((phi1_range, 3, 5))

    # 将结果填充到数据矩阵中
    for phi1_deg, phi2_solutions, phi3_solutions in sorted(result_list):
        for i, (phi2_deg, phi3_deg) in enumerate(zip(phi2_solutions, phi3_solutions)):
            if i < 3:
                data_matrix[phi1_deg, i, 0] = phi1_deg  # 存储phi1角度值
                data_matrix[phi1_deg, i, 1] = phi2_deg  # 存储phi2角度值
                data_matrix[phi1_deg, i, 2] = phi3_deg  # 存储phi3角度值
                data_matrix[phi1_deg, i, 3] = phi3_deg - 180  # 计算固定杆旋转角度并存储

    results_list = []  # 初始化结果列表
    # 遍历 data_matrix，处理每个 phi1_deg 的所有解
    for i in range(data_matrix.shape[0]):
        phi3_degrees = data_matrix[i, 0, 3]  # 提取每个二维矩阵第一行最后一列的数
        solutions = calculate_phi4_and_s2(l3_right, l4, phi3_degrees, e)
        for solution in solutions:
            description, phi4_degrees, s2 = solution  # 解包正解的角度值和对应的s2值
            results_list.append({'phi1_deg': i, 'phi4_deg': phi4_degrees, 's2': s2})
            data_matrix[i, 0, 4] = phi4_degrees  # 将 phi4 填入 data_matrix 的最后一列

    # 对机构各点进行运动学求解
    kinematics_thread = threading.Thread(target=compute_kinematics,
                                         args=(data_matrix, set_speed))  # 创建线程对象，传入 compute_kinematics 函数和所需的参数
    kinematics_thread.start()  # 启动线程

    # 求出信息矩阵
    global InfoMat  # 将InfoMat定义为全局变量，以便在回调函数中使用
    InfoMat = InfoMat_calculation(data_matrix, l1, l3_left, l3_right, l4, l5, s1, theta1_deg)

    thread = threading.Thread(target=run_frictionless_moment, args=(InfoMat, data_matrix, m, l), kwargs={'dt': 1})
    thread.start()


    # 绘制速度加速度曲线
    velocities, accelerations = plot_velocity_acceleration(InfoMat, dt=1.0, point_index=6)

    # 设置绘图和动画
    fig, ax = plt.subplots()
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)

    link1, = ax.plot([], [], 'k-')
    link2, = ax.plot([], [], 'k-')
    link3, = ax.plot([], [], 'k-')
    link4, = ax.plot([], [], 'k--')
    link5, = ax.plot([], [], 'k-')  # 使用绿色线表示CE线段
    link6, = ax.plot([], [], 'k-')
    link7, = ax.plot([], [], 'r-')

    # 创建文本标签元素
    textA = ax.text(0, 0, '', fontsize=12, color='black')
    textB = ax.text(0, 0, '', fontsize=12, color='black')
    textC = ax.text(0, 0, '', fontsize=12, color='black')
    textD = ax.text(0, 0, '', fontsize=12, color='black')
    textE = ax.text(0, 0, '', fontsize=12, color='black')
    textF = ax.text(0, 0, '', fontsize=12, color='black')
    textG = ax.text(0, 0, '', fontsize=12, color='red')

    joint_radius = 0.3  # 铰链
    joints = [plt.Circle((0, 0), joint_radius, fill=False, color='k') for _ in range(6)]
    for joint in joints:
        ax.add_patch(joint)

    # 初始化动画
    def init():
        link1.set_data([], [])
        link2.set_data([], [])
        link3.set_data([], [])
        link4.set_data([], [])
        for joint in joints:
            joint.set_center((0, 0))
        # 初始化标签位置，但不显示任何文本
        textA.set_text('')
        textB.set_text('')
        textC.set_text('')
        textD.set_text('')
        return link1, link2, link3, link4, *joints, textA, textB, textC, textD

    # 更新动画
    def update(frame):
        # 在函数内部可以直接使用 update.InfoMat 来引用 InfoMat 数据
        # InfoMat = update.InfoMat
        # 分别提取各点坐标
        xa, ya = InfoMat[frame, 0, :]
        xb, yb = InfoMat[frame, 1, :]
        xc, yc = InfoMat[frame, 2, :]
        xd, yd = InfoMat[frame, 3, :]
        xe, ye = InfoMat[frame, 4, :]
        xf, yf = InfoMat[frame, 5, :]
        xg, yg = InfoMat[frame, 6, :]

        link1.set_data([xa, xb], [ya, yb])
        link2.set_data([xb, xc], [yb, yc])
        link3.set_data([xc, xd], [yc, yd])
        link4.set_data([xd, xa], [yd, ya])
        link5.set_data([xd, xe], [yd, ye])  # 绘制DE线段
        link6.set_data([xe, xf], [ye, yf])  # 绘制EF线段
        link7.set_data([xf, xg], [yf, yg])

        joints[0].set_center((xa, ya))
        joints[1].set_center((xb, yb))
        joints[2].set_center((xc, yc))
        joints[3].set_center((xd, yd))
        joints[4].set_center((xe, ye))
        joints[5].set_center((xf, yf))

        # 更新文本标签的位置和文本
        textA.set_text('A')
        textA.set_position((xa + 0.7, ya - 0.7))
        textB.set_text('B')
        textB.set_position((xb - 1.5, yb - 1.5))
        textC.set_text('C')
        textC.set_position((xc - 1.5, yc + 0.5))
        textD.set_text('D')
        textD.set_position((xd - 0.4, yd + 0.7))
        textE.set_text('E')
        textE.set_position((xe + 0.5, ye + 0.5))
        textF.set_text('F')
        textF.set_position((xf + 0.5, yf))
        textG.set_text('G')
        textG.set_position((xg + 0.5, yg - 0.4))

        return link1, link2, link3, link4, link5, link6, link7, *joints, textA, textB, textC, textD, textE, textF, textG

    ani = animation.FuncAnimation(fig, update, frames=len(phi1_vals), init_func=init,
                                  blit=True, repeat=True, interval=set_speed)

    va_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 初始化两条线，一条用于速度，一条用于加速度
    line1, = ax1.plot([], [], 'b-', label='Velocity y')
    line2, = ax2.plot([], [], 'g-', label='Acceleration y')

    # 设置图表标题和轴标签
    ax1.set_title('Velocity of Point')
    # ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Velocity (units per second)')
    ax1.legend()
    ax1.set_xlim(0, len(velocities) - 4)  # 设置x轴范围
    ax1.set_ylim(-0.25, 0.25)  # 设置y轴范围

    ax2.set_title('Acceleration of Point')
    ax2.set_xlabel('Time (frames)')
    ax2.set_ylabel('Acceleration (units per second squared)')
    ax2.legend()
    ax2.set_xlim(0, len(accelerations) - 4)  # 设置x轴范围
    ax2.set_ylim(-0.02, 0.02)  # 设置y轴范围

    def va_init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def va_animate(frame):
        x_data = np.arange(0, frame + 1)
        y_data1 = velocities[:frame + 1, 1]
        y_data2 = accelerations[:frame + 1, 1]
        line1.set_data(x_data, y_data1)
        line2.set_data(x_data, y_data2)
        return line1, line2

    va_ani = animation.FuncAnimation(va_fig, va_animate, frames=len(velocities), init_func=va_init, blit=True,
                                     repeat=True, interval=set_speed)

    return ani, va_ani, fig, va_fig


    # plt.tight_layout()  # 调整子图间距
    # plt.grid()
    # plt.show()


if __name__ == '__main__':
    simulation(3, 6.5, 4.5, 6.7, 3, 8, 4.3, 5.5, 67, 1.72)

