import tkinter as tk
import webbrowser
from tkinter import PhotoImage
from tkinter import messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk, ImageSequence
from simulation import *

def about():
    # 创建自定义的关于窗口
    about_window = tk.Toplevel(root)
    about_window.title("关于")
    about_window.geometry("400x300")
    about_window.resizable(False, False)  # 禁止调整窗口大小

    # 创建标签显示关于信息
    about_label = tk.Label(about_window, text="这是一个插齿机运动仿真程序\n\n指导老师：杨谢柳\n作者：孟睿豪", padx=10, pady=10)
    about_label.pack()

    # 创建可点击的链接
    link = tk.Label(about_window, text="作者个人主页", fg="blue", cursor="hand2")
    link.pack()
    link.bind("<Button-1>", lambda e: webbrowser.open_new("https://mengruihao.github.io/"))

    # 正确处理 GIF 图像序列
    gif_image = Image.open("彩虹小蓝猫.gif")
    gif_frames = [ImageTk.PhotoImage(image=img) for img in ImageSequence.Iterator(gif_image)]

    gif_label = tk.Label(about_window)
    gif_label.pack()
    gif_index = 0

    def update_gif():
        nonlocal gif_index
        frame = gif_frames[gif_index]
        gif_label.configure(image=frame)
        gif_index = (gif_index + 1) % len(gif_frames)
        about_window.after(100, update_gif)

    update_gif()


def run_simulation():
    l1 = float(entry_l1.get())
    l2 = float(entry_l2.get())
    l3_left = float(entry_l3_left.get())
    l3_right = float(entry_l3_right.get())
    l4 = float(entry_l4.get())
    l5 = float(entry_l5.get())
    theta = float(entry_theta.get())
    v = float(entry_v.get())
    s1 = float(entry_s1.get())
    e = float(entry_e.get())

    ani, va_ani, fig, va_fig = simulation(l1, l2, l3_left, l3_right, l4, l5, e, s1, theta, v)

    # 创建一个顶层窗口来展示结果
    result_window = tk.Toplevel(root)
    result_window.title("仿真与曲线图")
    result_window.geometry("1700x600")  # 设置窗口的固定尺寸
    result_window.resizable(False, False)  # 禁止调整窗口大小

    # 创建一个Frame用于仿真结果的显示
    simulation_frame = tk.Frame(result_window)
    simulation_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 将仿真结果图形加入到Frame中
    canvas1 = FigureCanvasTkAgg(fig, master=simulation_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 创建一个Frame用于曲线图的显示
    curve_frame = tk.Frame(result_window)
    curve_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # 将曲线图加入到Frame中
    canvas2 = FigureCanvasTkAgg(va_fig, master=curve_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# def ui_start():
# 创建主窗口
root = tk.Tk()
root.title("插齿机运动仿真")
root.geometry("720x300")
root.resizable(False, False)  # 禁止调整窗口大小

# 创建菜单栏
menu_bar = tk.Menu(root)

# 创建“文件”菜单
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="退出", command=root.quit)
menu_bar.add_cascade(label="文件", menu=file_menu)

# 创建“帮助”菜单
help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="关于", command=about)
menu_bar.add_cascade(label="帮助", menu=help_menu)

# 将菜单栏添加到主窗口
root.config(menu=menu_bar)

# 创建框架一来放置l1和l2的标签与输入框
frame_1 = tk.Frame(root)
frame_1.pack(anchor="w", pady=5)

# 创建并放置标签和输入框在同一行
label_l1 = tk.Label(frame_1, text="杆l1长度", width=10)
label_l1.pack(side="left")
entry_l1 = tk.Entry(frame_1, width=10)
entry_l1.pack(side="left")

label_l2 = tk.Label(frame_1, text="杆l2长度", width=10)
label_l2.pack(side="left", padx=(20, 0))  # 添加一些水平间距
entry_l2 = tk.Entry(frame_1, width=10)
entry_l2.pack(side="left")

# 创建框架二来放置l3的标签与输入框
frame_2 = tk.Frame(root)
frame_2.pack(anchor="w", pady=5)

# 创建并放置标签和输入框在同一行
label_l3_left = tk.Label(frame_2, text="杆l3左侧长度", width=10)
label_l3_left.pack(side="left")
entry_l3_left = tk.Entry(frame_2, width=10)
entry_l3_left.pack(side="left")

label_l3_right = tk.Label(frame_2, text="杆l3右侧长度", width=10)
label_l3_right.pack(side="left", padx=(20, 0))  # 添加一些水平间距
entry_l3_right = tk.Entry(frame_2, width=10)
entry_l3_right.pack(side="left")

# 创建框架三来放置l4和l5的标签与输入框
frame_3 = tk.Frame(root)
frame_3.pack(anchor="w", pady=5)

# 创建并放置标签和输入框在同一行
label_l4 = tk.Label(frame_3, text="杆l4长度", width=10)
label_l4.pack(side="left")
entry_l4 = tk.Entry(frame_3, width=10)
entry_l4.pack(side="left")

label_l5 = tk.Label(frame_3, text="杆l5长度", width=10)
label_l5.pack(side="left", padx=(20, 0))  # 添加一些水平间距
entry_l5 = tk.Entry(frame_3, width=10)
entry_l5.pack(side="left")

# 创建框架四来放置l4和l5的标签与输入框
frame_4 = tk.Frame(root)
frame_4.pack(anchor="w", pady=5)

# 创建并放置标签和输入框在同一行
label_s1 = tk.Label(frame_4, text="距离S1", width=10)
label_s1.pack(side="left")
entry_s1 = tk.Entry(frame_4, width=10)
entry_s1.pack(side="left")

label_e = tk.Label(frame_4, text="距离e", width=10)
label_e.pack(side="left", padx=(20, 0))  # 添加一些水平间距
entry_e = tk.Entry(frame_4, width=10)
entry_e.pack(side="left")

# 创建框架五来放置l4和l5的标签与输入框
frame_5 = tk.Frame(root)
frame_5.pack(anchor="w", pady=5)

# 创建并放置标签和输入框在同一行
label_theta = tk.Label(frame_5, text="角度θ", width=10)
label_theta.pack(side="left")
entry_theta = tk.Entry(frame_5, width=10)
entry_theta.pack(side="left")

label_v = tk.Label(frame_5, text="角速度w", width=10)
label_v.pack(side="left", padx=(20, 0))  # 添加一些水平间距
entry_v = tk.Entry(frame_5, width=10)
entry_v.pack(side="left")

# 创建并放置按钮
button_login = tk.Button(root, text="运行仿真", command=run_simulation)
button_login.pack(pady=20)


# 等待一段时间再进入 root.mainloop()
root.after(1000000, lambda: None)  # 等待2秒

# 运行主循环
root.mainloop()
print("结束哩！")