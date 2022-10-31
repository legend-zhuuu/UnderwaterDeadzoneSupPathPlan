import numpy as np
from model import Point
import matplotlib.pyplot as plt
import scipy.special


def mod2pi(theta):
    """对2pi取模运算
    """
    return theta - 2.0 * np.pi * np.floor(theta / 2.0 / np.pi)


def compute_dist(point1, point2):
    return np.linalg.norm([(point1 - point2).x, (point1 - point2).y])


def is_obtuse(A, B, C):
    v1 = np.array([B.x - A.x, B.y - A.y])
    v2 = np.array([C.x - A.x, C.y - A.y])
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if cos < 1e-3:
        return True
    return False


def get_bezier(start_point, start_angle, end_point, end_angle):
    ctl_length = 0.003
    start_pos = start_point.x, start_point.y
    start_ctl_point = start_point + Point((np.cos(start_angle) * ctl_length, np.sin(start_angle) * ctl_length))
    start_ctl = start_ctl_point.x, start_ctl_point.y
    end_pos = end_point.x, end_point.y
    end_ctl_point = end_point + Point((np.cos(end_angle) * ctl_length, np.sin(end_angle) * ctl_length))
    end_ctl = end_ctl_point.x, end_ctl_point.y

    control_points = np.array([start_pos, start_ctl, end_ctl, end_pos])
    n_points = 100

    def bernstein_poly(n, i, t):
        return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)

    def bezier(t, control_points):
        n = len(control_points) - 1
        return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)

    traj = []

    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))

    path = np.array(traj)
    plt.plot(path[:, 0], path[:, 1])
    plt.show()


def get_dubins_path(Pi, Pf, alpha, beta, ratio, tp="RSR", show_path=True):
    ##########################################################
    # 功能:
    #   输入起止点的坐标,方向,转弯半径,曲线的种类，计算Dubins曲线
    #
    # 输入参数:
    #   起点坐标:
    #       Pi(x,y)
    #   终点坐标:
    #       Pf(x,y)
    #   起点角度:
    #       alpha(弧度)
    #   终点角度:
    #       beta(弧度)
    #   转弯半径:
    #       ratio
    #   曲线类型:
    #       tp "RSR","LSL","RSL","LSR"
    #       ( R顺时针 L逆时针 S直线 default："RSR" )
    #   是否可视化结果:
    #       show_path (default=True)
    #
    # 输出结果：三元组
    #   切出点坐标posiA:[A_x,A_y]
    #   切入点坐标posiB:[B_x,B_y]
    #   三段距离dist lst:[dist1出发圆弧,dist2直线,dist3到达圆弧
    #
    # Author: jia-yf19@mails.tsinghua.edu.cn
    # Date  : 2021.10.28
    #
    ##########################################################

    # 坐标变换
    Vec_Move = np.matrix([[1., 0., -Pi[0]], [0., 1., -Pi[1]], [0., 0., 1.]])  # 平移
    temp_theta = -(np.arctan2(Pf[1] - Pi[1], Pf[0] - Pi[0]) + np.pi)
    alpha_ = alpha + temp_theta
    beta_ = beta + temp_theta
    Vec_Rotate = np.matrix([[np.cos(temp_theta), -np.sin(temp_theta), 0.],
                            [np.sin(temp_theta), np.cos(temp_theta), 0.],
                            [0., 0., 1.]])  # 旋转
    Vec_Length = np.matrix([[1. / ratio, 0., 0.],
                            [0., 1. / ratio, 0.],
                            [0., 0., 1.]])  # 拉伸
    Vec_Trans = Vec_Length * Vec_Rotate * Vec_Move  # 变换矩阵

    temp_res = Vec_Trans * [[Pi[0]], [Pi[1]], [1]]
    Pi_ = [temp_res[0, 0], temp_res[1, 0]]
    temp_res = Vec_Trans * [[Pf[0]], [Pf[1]], [1]]
    Pf_ = [temp_res[0, 0], temp_res[1, 0]]

    # 曲线计算
    if tp == "RSR":  # 双逆时针路线
        O1_x, O1_y = np.sin(alpha_), -np.cos(alpha_)  # 起点圆
        O2_x, O2_y = Pf_[0] + np.sin(beta_), -np.cos(beta_)  # 终点圆
        theta = np.arctan2(O2_y - O1_y, O2_x - O1_x)  # 圆心夹角
        pho1 = (theta - alpha_) % (2.0 * np.pi)  # 起点圆弧度
        pho2 = np.sqrt((O1_x - O2_x) ** 2 + (O1_y - O2_y) ** 2)  # 直线长度
        pho3 = (beta_ - pho1 - pho2 - alpha_) % (2.0 * np.pi)  # 终点圆弧度
        A_x, A_y = np.sin(alpha_) - np.sin(alpha_ + pho1), -np.cos(alpha_) + np.cos(alpha_ + pho1)  # 切出点
        B_x, B_y = Pf_[0] + np.sin(beta_) - np.sin(alpha_ + pho1), -np.cos(beta_) + np.cos(alpha_ + pho1)  # 切入点
    elif tp == "LSL":  # 双顺时针路线
        O1_x, O1_y = -np.sin(alpha_), np.cos(alpha_)  # 起点圆
        O2_x, O2_y = Pf_[0] - np.sin(beta_), np.cos(beta_)  # 终点圆
        theta = np.arctan2(O2_y - O1_y, O2_x - O1_x)  # 圆心夹角
        pho1 = (theta - alpha_) % (2.0 * np.pi)  # 起点圆弧度
        pho2 = np.sqrt((O1_x - O2_x) ** 2 + (O1_y - O2_y) ** 2)  # 直线长度
        pho3 = (beta_ - pho1 - pho2 - alpha_) % (2.0 * np.pi)  # 终点圆弧度
        A_x, A_y = -np.sin(alpha_) + np.sin(alpha_ + pho1), np.cos(alpha_) - np.cos(alpha_ + pho1)  # 切出点
        B_x, B_y = Pf_[0] - np.sin(beta_) + np.sin(alpha_ + pho1), np.cos(beta_) - np.cos(alpha_ + pho1)  # 切入点
    elif tp == "RSL":  # 八字交叉路线
        O1_x, O1_y = np.sin(alpha_), -np.cos(alpha_)  # 起点圆
        O2_x, O2_y = Pf_[0] - np.sin(beta_), np.cos(beta_)  # 终点圆
        theta = np.arctan2(O2_y - O1_y, O2_x - O1_x)  # 圆心夹角
        dis_O1O2 = np.sqrt((O2_y - O1_y) ** 2 + (O2_x - O1_x) ** 2)
        if dis_O1O2 < 2.0:
            print("The geometry constraint is not satisfied!")
            return None
        else:
            theta_O1A = np.arccos(2. / dis_O1O2)  # 圆1圆心和A夹角
            A_x, A_y = np.sin(alpha_) + np.cos(theta + theta_O1A), -np.cos(alpha_) + np.sin(theta + theta_O1A)  # 切出点
            B_x, B_y = Pf_[0] - np.sin(beta_) - np.cos(theta + theta_O1A), np.cos(beta_) - np.sin(theta + theta_O1A)  # 切入点
    else:  # "LSR" # 另一种八字交叉路线
        O1_x, O1_y = -np.sin(alpha_), np.cos(alpha_)  # 起点圆
        O2_x, O2_y = Pf_[0] + np.sin(beta_), -np.cos(beta_)  # 终点圆
        theta = np.arctan2(O2_y - O1_y, O2_x - O1_x)  # 圆心夹角
        dis_O1O2 = np.sqrt((O2_y - O1_y) ** 2 + (O2_x - O1_x) ** 2)
        if dis_O1O2 < 2.0:
            print("The geometry constraint is not satisfied!")
            return None
        else:
            theta_O1A = np.arccos(2. / dis_O1O2)  # 圆1圆心和A夹角
            A_x, A_y = -np.sin(alpha_) + np.cos(theta - theta_O1A), np.cos(alpha_) + np.sin(theta - theta_O1A)  # 切出点
            B_x, B_y = Pf_[0] + np.sin(beta_) - np.cos(theta - theta_O1A), -np.cos(beta_) - np.sin(theta - theta_O1A)  # 切入点

    # 逆变换
    Vec_Trans_INV = Vec_Trans.I
    temp_res = Vec_Trans_INV * [[A_x], [A_y], [1]]
    A_x, A_y = (temp_res[0, 0], temp_res[1, 0])
    temp_res = Vec_Trans_INV * [[B_x], [B_y], [1]]
    B_x, B_y = (temp_res[0, 0], temp_res[1, 0])
    temp_res = Vec_Trans_INV * [[O1_x], [O1_y], [1]]
    O1_x, O1_y = (temp_res[0, 0], temp_res[1, 0])
    temp_res = Vec_Trans_INV * [[O2_x], [O2_y], [1]]
    O2_x, O2_y = (temp_res[0, 0], temp_res[1, 0])

    dis_lst = []  # 每一段的距离
    ati1, ati2 = np.arctan2(Pi[1] - O1_y, Pi[0] - O1_x), np.arctan2(A_y - O1_y, A_x - O1_x)
    if tp[0] == "R" and ati1 < ati2:
        ati1 += 2.0 * np.pi
    elif tp[0] == "L" and ati1 > ati2:
        ati1 -= 2.0 * np.pi
    dis_lst.append(ratio * abs(ati1 - ati2))  # 切出弧长度
    dis_lst.append(np.sqrt((A_x - B_x) ** 2 + (A_y - B_y) ** 2))  # 直线段距离
    atf1 = np.arctan2(Pf[1] - O2_y, Pf[0] - O2_x)
    atf2 = np.arctan2(B_y - O2_y, B_x - O2_x)
    if tp[2] == "R" and atf2 < atf1:
        atf2 += 2.0 * np.pi
    elif tp[2] == "L" and atf2 > atf1:
        atf2 -= 2.0 * np.pi
    dis_lst.append(ratio * abs(atf1 - atf2))  # 切入弧长度
    if show_path:
        plt.figure(figsize=(12, 12))
        # arrow_length = 0.75
        # # 出发向量
        # plt.arrow(Pi[0],
        #           Pi[1],
        #           arrow_length * np.cos(alpha),
        #           arrow_length * np.sin(alpha),
        #           width=0.01,
        #           length_includes_head=True,
        #           head_width=0.05,
        #           head_length=0.1,
        #           fc='b',
        #           ec='b')
        # # 到达向量
        # plt.arrow(Pf[0],
        #           Pf[1],
        #           arrow_length * np.cos(beta),
        #           arrow_length * np.sin(beta),
        #           width=0.01,
        #           length_includes_head=True,
        #           head_width=0.05,
        #           head_length=0.1,
        #           fc='r',
        #           ec='r')
        plt.scatter(np.array([Pi[0], A_x, B_x, Pf[0]]), np.array([Pi[1], A_y, B_y, Pf[1]]))  # 起止点,切点
        plt.scatter(np.array([O1_x, O2_x]), np.array([O1_y, O2_y]), c='r')  # 圆心
        the = np.linspace(-np.pi, np.pi, 200)  # 完整圆形轮廓
        plt.scatter((O1_x + ratio * np.cos(the)), (O1_y + ratio * np.sin(the)), s=0.1, c='r')
        plt.scatter((O2_x + ratio * np.cos(the)), (O2_y + ratio * np.sin(the)), s=0.1, c='r')

        # 出发圆弧轨迹
        the = np.linspace(ati2, ati1, 200)
        plt.plot((O1_x + ratio * np.cos(the)), (O1_y + ratio * np.sin(the)), c='b')

        # 直线段
        plt.plot([A_x, B_x], [A_y, B_y], c='b')

        # 到达圆弧轨迹
        the = np.linspace(atf1, atf2, 200)
        plt.plot((O2_x + ratio * np.cos(the)), (O2_y + ratio * np.sin(the)), c='b')

        ax = plt.gca()  # 设置等比例
        ax.set_aspect(1)
        plt.title("Dubins Path")
        plt.grid(True)
        plt.show()

    return ([A_x, A_y], [B_x, B_y], dis_lst)  # 切出点,切入点,每一段的距离


if __name__ == "__main__":
    path_point = get_dubins_path(Pi=(121.0120, 22.01), Pf=(121.0105, 22.01020), alpha=0, beta=-3.14, ratio=0.00005, tp="LSL")
    print(path_point)