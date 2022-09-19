import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class SimpleEnv():
    """
    A simple 2D environment with robots represented as discs with 1D dynamics:
    x_dot = u

    """

    def __init__(self, params, task_ves_info):
        self.dt = params.dt
        self.eps = params.eps
        self.robot_diameter = 0.5

        task_ves_info = task_ves_info["content"]["arguments"]
        taskArea = task_ves_info["taskArea"]

        targetInfo = task_ves_info["targetInfo"]
        target_number = len(targetInfo)
        tar_id = list()
        tar_center = list()
        for tar_info in targetInfo:
            tar_id.append(tar_info["targetId"])
            tar_center.append(tar_info["targetPos"])

        susTargetInfo = task_ves_info["susTargetInfo"]
        sus_target_number = len(susTargetInfo)
        sus_tar_id = list()
        sus_tar_center = list()
        for sus_tar_info in susTargetInfo:
            sus_tar_id.append(sus_tar_info["susTargetId"])
            sus_tar_area = sus_tar_info["susTargetArea"]
            center = [(sus_tar_area[0][0] + sus_tar_area[2][0]) / 2, (sus_tar_area[0][1] + sus_tar_area[1][1]) / 2]
            sus_tar_center.append(center)

        vesInfo = task_ves_info["vesInfo"]
        ves_number = len(vesInfo)
        tid = list()
        vesPos = list()
        sonarWidth = list()
        speed = list()
        for ves_info in vesInfo:
            tid.append(ves_info["tid"])
            vesPos.append(ves_info["vesPos"])
            sonarWidth.append(ves_info["sonarWidth"])
            speed.append(ves_info["speed"])

        target_threat_radius = task_ves_info["config"]["targetThreatRadius"]
        initial = task_ves_info["initial"]

        self.n_agents = ves_number
        self.n_tasks = sus_target_number

        map_point = taskArea[1]
        length_scale = (taskArea[2][0] - taskArea[0][0]) / 2
        width_scale = (taskArea[0][1] - taskArea[1][1]) / 2

        #initialize robot state
        # self.x = (np.random.rand(self.n_agents,2)-0.5)*4

        self.x = np.array([])
        for vespos in vesPos:
            if not self.x.size:
                self.x = (np.array(vespos) - np.array([map_point])) / np.array([length_scale, width_scale]) - np.array([1, 1])
            else:
                pos = (np.array(vespos) - np.array([map_point])) / np.array([length_scale, width_scale]) - np.array([1, 1])
                self.x = np.vstack((self.x, pos))
        self.x *= np.array([2 * length_scale / width_scale, 2])
        # print(self.x)

        # ---------task variables-------------
        # import task locations
        # self.tasks = params.tasks
        # print(params.tasks)

        self.tasks = np.array([])
        for center in sus_tar_center:
            if not self.tasks.size:
                self.tasks = (np.array(center) - np.array([map_point])) / np.array([length_scale, width_scale]) - np.array([1, 1])
            else:
                pos = (np.array(center) - np.array([map_point])) / np.array([length_scale, width_scale]) - np.array([1, 1])
                self.tasks = np.vstack((self.tasks, pos))
        self.tasks *= np.array([6, 2])
        # print(self.tasks)

        self.durations = params.durations

        # import task dependency matrix:
        # a "1" in row i, column j represents the dependence of task i on task j
        self.task_dependency_matrix = params.task_dependency_matrix

        # task_readiness vector represents the percentage of dependencies that are fulfilled for task i
        # a value of 1 means the task is ready
        self.task_readiness = np.zeros((self.n_tasks,))
        self.task_readiness_history = []
        self.assignment_matrix = np.zeros((self.n_agents,self.n_tasks),dtype=bool) # a "1" in row i, column j means task j is assigned to robot i
        self.assignment_list = [] # ordered list of tasks assigned to each agent, by task number
        self.task_done = np.zeros((self.n_tasks,), dtype=bool)
        self.task_times = np.zeros((self.n_tasks,))
        self.task_time_history = []
        self.task_done_history = []
        self.state_history = []


    def step(self, action):
        """
        :param action: n_agents x
         +2 numpy array
        :return: new state x, completion states of all tasks
        """
        #TODO: check if action is valid

        self.x = self.x + self.dt*action

        self.check_progress()
        self.state_history.append(self.x)
        return self.x, self.task_done

    def check_progress(self):
        for robot in range(self.n_agents):
            assigned_tasks = self.assignment_list[robot]
            for task in assigned_tasks:
                loc = self.tasks[task,:]
                dist = np.linalg.norm(loc-self.x[robot,:])
                if dist<self.eps and self.task_readiness[task]==1:
                    self.task_times[task] += self.dt
                    if self.task_times[task]>self.durations[task]:
                        self.task_done[task] = True

        donecopy = np.copy(self.task_done)
        timecopy = np.copy(self.task_times)
        self.task_done_history.append(donecopy)
        self.task_time_history.append(timecopy)

        # propagate updates to task_readiness vector
        self.update_task_readiness()

    def build_agent_assignment(self):
        # build agent assignment list
        for robot in range(self.n_agents):
            inds = np.arange(self.n_tasks)
            assigned_tasks = inds[self.assignment_matrix[robot, :]]
            self.assignment_list.append(assigned_tasks)

    def build_assignment_matrix(self):
        for robot in range(self.n_agents):
            assignment = self.assignment_list[robot]
            self.assignment_matrix[robot,assignment] = True
            
    def update_task_readiness(self):
        task_dependency_count = np.sum(self.task_dependency_matrix,axis=1) # count the number of tasks each task is dependent upon
        task_dependency_current = self.task_dependency_matrix @ self.task_done
        self.task_readiness = np.nan_to_num(np.divide(task_dependency_current,task_dependency_count),nan=1,posinf=1) #if zero dependencies, gives 1
        self.task_readiness_history.append(np.copy(self.task_readiness))
        
    def plot(self):
        # animate
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-6, 6), ylim=(-2, 2))
        ax.set_aspect('equal')
        ax.grid()
        ax.set_title('Simple Environment - lightest robot has highest number')

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

        circles = []
        circle_colors = []
        for r in range(self.n_agents):
            circle_colors.append(((r+0.5)/self.n_agents,(r+0.5)/self.n_agents,(r+0.5)/self.n_agents))
        task_circles = []

        def init():
            for task in range(self.n_tasks):
                task_circle = plt.Circle(self.tasks[task, :], 0.3, facecolor='white', edgecolor='k', linewidth=1,alpha=1)
                task_circles.append(task_circle)
                ax.add_patch(task_circles[task])
                xy = (self.tasks[task,0]-0.1,self.tasks[task,1]-0.1)
                ax.annotate(str(task),xy = xy, xytext=xy)

            for r in range(self.n_agents):
                circles.append(
                    plt.Circle((self.state_history[0][r, :]), 0.2, facecolor=circle_colors[r], edgecolor='k', linewidth=1))
                #xy = (self.state_history[0][r, 0] - 0.1, self.state_history[0][r, 1] - 0.1)
                #ax.annotate(str(r + 1), xy=xy, xytext=xy)

                ax.add_patch(circles[r])


            return circles+task_circles

        def animate(i):
            for r in range(self.n_agents):
                circles[r].center = self.state_history[i][r, :]
                #xy = (self.state_history[i][r, 0] - 0.1, self.state_history[i][r, 1] - 0.1)
                #t = ax.annotate(str(r + 1), xy=xy, xytext=xy)
                #t.xy = (1,2)

            current_task_done = self.task_done_history[i]
            current_readiness = self.task_readiness_history[i]
            for task in range(self.n_tasks):
                c = ''
                if current_task_done[task]:
                    c = 'forestgreen'
                elif current_readiness[task] == 1:
                    c = 'yellow'
                else:
                    c = 'tomato'
                task_circles[task].set_facecolor(c)

            return circles+task_circles



        ani = animation.FuncAnimation(fig, animate, range(len(self.state_history)),
                                      interval=self.dt * 1000, blit=False, init_func=init)
        ani.save('imgs/most_recent_animation.mp4',writer=writer)
        plt.show()

    def output_trajs(self):
        pass