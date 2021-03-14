import numpy as np
import matplotlib.pyplot as plt


class Reserford:
    """
    ???
    """

    def __init__(self):
        self.alphaMass = 4.001506179125
        self.alphaEnergy = 5
        self.alphaCharge = 2 * 1.6021766208e-19
        self.coreMass = 196.966569
        self.coreCharge = 79 * 1.6021766208e-19
        self.R = 1.66e-10
        self.alpha_x = 0
        self.alpha_y = 0
        self.core_x = 1.66e-46
        self.core_y = 1.66e-49
        self.k = 1.380649e-23
        self.time = 1e-53
        self.delta_t = 1e-57
        self.val = []

    def calculate(self):
        """via Verlet integration"""

        self.alphaMass *= 1.660538918685e-27
        self.alphaEnergy *= 1.6021766e-13
        self.coreMass *= 1.660538918685e-27

        def apply_forces_alpha():
            nonlocal pos_alpha, pos_core
            r12 = pos_alpha - pos_core
            mod_r12 = np.sqrt(np.dot(r12, r12))
            f = self.k * self.alphaCharge * self.coreCharge / np.dot(r12, r12) * r12 / mod_r12\
                * (0, 1 - mod_r12 / self.R)[int(mod_r12 < self.R)]
            a = f / self.alphaMass
            return a

        def apply_forces_core():
            nonlocal pos_alpha, pos_core
            r12 = pos_core - pos_alpha
            mod_r12 = np.sqrt(np.dot(r12, r12))
            f = self.k * self.alphaCharge * self.coreCharge * self.R / np.dot(r12, r12) * r12 / mod_r12\
                * (0, 1 - mod_r12 / self.R)[int(mod_r12 < self.R)]
            a = f / self.coreMass
            return a

        def update(dt):
            nonlocal pos_alpha, vel_alpha, acc_alpha, pos_core, vel_core, acc_core

            new_pos_alpha = pos_alpha + vel_alpha * dt + acc_alpha * dt * dt / 2
            new_acc_alpha = apply_forces_alpha()
            new_vel_alpha = vel_alpha + (acc_alpha + new_acc_alpha) * dt / 2

            new_pos_core = pos_core + vel_core * dt + acc_core * dt * dt / 2
            new_acc_core = apply_forces_core()
            new_vel_core = vel_core + (acc_core + new_acc_core) * dt / 2

            pos_core = new_pos_core
            acc_core = new_acc_core
            vel_core = new_vel_core

            pos_alpha = new_pos_alpha
            acc_alpha = new_acc_alpha
            vel_alpha = new_vel_alpha

        pos_core = np.array([self.core_x, self.core_y])
        vel_alpha = np.array([np.sqrt(2 * self.alphaEnergy / self.alphaMass), 0])

        pos_alpha = np.array([self.alpha_x, self.alpha_y])
        vel_core = np.array([0, 0])

        acc_alpha = apply_forces_alpha()
        acc_core = apply_forces_core()

        self.val = []
        t = 0
        while t < self.time:
            self.val.append([pos_alpha, vel_alpha, acc_alpha, pos_core, vel_core, acc_core, t])
            t += self.delta_t
            update(self.delta_t)

    def plot(self):
        """
        1. alpha-p. and core position trajectory
        2. alpha-p. velocity hodograph
        3. core velocity hodograph
        4. alpha-p. acceleration hodograph
        5. core acceleration hodograph
        """
        if __name__ == '__main__':
            pos_both_graph = plt.subplot(211)
            pos_alpha_graph = plt.subplot(223)
            pos_core_graph = plt.subplot(224)

            pos_both_graph.set_xlabel('x (м)')
            pos_both_graph.set_ylabel('y (м)')
            pos_both_graph.set_title('Траектория обоих частиц')
            pos_both_graph.grid()

            pos_alpha_graph.set_xlabel('x (м)')
            pos_alpha_graph.set_ylabel('y (м)')
            pos_alpha_graph.set_title('Траектория альфа-частицы')
            pos_alpha_graph.grid()

            pos_core_graph.set_xlabel('x (м)')
            pos_core_graph.set_ylabel('y (м)')
            pos_core_graph.set_title('Траектория ядра атома золота')
            pos_core_graph.grid()

            pos_both_graph.plot([row[0][0] for row in self.val], [row[0][1] for row in self.val])
            pos_both_graph.scatter([row[3][0] for row in self.val], [row[3][1] for row in self.val], marker='.')
            pos_alpha_graph.plot([row[0][0] for row in self.val], [row[0][1] for row in self.val])
            pos_core_graph.plot([row[3][0] for row in self.val], [row[3][1] for row in self.val])

            plt.show()

        else:
            pos_both_graph = plt.subplot(211)
            pos_alpha_graph = plt.subplot(223)
            pos_core_graph = plt.subplot(224)

            pos_both_graph.set_xlabel('x (м)')
            pos_both_graph.set_ylabel('y (м)')
            pos_both_graph.set_title('Траектория обоих частиц')
            pos_both_graph.grid()

            pos_alpha_graph.set_xlabel('x (м)')
            pos_alpha_graph.set_ylabel('y (м)')
            pos_alpha_graph.set_title('Траектория альфа-частицы')
            pos_alpha_graph.grid()

            pos_core_graph.set_xlabel('x (м)')
            pos_core_graph.set_ylabel('y (м)')
            pos_core_graph.set_title('Траектория ядра атома золота')
            pos_core_graph.grid()

            pos_both_graph.plot([row[0][0] for row in self.val], [row[0][1] for row in self.val])
            pos_both_graph.scatter([row[3][0] for row in self.val], [row[3][1] for row in self.val], marker='.')
            pos_alpha_graph.plot([row[0][0] for row in self.val], [row[0][1] for row in self.val])
            pos_core_graph.plot([row[3][0] for row in self.val], [row[3][1] for row in self.val])

            plt.show()


if __name__ == '__main__':
    r = Reserford()
    r.calculate()
    r.plot()
