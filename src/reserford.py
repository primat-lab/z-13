import numpy as np
import matplotlib.pyplot as plt


class Reserford:
    """
    ???
    """

    def __init__(self):
        self.alphaMass = 6.644
        self.alphaEnergy = 8.0109
        self.alphaCharge = 2 * 1.60217662
        self.coreMass = 410
        self.coreCharge = 79 * 1.60217662
        self.alpha_x = 0
        self.alpha_y = 0
        self.core_x = 1
        self.core_y = 0.51
        self.k = 0.05
        self.time = 10
        self.delta_t = 0.001
        self.val = []

    def calculate(self):
        """via Verlet integration"""

        def apply_forces_alpha():
            nonlocal pos_alpha, pos_core
            r12 = pos_alpha - pos_core
            f = self.k * self.alphaCharge * self.coreCharge / np.dot(r12, r12) * r12 / np.sqrt(np.dot(r12, r12))
            a = f / self.alphaMass
            return a

        def apply_forces_core():
            nonlocal pos_alpha, pos_core
            r12 = pos_core - pos_alpha
            f = self.k * self.alphaCharge * self.coreCharge / np.dot(r12, r12) * r12 / np.sqrt(np.dot(r12, r12))
            a = f / self.coreMass
            return a

        def update(dt):
            nonlocal pos_alpha, vel_alpha, acc_alpha, pos_core, vel_core, acc_core

            new_pos_core = pos_core + vel_core * dt + acc_core * dt * dt / 2
            new_pos_alpha = pos_alpha + vel_alpha * dt + acc_alpha * dt * dt / 2
            new_acc_core = apply_forces_core()
            new_vel_core = vel_core + (acc_core + new_acc_core) * dt / 2
            new_acc_alpha = apply_forces_alpha()
            new_vel_alpha = vel_alpha + (acc_alpha + new_acc_alpha) * dt / 2

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
            ax1 = plt.subplot(311)
            ax2 = plt.subplot(323)
            ax1.plot([row[0][0] for row in self.val], [row[0][1] for row in self.val])
            ax1.plot([row[3][0] for row in self.val], [row[3][1] for row in self.val])
            ax1.scatter([row[3][0] for row in self.val], [row[3][1] for row in self.val])
            ax2.plot([row[1][0] for row in self.val], [row[1][1] for row in self.val])
            ax3 = plt.subplot(324)
            ax3.plot([row[4][0] for row in self.val], [row[4][1] for row in self.val])
            ax4 = plt.subplot(325)
            ax4.plot([row[2][0] for row in self.val], [row[2][1] for row in self.val])
            ax5 = plt.subplot(326)
            ax5.plot([row[5][0] for row in self.val], [row[5][1] for row in self.val])
            plt.show()
        else:
            plt.scatter([row[3][0] for row in self.val], [row[3][1] for row in self.val])
            plt.plot([row[0][0] for row in self.val], [row[0][1] for row in self.val])
            plt.show()
            plt.plot([row[3][0] for row in self.val], [row[3][1] for row in self.val])
            plt.show()
            plt.plot([row[1][0] for row in self.val], [row[1][1] for row in self.val])
            plt.show()
            plt.plot([row[4][0] for row in self.val], [row[4][1] for row in self.val])
            plt.show()
            plt.plot([row[2][0] for row in self.val], [row[2][1] for row in self.val])
            plt.show()
            plt.plot([row[5][0] for row in self.val], [row[5][1] for row in self.val])
            plt.show()


if __name__ == '__main__':
    r = Reserford()
    r.calculate()
    r.plot()
