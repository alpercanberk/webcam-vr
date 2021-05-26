import cv2
import numpy as np
from config import *
from utils import *

class Circle:
    def __init__(self, x=[0, 0], size=10, mass=1, v=[0,0], color=[255, 255, 255]):
        self.x = x
        self.size = size
        self.mass = mass
        self.v = v
        self.color = color

    def draw(self, img):
        # print(tuple(self.x))
        coordinates = [int(x) for x in self.x]
        cv2.circle(img, coordinates, self.size, self.color, cv2.FILLED)

    def update(self, delta_t):
        for i in range(len(self.x)):
            self.x[i] = self.x[i] + self.v[i] * delta_t
            if(self.x[i] + self.size > SCREEN_SIZE[i]):
                #readjust position
                self.x[i] = SCREEN_SIZE[i] - self.size - 5
                #bounce
                self.v[i] = -.9 * self.v[i]
            if(self.x[i] - self.size < 0):
                #readjust position
                self.x[i] = 5 + self.size
                #bounce
                self.v[i] = -.9 * self.v[i]
            self.v[i] = .99 * self.v[i]

    def update_hand_collision(self, canvas, lines, colliding_v):
        for line in lines:
            if(self.colliding_with_line([line])):
                nearest_to_wall = nearest2line(self.x, line.x1, line.x2)
                # cv2.circle(canvas, parse_coord(nearest_to_wall), 3, (255, 255, 255), -1)
                # cv2.line(canvas, parse_coord(self.x), parse_coord(nearest_to_wall), (255, 255, 255))
                orth_vec = np.array(self.x) - np.array(nearest_to_wall)
                dist = np.linalg.norm(orth_vec)
                orth_vec = orth_vec/dist
                rad, deg = vector_angle(np.array([orth_vec[0], orth_vec[1]]))
                c_matrix = collision_matrix(rad + np.pi/2)

                offset_vector = (self.size - dist) * (orth_vec/np.linalg.norm(orth_vec))

                dist = np.linalg.norm(orth_vec)

                temp_v = self.v - colliding_v
                temp_v[1] = -1 * temp_v[1]

                new_v = np.matmul(c_matrix, np.array(self.v))

                self.x[0] +=  offset_vector[0]
                self.x[1] +=  offset_vector[1]

                self.v[0] = new_v[0]
                self.v[1] = -1*(new_v[1])

    def update_ball_collision(self, canvas, other_balls):
        for other_ball in other_balls:
            if(np.linalg.norm(np.array(other_ball.x) - np.array(self.x)) < self.size + other_ball.size):

                orth_vec = np.array(other_ball.x) - np.array(self.x)
                dist = np.linalg.norm(orth_vec)
                orth_vec = orth_vec/np.linalg.norm(orth_vec)
                rad, deg = vector_angle(np.array([orth_vec[0], orth_vec[1]]))
                c_matrix = collision_matrix(rad + np.pi/2)

                offset_vector = orth_vec/np.linalg.norm(orth_vec)
                pos_offset_scale = 5

                #idk why these are negative?
                self.x[0] -= (self.size + other_ball.size - dist) * offset_vector[0]
                self.x[1] -= (self.size + other_ball.size - dist) * offset_vector[1]

                v_1_i = self.v
                v_2_i = other_ball.v

                v_1_f = np.zeros((2,))

                m_1 = self.size
                m_2 = other_ball.size

                sum = m_1 + m_2

                for i in range(0, 2):
                    v_1_f[i] = .9 * ((m_1 - m_2)*v_1_i[i] + (2*m_2)*v_2_i[i])/sum

                temp_v = v_1_f
                temp_v[1] = -1 * temp_v[1]

                self.v[0] = temp_v[0]
                self.v[1] = temp_v[1]
            # else:
                # self.color = (255, 0, 0)


    def colliding_with_pts(self, pts_array):
        for p in pts_array:
            if(np.linalg.norm(p - self.x) < self.size):
                return True
        return False

    def colliding_with_line(self, lines):
        colliding_line = None
        min_dist = 10e5
        for line in lines:
            dist = pnt2line(self.x, line.x1, line.x2)
            if(dist < self.size and dist < min_dist):
                colliding_line = line
        return colliding_line


class Line:
    def __init__(self, x1=[0, 0], x2=[0, 0], color=(255, 255, 255)):
        self.x1 = x1
        self.x2 = x2
        self.color = color

    def draw(self, img):
        # print(tuple(self.x))
        cv2.line(img, self.x1, self.x2, self.color, thickness=2)

    def update_start(self, new_x1):
        self.x1 = new_x1

    def update_end(self, new_x2):
        self.x2 = new_x2


if __name__ == "__main__":
    print(find_min_dist((635, 617), (232, 183), (263,207)))
