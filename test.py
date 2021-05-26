import cv2
import mediapipe as mp
import time
import numpy as np
from sprites import *

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands = NUM_HANDS, min_detection_confidence=.7)
mpDraw = mp.solutions.drawing_utils

pTime = time.time()
cTime = time.time()

features = np.zeros((NUM_HANDS, 21, 2))
prev_features = np.zeros((NUM_HANDS, 21, 2))
delta_pos = np.zeros((NUM_HANDS, 21, 2))

velocity_timesteps = 10
velocities = np.zeros((NUM_HANDS, velocity_timesteps, 21, 2))

ball_count = 20
balls = []
for i in range(ball_count):
    x_random = [np.random.random_integers(SCREEN_SIZE[0]), np.random.random_integers(SCREEN_SIZE[1])]
    v_random = [np.random.random_integers(-100, 100), np.random.random_integers(-100, 100)]
    c_random = (int(np.random.random_integers(30, 255)), int(np.random.random_integers(30, 255)), int(np.random.random_integers(30, 255)))
    s_random = np.random.random_integers(50, 60)
    balls.append(Circle(x_random, s_random, 1, v_random, c_random))

circle_1 = Circle([100,100], 100, 1, v=[100,0], color=(255, 255, 255))
circle_2 = Circle([1000,100], 80, 1, v=[-100,0])

connections = []
for _ in range(NUM_HANDS):
    connections.append([])
for hand_index in range(NUM_HANDS):
    for _ in range(len(HAND_CONNECTIONS)):
        connections[hand_index].append(Line([0,0], [0,0], (255, 0, 0)))
# wall = Line([300, 300], [800, 100], (255, 255, 255))
# wall = Line([1000, 700], [1200, 400], (255, 255, 255))
# wall = Line([1000, 200], [1200, 400], (255, 255, 255))
# wall = Line([0, 500], [200, 700])

while True:
    success, img = cap.read()
    canvas = np.zeros(img.shape[:3], dtype='uint8')
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(cv2.flip(imgRGB,1))
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        if(len(results.multi_hand_landmarks) > 1):
            for hand_index in range(len(results.multi_hand_landmarks)):
                print(">>>>")
                print(hand_index)
                print(">>>>")
                for id, lm in enumerate(results.multi_hand_landmarks[hand_index].landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    features[hand_index, id] = np.array([cx, cy])
                    delta_pos[hand_index, id] = prev_features[hand_index, id] - features[hand_index,id]
                    prev_features[hand_index, id] = features[hand_index, id]
                    cv2.circle(canvas, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                    for i in range(len(HAND_CONNECTIONS)):
                        if(id == HAND_CONNECTIONS[i][0]):
                            connections[hand_index][i].update_start([cx, cy])
                        if(id == HAND_CONNECTIONS[i][1]):
                            connections[hand_index][i].update_end([cx, cy])
            for hand in connections:
                for connection in hand:
                    connection.draw(canvas)
        else:
            for hand_index in range(NUM_HANDS):
                for id in range(21):
                    features[hand_index, id] = 0
                    delta_pos[hand_index, id] = 0

    cTime = time.time()
    delta_time = cTime - pTime
    fps = 1 / delta_time
    pTime = cTime

    velocities[:, 0:velocity_timesteps-1, :, :] = velocities[:, 1:velocity_timesteps, :, :]
    velocities[:, -1, :, :] = delta_pos/delta_time

    mean_v = np.mean(np.mean(velocities, axis=0), axis=0)

    mean_v[0] = -1 * mean_v[0]
    mean_v[1] = -1 * mean_v[1]

    for i in range(len(balls)):
        current_ball = balls[i]
        other_balls = [x for ind,x in enumerate(balls) if ind!=i]
        current_ball.draw(canvas)
        current_ball.update_hand_collision(canvas, connections[0], mean_v)
        current_ball.update_hand_collision(canvas, connections[1], mean_v)
        current_ball.update_ball_collision(canvas, other_balls)
        current_ball.update(delta_time)

    cv2.imshow("Image", canvas)


    if cv2.waitKey(20) & 0xFF==ord('d'):
            break
