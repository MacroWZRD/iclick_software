import cv2 as cv
import numpy as np
import mediapipe as mp
from pynput.mouse import Controller
import FileOpenerModule as fm
import math

mouse = Controller()

EYELEVEL = [168, 6]

new_screen_normal = lambda : list(map(float, fm.ReadLineWithKey("runtimesettings.txt", "ScreenRatio").split()[1:]))
screen_normal = new_screen_normal()

asking_time = 0

# left eyes indices
LEFT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398
]

# right eyes indices
RIGHT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246
]


class irisDetector():

    def __init__(self, static_image_mode=False, maxFaces=1, refineLm=True, detectionCon=0.5, trackCon=0.5 ):
        self.static_image_mode = static_image_mode
        self.max_num_faces = maxFaces
        self.refine_landmarks = refineLm
        self.min_detection_confidence = detectionCon
        self.min_tracking_confidence = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.static_image_mode, self.max_num_faces, self.refine_landmarks, self.min_detection_confidence, self.min_tracking_confidence)
        self.face_coordinates = []

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

        # iris indices
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # eye indices
        self.LEFT_EYE = [362, 466, 374, 385]
        self.RIGHT_EYE = [33, 173, 145, 159]

    def findIris(self, img, draw=True):

        img = cv.flip(img, 1)
        img_h, img_w = img.shape[:2]

        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.imgRGB)

        if self.results.multi_face_landmarks:

            # create a list containing the coordinates of all the landmarks on the face
            self.mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in self.results.multi_face_landmarks[0].landmark])

            # map the iris points
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(self.mesh_points[self.LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(self.mesh_points[self.RIGHT_IRIS])

            # calculate the center of each iris
            self.center_left = np.array([l_cx, l_cy], dtype=np.int32)
            self.center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # draw the circle for the iris
            if draw:
                cv.circle(img, self.center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(img, self.center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

        return img

    def detectBlink(self, img):

        blinkDbl = False
        blinkRight = False
        blinkLeft = False

        face_coordinates = []

        try:
            for faceLms in self.results.multi_face_landmarks:
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)

                    face_coordinates.append([id, x, y])

            self.face_coordinates = face_coordinates

            # map left eye points
            rEye_left_point = (self.face_coordinates[33][1], self.face_coordinates[33][2])
            rEye_right_point = (self.face_coordinates[173][1], self.face_coordinates[173][2])
            rEye_center_top = (self.face_coordinates[159][1], self.face_coordinates[159][2])
            rEye_center_bottom = (self.face_coordinates[145][1], self.face_coordinates[153][2])

            # map right eye points
            lEye_left_point = (face_coordinates[398][1], face_coordinates[398][2])
            lEye_right_point = (face_coordinates[263][1], face_coordinates[263][2])
            lEye_center_top = (face_coordinates[386][1], face_coordinates[386][2])
            lEye_center_bottom = (face_coordinates[374][1], face_coordinates[374][2])

            # draw the lines for the left eye
            right_hor_line = cv.line(img, rEye_left_point, rEye_right_point, (0, 255, 0), 1)
            right_ver_line = cv.line(img, rEye_center_top, rEye_center_bottom, (0, 255, 0), 1)

            # draw the lines of the right eye
            left_hor_line = cv.line(img, lEye_left_point, lEye_right_point, (0, 255, 0), 1)
            left_ver_line = cv.line(img, lEye_center_top, lEye_center_bottom, (0, 255, 0), 1)

            # calculate the length of the lines for the right eye
            right_hor_line_length = math.hypot(rEye_left_point[0] - rEye_right_point[0], rEye_left_point[1] - rEye_right_point[1])
            right_ver_line_length = math.hypot(rEye_center_top[0] - rEye_center_bottom[0], rEye_center_top[1] - rEye_center_bottom[1])

            right_ratio = right_hor_line_length/ 1/100000 if right_ver_line_length <= 0 else right_ver_line_length

            # calculate the length of the lines for the left eye
            left_hor_line_length = math.hypot(lEye_left_point[0] - lEye_right_point[0], lEye_left_point[1] - lEye_right_point[1])
            left_ver_line_length = math.hypot(lEye_center_top[0] - lEye_center_bottom[0], lEye_center_top[1] - lEye_center_bottom[1])

            left_ratio = left_hor_line_length/ 1/100000 if left_ver_line_length <= 0 else left_ver_line_length

            # check if both eyes are closed
            if right_ratio < 3.5 and left_ratio < 3.5:
                blinkDbl = True

            elif left_ratio < 3.5:
                blinkLeft = True

            elif right_ratio < 3.5:
                blinkRight = True

            return blinkDbl, blinkLeft, blinkRight

        except:

            return blinkDbl, blinkLeft, blinkRight

    def moveCursor(self, img, w, h, x, y):

        if self.results.multi_face_landmarks:
            global screen_normal
            sensitivity = 10

            gaze_ratio = ((np.max(self.mesh_points[EYELEVEL][:, 0]) + np.min(self.mesh_points[EYELEVEL][:, 0])) / 2,
                          (np.max(self.mesh_points[EYELEVEL][:, 1]) + np.min(self.mesh_points[EYELEVEL][:, 1])) / 2)
            if len(fm.ReadLineWithKey("runtimesettings.txt", "Recalibrating").split()) > 1:
                #mouse.position =
                bbox = [(0, 0), (640, 480)]

                cv.rectangle(img, bbox[0], bbox[1], (244, 242, 237), -1)

                cv.putText(img, "Align the Corners", (155, 240), cv.FONT_HERSHEY_PLAIN, 2, (66, 45, 43), 1)

                circles = ((25, 455), (615, 25), (25, 25), (615, 455))
                for i in range(8):
                    cv.circle(img, circles[i//2], 12 if i%2 == 0 else 3, (60, 35, 239) if i%2 == 0 else (66, 45, 43), -1)

                trackingPoint = [self.face_coordinates[4][1], self.face_coordinates[4][2]]
                cv.rectangle(img, (trackingPoint[0] - 270, trackingPoint[1] - 210), (trackingPoint[0] + 270, trackingPoint[1] + 210) , (174, 153, 141), 1)
                screen_normal = new_screen_normal()

            pure_ratios = (gaze_ratio[0] - screen_normal[0]) / (screen_normal[2] - screen_normal[0]) if (screen_normal[2] - screen_normal[0]) != 0 else 1 / 100, \
                          (gaze_ratio[1] - screen_normal[1]) / (screen_normal[3] - screen_normal[1]) if (screen_normal[3] - screen_normal[1]) != 0 else 1 / 100
            fm.WriteLineWithKey("data.txt", "GazeRatio", str(gaze_ratio[0]) + " " + str(gaze_ratio[1]))
            fm.WriteLineWithKey("data.txt", "Center", str(x + w//2) + " " + str(y + h//2))
            inversed = len(fm.ReadLineWithKey("runtimesettings.txt", "Inversed").split()) < 2
            try:
                mouse.move((-1 * abs(pure_ratios[0] - 0.3) * sensitivity if pure_ratios[0] < 0.3 else
                            (1 * abs(pure_ratios[0] - 0.8) * sensitivity if pure_ratios[0] > 0.8 else 0))
                           * (-1 if inversed else 1),
                           (-1 * abs(pure_ratios[1] - 0.3) * sensitivity if pure_ratios[1] < 0.3 else
                            (1 * abs(pure_ratios[1] - 0.8) * sensitivity if pure_ratios[1] > 0.8 else 0))
                           * (-1 if inversed else 1))
            except:
                pass

    def findPosition(self):
        return self.center_left, self.center_right

a = irisDetector()