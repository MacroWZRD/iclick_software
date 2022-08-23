from pynput import keyboard
from pynput.mouse import Controller
import FileOpenerModule as fm
import time

mouse = Controller()
fm.WriteLineWithKey("runtimesettings.txt", "Recalibrating", "")

def MoveCalibrate():
    #print("RECALIBRATING")
    #GUI is in the original script
    screen_normal = [-10000, -10000, 10000, 10000] #will work to shrink the range
    start_time = time.time()
    duration = float(fm.ReadLineWithKey("settings.txt", "CalibrationTime").split()[1])
    fm.WriteLineWithKey("runtimesettings.txt", "Recalibrating", "TRUE")
    coor = tuple(map(int, fm.ReadLineWithKey("data.txt", "Center").split()[1:]))
    mouse.position = coor
    while time.time() - start_time < duration:
        raw = fm.ReadLineWithKey("data.txt", "GazeRatio")
        if raw == None:
            continue
        gaze_ratios = tuple(map(float, raw.split()[1:]))
        screen_normal[0], screen_normal[2] = max(gaze_ratios[0], screen_normal[0]), min(gaze_ratios[0], screen_normal[2])
        screen_normal[1], screen_normal[3] = max(gaze_ratios[1], screen_normal[1]), min(gaze_ratios[1], screen_normal[3])

    fm.WriteLineWithKey("runtimesettings.txt", "ScreenRatio",  str(screen_normal[0]) + " " + str(screen_normal[1]) + " " + str(screen_normal[2]) + " " + str(screen_normal[3]))
    fm.WriteLineWithKey("runtimesettings.txt", "Recalibrating", "")
    print("Recalibration Finished")

def InverseCalibrate():
    is_inversed = len(fm.ReadLineWithKey("runtimesettings.txt", "Inversed").split()) > 1
    fm.WriteLineWithKey("runtimesettings.txt", "Inversed", "" if is_inversed else "True")
    print("Inversed")

with keyboard.GlobalHotKeys({"<ctrl>+r" : MoveCalibrate, "<ctrl>+i" : InverseCalibrate}) as h:
    h.join()