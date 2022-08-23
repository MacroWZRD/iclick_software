from tkinter import *
from tkinter import ttk
import cv2 as cv
import IrisTrackingBasics as itm
import FileOpenerModule as fm
from pynput.mouse import Button, Controller
from PIL import Image, ImageTk
import os
import time

width=640
height=480

#set up GUI
root = Tk()
root.title("iclick.exe")
root.wm_iconbitmap("CLICK_9.ico")
root.resizable(False, False)
root.geometry("665x620+0+0")
guiFrame = ttk.Frame(root, padding=10)
guiFrame.grid()

#set control frame above video
controlFrame = ttk.Frame(guiFrame, padding=5)
controlFrame.grid(column=0, row=1)

#set controls
label = ttk.Label(controlFrame, text='Scaling Time:')
label.grid(column=0, row=0, padx=0)
s1 = IntVar()
s1.set(int(fm.ReadLineWithKey("settings.txt", "CalibrationTime").split()[1]))
cs1 = Scale(controlFrame, from_=5, to=30, variable=s1.get(), orient='horizontal', length=200)
cs1.grid(column=1, row=0, padx=10, ipady=8)

b1 = BooleanVar()
cb1 = ttk.Checkbutton(controlFrame, text='Inverse', variable=b1)
cb1.grid(column=2, row=0, padx=10)

#os.startfile(r"..\ICLICK\Recalibrate.exe")
def quitGui():
    print('quit')
    #os.system(r"TASKKILL /F /IM Recalibrate.exe /T")
    fm.WriteLineWithKey("settings.txt", "CalibrationTime", cs1.get()) #must assign when programs end since it might make file blank
    cap.release()
    root.destroy()


buttonQuit = ttk.Button(controlFrame, text='Quit', command=quitGui)
buttonQuit.grid(column=3, row=0)

# set video frame below controls
imageFrame = ttk.Frame(guiFrame, width=width, height=height)
imageFrame.grid(column=0, row=0)
l_img = ttk.Label(imageFrame)
l_img.grid(row=0, column=0)

# set footer frame below camera
footerFrame = ttk.Frame(guiFrame, padding=5)
footerFrame.grid(column=0, row=2)

shortcut_label = ttk.Label(footerFrame, text='Short-Cuts:           Ctrl + r (recalibrate/scale)             Ctrl + i (inverse controls)')
shortcut_label.grid(column=0, row=2, padx=0)

# def main():

current_cooldown_time = time.time()
previous_cooldown_time = time.time()

mouse = Controller()

EYELEVEL = [168, 6]
screen_normal = [-1, -2, -3, -4]

blinkCounter = 0

cap = cv.VideoCapture(0)
detector = itm.irisDetector()

def myloop():
    b1.set(len(fm.ReadLineWithKey("runtimesettings.txt", "Inversed").split()) > 1)

    global blinkCounter, previous_cooldown_time
    success, frame = cap.read()
    frame = detector.findIris(frame)
    current_cooldown_time = time.time()

    blinkDbl, blinkLeft, blinkRight = detector.detectBlink(frame)
    lmList = None
    try:
        lmList = detector.findPosition()
    except:
        return
    lmLeftEye = lmList[0]
    lmRightEye = lmList[1]

    cv.putText(frame, f'LeftEye:  [{lmLeftEye[0]}], [{lmLeftEye[1]}]',
               (10, 440), cv.FONT_HERSHEY_PLAIN,
               1, (255, 255, 255), 1)

    cv.putText(frame, f'RightEye: [{lmRightEye[0]}], [{lmRightEye[1]}]',
               (10, 460), cv.FONT_HERSHEY_PLAIN,
               1, (255, 255, 255), 1)

    cv.putText(frame, f"BlinkCounter: [{'infinite' if blinkCounter > 999 else blinkCounter}]", (450, 440), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv.putText(frame, f"Blink(both): [{blinkDbl}]", (450, 460), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv.putText(frame, f"Blink(Left):  [{blinkLeft}]", (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv.putText(frame, f"Blink(Right): [{blinkRight}]", (10, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv.line(frame, (320, 230), (320, 250), (0, 255, 0), 1)
    cv.line(frame, (310, 240), (330, 240), (0, 255, 0), 1)
    w, h, x, y = map(int, root.geometry().replace("x", " ").replace("+", " ").split())
    detector.moveCursor(frame, w, h, x, y)
    if blinkLeft and (current_cooldown_time - previous_cooldown_time) > 1:
        mouse.click(Button.right, 1)
        blinkCounter += 1
        previous_cooldown_time = time.time()

    if blinkRight and (current_cooldown_time - previous_cooldown_time) > 1:
        mouse.click(Button.left, 1)
        blinkCounter += 1
        previous_cooldown_time = time.time()

    if blinkDbl and (current_cooldown_time - previous_cooldown_time) > 1:
        mouse.click(Button.left, 2)
        blinkCounter += 1
        previous_cooldown_time = time.time()

    frameRGBA = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    img = Image.fromarray(frameRGBA)
    imgtk = ImageTk.PhotoImage(image=img)
    l_img.imgtk = imgtk
    l_img.configure(image=imgtk)
    l_img.after(10, myloop)

myloop()
root.protocol("WM_DELETE_WINDOW", quitGui)
root.mainloop()



