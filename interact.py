import cv2
import numpy as np
import pygame.display
from PIL import Image
import analyze
import preprocess
import torch
import time
import os
import ddsp
import random

def getCroppedImage(img):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    if faces != ():
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x-int(w/2), y-int(h/2)), (x+w+int(w/2), y+h+int(h/2)), (255, 0, 0), 2)
            X = max(x - int(w / 2), 0)
            Y = max(y - int(h / 2), 0)
            W = min(2 * w, img.shape[1] - X)
            H = min(2 * h, img.shape[1] - Y)
            face = img[Y:Y + H, X:X + W]
            resize_dim = (int(face.shape[1] * 218 / face.shape[0]), 218)
            resized = cv2.resize(face, resize_dim, interpolation=cv2.INTER_AREA)
            margin = int((resized.shape[1] - 178) / 2)
            cropped = resized[:, margin:margin + 178]
        return True, cropped
    else:
        return False, None


def capture():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAMWID)  # width=640
    cap.set(4, CAMHEI)  # height=480

    if cap.isOpened():
        _, frame = cap.read()
        cap.release()  # releasing camera immediately after capturing picture
        return frame


def croppedCapture():
    cont = True
    while (cont):
        c, x = getCroppedImage(capture())
        cont = not c
        if cont:
            time.sleep(0.5)
    return x


def CV2PIL(img):
    color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(color_converted)
    return pil


def getImage():
    return preprocess.vae_preProc(CV2PIL(croppedCapture()))


def postprocess(signal, smooth=False, window=11):
    signal = np.clip(signal, -1.0, 1.0)
    signal = (signal + 1.0) / 2.0

    MIDIBASE = 60.0 / ddsp.spectral_ops.F0_RANGE
    MIDIRANGE = 24.0 / ddsp.spectral_ops.F0_RANGE
    signal[0] = signal[0] * MIDIRANGE + MIDIBASE

    ld_BASE = 0.2
    ld_RANGE = 0.8
    signal[1] = ld_BASE + ld_RANGE * signal[1]
    if smooth:
        if window % 2 == 0:
            window = window + 1
        halfwindow = int((window - 1) / 2)
        for i in range(signal.shape[1]):
            bottom_index = max([0, i - halfwindow])
            top_index = min(i + halfwindow + 1, signal.shape[1])
            mu = np.mean(np.array(signal[:, bottom_index:top_index]), axis=1)
            signal[:, i] = mu

    return signal


def combine(vidname, audname):
    import moviepy.editor as mpe
    my_clip = mpe.VideoFileClip(vidname)
    audio_background = mpe.AudioFileClip(audname)
    clip = my_clip.set_audio(audio_background)
    clip.write_videofile(".\\output.mp4", fps=25)
    cv2.destroyAllWindows()
    clip.preview()
    clip.close()
    del clip
    del my_clip
    del audio_background
    pygame.display.quit()


def generate_output(features, timestepsPerFrame, instr, smooth=False):
    f = features.transpose(1, 0)
    f = postprocess(f, smooth)
    first = f[0]
    second = f[1]
    outfirst = []
    outsecond = []
    for i in range(len(first)):
        # if i < len(first)-1:
        #     stepfirst = first[i + 1] - first[i]
        #     stepsecond = second[i + 1] - second[i]
        #     stepfirst = stepfirst / float(timestepsPerFrame)
        #     stepsecond = stepsecond / float(timestepsPerFrame)
        # else:
        stepfirst = 0
        stepsecond = 0
        for j in range(int(timestepsPerFrame)):
            valfirst = first[i] + j * stepfirst
            valsecond = second[i] + j * stepsecond
            outfirst.append(valfirst)
            outsecond.append(valsecond)
    outt = np.array([outfirst, outsecond])
    print(outt.shape)

    signal = outt
    #
    np.save("signal.npy", signal)
    # CALL SCRIPT TO SAVE SOUND
    cmd = "python .\\face_decoder.py " + instr + " .\\signal.npy .\\output.wav"
    os.system(cmd)
    combine('.\\all.mp4', '.\\output.wav')


if __name__ == "__main__":
    # frame0 = None
    # frame0var = False
    if not os.path.exists(".\\fullPCA.npy") or not os.path.exists(".\\embeddings.npy"):
        os.system("python interact.py")
    Q = np.load(".\\fullPCA.npy")
    mean_embedding=np.load(".\\embeddings.npy")
    mean_embedding=np.mean(mean_embedding, axis=0)
    Qrand = np.random.randn(100,2)
    T_START = time.time()
    T_1024 = time.time()
    CAMWID = 480
    CAMHEI = 640
    SCALE = 1
    QSCALE = 3
    RECT_X = 200
    RECT_Y = 50
    inpW = 178
    inpH = 218
    instr = "violin"
    RECT_W = int(inpW * SCALE)
    RECT_H = int(inpH * SCALE)
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
    video_capture = None
    captured_features = []
    cap.set(3, CAMWID)  # width=640
    cap.set(4, CAMHEI)  # height=480
    cont = True
    show_rec = False
    show_plot = False
    smooth = False
    recording = False
    rand = False
    celeb=False
    celebFile="000001.jpg"
    model = analyze.model
    model.load_last_model(analyze.MODEL_PATH)
    framecountt=0
    frametimee=time.time()
    while (cont):
        if cap.isOpened():
            _, frame = cap.read()
            # if frame0var == False:
            #     frame0 = frame
            #     frame0var = True
            mug = frame[RECT_Y:+RECT_Y + RECT_H, RECT_X:RECT_X + RECT_W].copy()
            mug = cv2.resize(mug, (inpW, inpH))
            if celeb:
                mug=cv2.imread('.\\img_align_celeba\\'+celebFile)

            cv2.rectangle(frame, (RECT_X, RECT_Y), (RECT_X + RECT_W, RECT_Y + RECT_H), (255, 0, 0), 3)
            z = analyze.get_mean(preprocess.vae_preProc(CV2PIL(mug)), model, device=analyze.device)
            z=z-mean_embedding
            znp=np.squeeze(z.numpy())
            if rand:
                x = np.matmul(znp, Qrand) / QSCALE
            else:
                x = np.matmul(znp, Q) / QSCALE
            if recording:
                captured_features.append(x)

            with torch.no_grad():
                out = model.decode(z).cpu().numpy()

            R = inpH
            twoR = 2 * R
            HALFPAD = 20
            PAD = 2 * HALFPAD

            rec = out[0].transpose(1, 2, 0)
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            mug = cv2.resize(mug, (2 * inpW, 2 * inpH), interpolation=cv2.INTER_AREA)

            rec = cv2.resize(rec, (2 * inpW, 2 * inpH), interpolation=cv2.INTER_AREA)
            outt = rec
            plot = np.zeros(shape=(twoR, twoR, 3))
            cv2.circle(plot, (R, R), R, (0.8, 0, 0), 1)
            cv2.line(plot, (0, R), (twoR, R), (0.8, 0, 0), 1)
            cv2.line(plot, (R, 0), (R, twoR), (0.8, 0, 0), 1)
            if x[0] > 1:
                x[0] = 1
            elif x[0] < -1:
                x[0] = -1
            if x[1] > 1:
                x[1] = 1
            elif x[1] < -1:
                x[1] = -1
            y = np.array(x)
            y[1] = -y[1]
            dotCoor = inpH * y + inpH
            plot = np.concatenate((np.zeros(shape=(twoR, PAD, 3)), plot, np.zeros(shape=(twoR, PAD, 3))), axis=1)
            plot = np.concatenate(
                (np.zeros(shape=(PAD, plot.shape[1], 3)), plot, np.zeros(shape=(PAD, plot.shape[1], 3))), axis=0)
            mug = np.concatenate(
                (np.zeros(shape=(PAD, mug.shape[1], 3)), mug, np.zeros(shape=(PAD, mug.shape[1], 3))), axis=0)
            rec = np.concatenate(
                (np.zeros(shape=(PAD, rec.shape[1], 3)), rec, np.zeros(shape=(PAD, rec.shape[1], 3))), axis=0)
            # plot = np.concatenate((np.zeros(shape=(PAD, twoR, 3)), plot, np.zeros(shape=(PAD, twoR, 3))), axis=1)
            color=(0.0,0.0,0.0)
            if rand:
                color=(0.0,1.0,1.0)
            else:
                color=(0.0,0.0,1.0)
            cv2.circle(plot, (PAD + int(dotCoor[0]), PAD + int(dotCoor[1])), 5, color, -1)
            out = mug / 255.0
            pres = mug / 255.0
            out = np.concatenate((out, rec), axis=1)
            out = np.concatenate((out, plot), axis=1)
            if show_rec:
                pres = np.concatenate((pres, rec), axis=1)
            if show_plot:
                pres = np.concatenate((pres, plot), axis=1)

            if recording:
                if video_writer is not None:
                    # record video
                    data = np.asarray(out * 255, dtype=np.uint8)
                    t=time.time()
                    video_writer.write(data)
                    if t<frametimee+1:
                        framecountt=framecountt+1
                    else:
                        print(framecountt)
                        frametimee=t
                        framecountt=0
                key = cv2.waitKey(2)
            else:
                key = cv2.waitKey(10)
            if recording:
                cv2.circle(pres, (HALFPAD, HALFPAD), 10, (1.0, 0.0, 0.0), -1)
            # cv2.imshow("mug", mug)
            cv2.imshow("Instrumental", pres)

            # cv2.imshow("plot", plot)
            # cv2.imshow("rec", rec)
            # cv2.imshow("full frame", frame)
            if key == ord("q"):
                if not recording:
                    cap.release()
            elif key == ord("r"):
                if not recording:
                    # video recorder
                    video_writer = cv2.VideoWriter("all.mp4", fourcc, 25, (out.shape[1], out.shape[0]))
                else:
                    video_writer.release()
                    video_writer = None
                    generate_output(np.array(captured_features), int(16000 / (25 * 64)), instr, smooth)
                    captured_features = []
                recording = not recording
                print(recording)
            elif key == ord("p"):
                show_plot = not show_plot
            elif key == ord("o"):
                show_rec = not show_rec
            elif key == ord("a"):
                RECT_X = RECT_X - 2
            elif key == ord("d"):
                RECT_X = RECT_X + 2
            elif key == ord("w"):
                RECT_Y = RECT_Y - 2
            elif key == ord("s"):
                RECT_Y = RECT_Y + 2
            elif key == ord("x"):
                SCALE = SCALE + 0.1
                if SCALE > 3:
                    SCALE = 3
                RECT_W = int(178 * SCALE)
                RECT_H = int(218 * SCALE)
            elif key == ord("z"):
                SCALE = SCALE - 0.1
                if SCALE < 0.5:
                    SCALE = 0.5
                RECT_W = int(178 * SCALE)
                RECT_H = int(218 * SCALE)
            elif key == ord("v"):
                if QSCALE > 0.1:
                    QSCALE = QSCALE - 0.1
                else:
                    QSCALE = QSCALE / 2.0
                    if QSCALE < 0.0001:
                        QSCALE = 0.0001
            elif key==ord("b"):
                #if recording=False:
                celeb=not celeb
            elif key==ord("n"):
                r=np.random.randint(202000)
                d=int(np.log10(r))
                celebFile=""
                for i in range(5-d):
                    celebFile+="0"
                celebFile=celebFile+str(r)+".jpg"
            elif key == ord("c"):
                if QSCALE < 0.1:
                    QSCALE = QSCALE * 2.0
                else:
                    QSCALE = QSCALE + 0.1

            elif key == ord("e"):
                if not recording:
                    smooth = not smooth
                    if smooth:
                        print("Smooth On")
                    else:
                        print("Smooth Off")
            elif key == ord("h"):
                if not recording:
                    instr = "sax"
                    print("Instrument: " + instr)
            elif key == ord("g"):
                if not recording:
                    instr = "violin"
                    print("Instrument: " + instr)
            elif key == ord("t"):
                if not recording:
                    instr = "trumpet"
                    print("Instrument: " + instr)
            elif key == ord("f"):
                if not recording:
                    instr = "flute"
                    print("Instrument: " + instr)
            elif key == ord(" "):
                if not recording:
                    rand=not rand
                    if rand:
                        Qrand=np.random.randn(100,2)
                        print("Random projection")
                    else:
                        print("PCA projection")
        else:
            cont = False
