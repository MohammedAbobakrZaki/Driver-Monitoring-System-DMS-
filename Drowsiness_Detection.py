import spidev  # SPI communication
import time  # Used for timing and delays
import cv2  # OpenCV library for computer vision
import dlib  # Used for face detection and landmark prediction
from imutils import face_utils, resize  # Used for facial landmark processing and image resizing
from scipy.spatial import distance  # Used for calculating distances
import RPi.GPIO as GPIO  # GPIO library for Raspberry Pi

# SPI Bus and CE pin selection
spi_bus = 0
spi_device = 0

# SPI activating
spi = spidev.SpiDev()
spi.open(spi_bus, spi_device)
spi.max_speed_hz = 1000000  # Transmitting speed is 1Mbps

# Define GPIO to LCD mapping
LCD_RS = 7
LCD_E = 8
LCD_D4 = 25
LCD_D5 = 24
LCD_D6 = 23
LCD_D7 = 18

# Define some device constants
LCD_WIDTH = 16  # Maximum characters per line
LCD_CHR = True
LCD_CMD = False

LCD_LINE_1 = 0x80  # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0  # LCD RAM address for the 2nd line

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005

# LED control (using GPIO pin 21 for Buzzer and GPIO pin 16 for LED)
LED_Pin = 21
LED_Pin_2 = 16

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_Pin, GPIO.OUT)
GPIO.setup(LED_Pin_2, GPIO.OUT)

# Turn OFF all output pins initially
GPIO.output(LED_Pin, GPIO.LOW)
GPIO.output(LED_Pin_2, GPIO.LOW)


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # Distance between landmark points
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)  # EAR formula
    return ear


def lcd_init():
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    time.sleep(E_DELAY)


def lcd_byte(bits, mode):
    GPIO.output(LCD_RS, mode)
    for pin, value in zip([LCD_D4, LCD_D5, LCD_D6, LCD_D7], [bits & 0x10, bits & 0x20, bits & 0x40, bits & 0x80]):
        GPIO.output(pin, bool(value))
    lcd_toggle_enable()
    for pin, value in zip([LCD_D4, LCD_D5, LCD_D6, LCD_D7], [bits & 0x01, bits & 0x02, bits & 0x04, bits & 0x08]):
        GPIO.output(pin, bool(value))
    lcd_toggle_enable()


def lcd_toggle_enable():
    time.sleep(E_DELAY)
    GPIO.output(LCD_E, True)
    time.sleep(E_PULSE)
    GPIO.output(LCD_E, False)
    time.sleep(E_DELAY)


def lcd_string(message, line):
    message = message.ljust(LCD_WIDTH, " ")
    lcd_byte(line, LCD_CMD)
    for char in message:
        lcd_byte(ord(char), LCD_CHR)


def update_lcd_and_gpio(state):
    lcd_init()
    if state == "sleep":
        lcd_string("Sleep! wake up", LCD_LINE_1)
        GPIO.output(LED_Pin, GPIO.LOW)
        GPIO.output(LED_Pin_2, GPIO.LOW)
        spi.xfer2([0x01])
    else:
        lcd_string("Be Careful!", LCD_LINE_1)
        GPIO.output(LED_Pin, GPIO.HIGH)
        GPIO.output(LED_Pin_2, GPIO.HIGH)
        spi.xfer2([0x00])


# Drowsiness detection parameters
thresh = 0.20  # Eye aspect ratio threshold for drowsiness detection
frame_check = 20  # Number of consecutive frames below threshold for alert

# Face detection and landmark prediction models
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("/home/rasp/Final/shape_predictor_68_face_landmarks.dat")

# Eye landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

flag = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from camera")
            break

        frame = resize(frame, width=480)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = face_utils.shape_to_np(predict(gray, subject))
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    update_lcd_and_gpio("sleep")
            else:
                flag = 0
                update_lcd_and_gpio("awake")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Ensure all GPIO pins are turned off
    GPIO.output(LED_Pin, GPIO.LOW)
    GPIO.output(LED_Pin_2, GPIO.LOW)
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
