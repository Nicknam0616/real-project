import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5 import uic
import openai
from deep_translator import GoogleTranslator
import cv2
from PyQt5.QtGui import QImage, QPixmap
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import socketio
import mediapipe as mp
import numpy as np
import serial
from geopy.distance import geodesic
import time

# Flask 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio_server = SocketIO(app, async_mode='threading')

@app.route('/')
def home():
    return render_template('index.html')

@socketio_server.on('message')
def handleMessage(msg):
    print('Message:', msg['msg'])
    emit('message', {'user': msg['user'], 'msg': msg['msg']}, broadcast=True)

# UI 파일 연결
form_class = uic.loadUiType("real.ui")[0]

class AIThread(QThread):
    response_signal = pyqtSignal(str, str)

    def __init__(self, user_input, client, translator, parent=None):
        super().__init__(parent)
        self.user_input = user_input
        self.client = client
        self.translator = translator

    def run(self):
        print("AIThread started")
        try:
            translated_input = self.translator.translate(self.user_input)
            completion = self.client.chat.completions.create(
                model="hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF/llama-3.2-3b-instruct-q8_0.gguf",
                messages=[
                    {"role": "system", "content": "You're an expert in strategy and tactics, help me ..."},
                    {"role": "user", "content": translated_input}
                ],
                temperature=0.7,
            )

            # 번역 결과 처리
            translated_output = GoogleTranslator(source='en', target='ko').translate(
                completion.choices[0].message.content)

            # 결과 신호 전송
            self.response_signal.emit(self.user_input, translated_output)

        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")  # 구체적인 오류 메시지 출력
        except Exception as e:
            print(f"AIThread error: {e}")

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, mode="basic"):
        super().__init__()
        self.mode = mode
        self.running = True
        self.cap = cv2.VideoCapture(0)  # 비디오 캡처 객체 초기화
        #self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.2)

    def run(self):
        print("CameraThread started")
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # 프레임 크기 조정
                frame = cv2.resize(frame, (640, 480))
                rgb_image = self.process_frame(frame)

                # QImage로 변환
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # 신호를 통해 UI 업데이트
                self.change_pixmap_signal.emit(qt_image)

    def stop(self):
        print("Stopping CameraThread")
        self.running = False
        self.cap.release()  # 비디오 캡처 객체 해제
        cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

    def process_frame(self, frame):
        if self.mode == "basic":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.mode == "black_white":
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        elif self.mode == "red_only":
            return self.red_only_detection(frame)
        #elif self.mode == "face_tracking":
        #    return self.face_tracking(frame)
        return frame

    def red_only_detection(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 90])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result

    #def face_tracking(self, frame):
    #    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #    results = self.face_detection.process(rgb_image)
    #    if results.detections:
    #        for detection in results.detections:
    #            bbox = detection.location_data.relative_bounding_box
    #            ih, iw, _ = frame.shape
    #            x, y, w, h = (int(bbox.xmin * iw), int(bbox.ymin * ih),
    #                          int(bbox.width * iw), int(bbox.height * ih))
    #            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def change_mode(self, mode):
        print(f"Changing camera mode to: {mode}")
        self.mode = mode

class WindowClass(QMainWindow, form_class):
    incoming_packet_signal = pyqtSignal(str)
    outgoing_packet_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setupUi(self)  # UI 설정
        self.client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.translator = GoogleTranslator(source='ko', target='en')

        # 시리얼 포트 자동 설정
        port_number = 'COM13'  # 원하는 포트 번호로 수정하세요
        self.bluetooth = self.connect_serial_port(port_number)
        if not self.bluetooth:
            QMessageBox.warning(self, "Port Error", f"Cannot find port {port_number}.")
            return

        # CameraThread 인스턴스 생성
        self.camera_thread = CameraThread("basic")
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.start()

        # 시그널 연결
        self.incoming_packet_signal.connect(self.append_incoming_packet)
        self.outgoing_packet_signal.connect(self.append_outgoing_packet)

        self.key_status = {
            'W': False,
            'A': False,
            'S': False,
            'D': False
        }

        # UI 요소 연결
        self.gpt_button.clicked.connect(self.start_ai_thread)
        self.chat_b.clicked.connect(self.send_message)
        self.gpt_real.setReadOnly(True)

        # UI 설정
        self.fire_status = False  # 발사 상태 초기화
        self.fire_button.clicked.connect(self.fire)

        self.setup_readonly_fields()

        # 속도 슬라이더 설정
        self.setup_speed_slider()

        # 카메라 모드 변경 버튼 연결
        self.setup_camera_mode_buttons()

        # 목표 경도 입력 필드
        self.setup_navigation_buttons()

        # 타이머 설정
        self.movement_timer = QTimer(self)
        self.movement_timer.timeout.connect(self.send_movement_commands)
        self.movement_timer.start(100)  # 100ms마다 패킷 전송

        # 소켓.io 클라이언트 초기화
        self.socket = socketio.Client()
        self.connect_socket()

    def connect_serial_port(self, port_name):
        print("Connecting to serial port...")
        try:
            return serial.Serial(port_name, 9600, timeout=1)  # 포트 번호를 직접 지정
        except serial.SerialException:
            print(f"{port_name} 포트를 찾을 수 없습니다.")
            return None

    def update_image(self, qt_image):
        print("Updating image in UI.")
        self.main.setPixmap(QPixmap.fromImage(qt_image))

    def keyPressEvent(self, event):
        print(f"Key pressed: {event.key()}")
        if event.key() == Qt.Key_W:
            self.key_status['W'] = True
        elif event.key() == Qt.Key_S:
            self.key_status['S'] = True
        elif event.key() == Qt.Key_A:
            self.key_status['A'] = True
        elif event.key() == Qt.Key_D:
            self.key_status['D'] = True

    def keyReleaseEvent(self, event):
        print(f"Key released: {event.key()}")
        if event.key() == Qt.Key_W:
            self.key_status['W'] = False
        elif event.key() == Qt.Key_S:
            self.key_status['S'] = False
        elif event.key() == Qt.Key_A:
            self.key_status['A'] = False
        elif event.key() == Qt.Key_D:
            self.key_status['D'] = False

    def send_movement_commands(self):
        print("Sending movement commands...")
        fire_status = 1 if self.fire_status else 0  # 발사 버튼 상태에 따라 1 또는 0 설정
        if self.key_status['W']:
            packet = f'1,{fire_status}\n'
        elif self.key_status['A']:
            packet = f'2,{fire_status}\n'
        elif self.key_status['S']:
            packet = f'3,{fire_status}\n'
        elif self.key_status['D']:
            packet = f'4,{fire_status}\n'
        else:
            packet = f'0,{fire_status}\n'  # 아무 키도 눌리지 않을 때

        self.bluetooth.write(packet.encode())  # 패킷 전송
        self.outgoing_packet_signal.emit(packet)  # 나가는 패킷 추가

    def fire(self):
        print("Fire command issued.")
        self.fire_status = True  # 발사 상태를 1로 설정
        QTimer.singleShot(100, self.reset_fire_status)  # 100ms 후 발사 상태를 초기화

    def reset_fire_status(self):
        print("Resetting fire status.")
        self.fire_status = False  # 발사 상태를 0으로 설정

    def update_current_location(self):
        print("Updating current location...")
        current_coords = self.get_current_location()
        if current_coords:
            self.line2.setText(f"현재 위치: {current_coords[0]}, {current_coords[1]}")
            # 나침반 값 업데이트
            compass_value = self.get_compass_value()
            self.line4.setText(f"Compass: {compass_value}")

            # 목표 위치와 거리 계산
            target_coords = self.get_target_location()
            if target_coords:
                distance = geodesic(current_coords, target_coords).meters
                self.line3.setText(f"남은 거리: {distance:.2f} m")
                # 자율주행 신호 전송
                self.send_navigation_command(current_coords, target_coords)

    def start_navigation(self):
        print("Starting navigation...")
        target_location = self.line1.toPlainText().strip()
        if not target_location:
            QMessageBox.warning(self, "입력 오류", "목표 위도와 경도를 입력하세요.")
            return

        try:
            target_lat, target_lon = map(float, target_location.split(','))
            self.bluetooth.write(f'G,{target_lat},{target_lon},\n'.encode())
            self.line1.append(f"목표 위치 설정: {target_location}")
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "올바른 형식(예: 37.395825, 127.045556)으로 입력하세요.")

    def move_and_return(self):
        print("Moving and returning...")
        target_location = self.line1.toPlainText().strip()
        if not target_location:
            QMessageBox.warning(self, "입력 오류", "목표 위도, 경도를 입력하세요.")
            return

        try:
            target_lat, target_lon = map(float, target_location.split(','))
            self.bluetooth.write(f'R,{target_lat},{target_lon},\n'.encode())
            self.line1.append(f"복귀 명령 전송: {target_location}")
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "올바른 형식(예: 37.395825, 127.045556)으로 입력하세요.")

    def stop_and_clear(self):
        print("Stopping and clearing...")
        self.bluetooth.write('S,\n'.encode())
        self.line1.clear()

    def get_current_location(self):
        print("Getting current location...")
        while self.bluetooth.in_waiting > 0:
            line = self.bluetooth.readline().decode('utf-8').strip()
            if line.startswith("Lat/Long:"):
                parts = line.split()
                latitude = float(parts[1].strip(','))
                longitude = float(parts[2])
                self.append_incoming_packet(f"현재 위치: {latitude}, {longitude}")  # 들어오는 데이터 추가
                return (latitude, longitude)
        return None

    def get_target_location(self):
        print("Getting target location...")
        target_location = self.line1.toPlainText().strip()
        if target_location:
            parts = target_location.split(',')
            return (float(parts[0]), float(parts[1]))
        return None

    def get_compass_value(self):
        print("Getting compass value...")
        while self.bluetooth.in_waiting > 0:
            line = self.bluetooth.readline().decode('utf-8').strip()
            if line.startswith("Compass:"):
                compass_value = float(line.split(":")[1].strip())
                self.append_incoming_packet(f"Compass: {compass_value}")  # 들어오는 데이터 추가
                return compass_value
        return 0

    def get_temperature(self):
        print("Getting temperature...")
        while self.bluetooth.in_waiting > 0:
            line = self.bluetooth.readline().decode('utf-8').strip()
            if line.startswith("Temperature:"):
                temperature = float(line.split(":")[1].strip())
                self.append_incoming_packet(f"온도 값: {temperature}°C")  # 들어오는 데이터 추가
                return temperature
        return None

    def get_humidity(self):
        print("Getting humidity...")
        while self.bluetooth.in_waiting > 0:
            line = self.bluetooth.readline().decode('utf-8').strip()
            if line.startswith("Humidity:"):
                humidity = float(line.split(":")[1].strip())
                self.append_incoming_packet(f"습도 값: {humidity}%")  # 들어오는 데이터 추가
                return humidity
        return None

    def send_navigation_command(self, current_coords, target_coords):
        print("Sending navigation command...")
        target_lat, target_lon = target_coords
        if current_coords[0] < target_lat:
            self.bluetooth.write('auto front,\n'.encode())
            self.append_incoming_packet("자율주행: 앞으로 이동")  # 들어오는 데이터 추가
        elif current_coords[0] > target_lat:
            self.bluetooth.write('auto back,\n'.encode())
            self.append_incoming_packet("자율주행: 뒤로 이동")  # 들어오는 데이터 추가

        if current_coords[1] < target_lon:
            self.bluetooth.write('auto right,\n'.encode())
            self.append_incoming_packet("자율주행: 오른쪽으로 이동")  # 들어오는 데이터 추가
        elif current_coords[1] > target_lon:
            self.bluetooth.write('auto left,\n'.encode())
            self.append_incoming_packet("자율주행: 왼쪽으로 이동")  # 들어오는 데이터 추가

    def red_only_mode(self):
        print("Changing to red-only mode.")
        self.camera_thread.change_mode("red_only")

    def start_ai_thread(self):
        print("Starting AI thread.")
        user_input = self.gpt_input.text()
        if user_input:
            self.ai_thread = AIThread(user_input, self.client, self.translator)
            self.ai_thread.response_signal.connect(self.update_chat)
            self.ai_thread.start()

    def update_speed_set(self):
        speed_value = self.speed_slider.value()
        self.speed_set.setText(f"{speed_value}")
        packet = f'speed_set,{speed_value},\n'
        self.bluetooth.write(packet.encode())
        self.append_outgoing_packet(packet)

    def send_message(self):
        print("Sending message...")
        message = self.chat_i.text()
        if message:
            self.socket.emit('message', {'user': 'ui', 'msg': message})
            self.append_outgoing_packet(message)  # 나가는 패킷 추가
            self.chat_i.clear()

    def receive_message(self, data):
        print("Receiving message...")
        self.chat.append(f"{data['user']}: {data['msg']}")
        self.append_incoming_packet(data['msg'])  # 들어오는 패킷 추가

    def update_chat(self, user_input, translated_output):
        print("Updating chat...")
        self.gpt_real.append(f"User: {user_input}")
        self.gpt_real.append(f"AI: {translated_output}")
        self.gpt_input.clear()

    def update_line(self, text):
        print("Updating line...")
        self.line1.append(text)

    def change_camera_mode(self, mode):
        print(f"Changing camera mode to: {mode}")
        self.camera_thread.change_mode(mode)

    # UI 요소를 읽기 전용으로 설정
    def setup_readonly_fields(self):
        self.under1.setReadOnly(True)
        self.under2.setReadOnly(True)
        self.under3.setReadOnly(True)
        self.under4.setReadOnly(True)
        self.under5.setReadOnly(True)
        self.under6.setReadOnly(True)
        self.main_under.setReadOnly(True)
        self.chat.setReadOnly(True)
        self.line2.setReadOnly(True)
        self.line3.setReadOnly(True)
        self.line4.setReadOnly(True)
        self.line5.setReadOnly(True)
        self.line6.setReadOnly(True)
        self.line7.setReadOnly(True)
        self.line8.setReadOnly(True)
        self.line9.setReadOnly(True)
        self.speed_set.setReadOnly(True)
        self.speed_real.setReadOnly(True)
        self.temperature.setReadOnly(True)
        self.humidity.setReadOnly(True)

        self.under3.setStyleSheet("background-color: lightyellow;")  # 배경색 설정

    # 속도 슬라이더 설정
    def setup_speed_slider(self):
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setSingleStep(1)
        self.speed_slider.setPageStep(10)
        self.speed_slider.setValue(0)
        self.speed_slider.setTracking(True)
        self.speed_slider.valueChanged.connect(self.update_speed_set)

    # 카메라 모드 변경 버튼 연결
    def setup_camera_mode_buttons(self):
        self.b_c.clicked.connect(lambda: self.change_camera_mode("basic"))
        self.b_d.clicked.connect(lambda: self.change_camera_mode("black_white"))
        self.b_r.clicked.connect(self.red_only_mode)
        #self.b_f.clicked.connect(lambda: self.change_camera_mode("face_tracking"))

    # 목표 경도 입력 필드
    def setup_navigation_buttons(self):
        self.b1.clicked.connect(self.start_navigation)
        self.b2.clicked.connect(self.move_and_return)
        self.b3.clicked.connect(self.stop_and_clear)

    # 소켓 연결 및 예외 처리
    def connect_socket(self):
        try:
            self.socket.connect('http://localhost:5000', namespaces=['/'])  # 특정 네임스페이스를 지정할 수 있습니다.
            self.socket.on('message', self.receive_message)  # 소켓 메시지 연결
        except socketio.exceptions.ConnectionError as e:
            QMessageBox.warning(self, "서버 연결 오류", f"Flask 서버에 연결할 수 없습니다: {e}")
#
    def append_incoming_packet(self, packet):
        """패킷을 UI에 추가하는 메서드"""
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M")
        self.line1.append(f"-> {current_time} | {packet}")  # 또는 적절한 UI 요소에 추가

    def append_outgoing_packet(self, packet):
        """나가는 패킷을 UI에 추가하는 메서드"""
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M")
        packet_display = packet.replace("\n", "\\n")  # \n을 \\n으로 표시
        self.line1.append(f"<- {current_time} | {packet_display}")  # 또는 적절한 UI 요소에 추가

if __name__ == "__main__":
    import threading

    # Flask 서버를 별도의 스레드에서 실행
    flask_thread = threading.Thread(target=lambda: socketio_server.run(app, host='0.0.0.0', allow_unsafe_werkzeug=True))
    flask_thread.start()

    # QApplication 인스턴스 생성
    app = QApplication(sys.argv)

    # WindowClass의 인스턴스 생성
    myWindow = WindowClass()

    # 프로그램 화면을 보여주는 코드
    myWindow.show()

    # 프로그램을 이벤트 루프로 진입시키는 (프로그램을 작동시키는) 코드
    sys.exit(app.exec_())