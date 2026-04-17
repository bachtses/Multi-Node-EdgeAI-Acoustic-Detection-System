import serial
import time
import re
import threading  

class LoRaModule:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._io_lock = threading.Lock()   

        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"\033[96m[LoRa] Serial port {port} opened.\033[0m")
        except Exception as e:
            print(f"\033[91m[LoRa INIT ERROR] Could not open serial port: {e}\033[0m")
            self.ser = None

    def send_command(self, command, delay=0.2):
        if self.ser is None:
            return ""
        if not self.ser.is_open:
            self.ser.open()

        with self._io_lock:                    
            self.ser.write((command + '\r\n').encode())

        time.sleep(delay)
        return self.ser.read_all().decode(errors='ignore').strip()

    def enter_at_mode(self):
        if self.ser is None:
            return ""
        with self._io_lock:                      
            self.ser.write(b'+++')
        time.sleep(1)
        return self.ser.read_all().decode(errors='ignore').strip()

    def exit_at_mode(self):
        return self.send_command('AT+EXIT')

    def setup_module(self):
        print("\033[96m[LoRa] Configuring module...\033[0m")
        self.enter_at_mode()
        self.send_command('AT+MODE=0')         # 0 Packet mode / 1 Stream mode
        self.send_command('AT+SF=7')           # Spreading factor
        self.send_command('AT+BW=0')           # Bandwidth: 125 kHz
        self.send_command('AT+CR=1')           # Coding rate: 4/5
        self.send_command('AT+PWR=22')         # Max power
        self.send_command('AT+NETID=0')
        self.send_command('AT+TXCH=29')
        self.send_command('AT+RXCH=29')
        self.send_command('AT+BAUD=115200')
        self.send_command('AT+COMM="8N1"')
        self.send_command('AT+RSSI=1')    
        self.exit_at_mode()
        print("\033[96m[LoRa] Module configured successfully.\033[0m")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def is_open(self):
        return self.ser and self.ser.is_open

    def flush(self):
        if self.ser:
            self.ser.flushInput()
            self.ser.flushOutput()


class LoRaTransmitter(LoRaModule):
    def send_message(self, message):
        if self.ser is None:
            print("\033[91m[LoRa TX ERROR] Serial port not initialized.\033[0m")
            return
        try:
            with self._io_lock:
                self.ser.write((message + '\n').encode())
                self.ser.flush()
        except Exception as e:
            print(f"\033[91m[LoRa TX ERROR] {e}\033[0m")


class LoRaReceiver(LoRaModule):
    def setup_module(self):
        print("\033[96m[LoRa] Configuring module...\033[0m")
        self.enter_at_mode()
        self.send_command('AT+MODE=0')
        self.send_command('AT+RSSI=1')
        self.exit_at_mode()
        print("\033[96m[LoRa] Module configured successfully.\033[0m")

    def listen(self, stop_event=None):
        if self.ser is None:
            print("\033[91m[LoRa RX ERROR] Serial port not initialized.\033[0m")
            return
        print("\033[96m[LoRa RX] Listening for messages...\033[0m")
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                line = self.ser.readline().decode(errors='ignore').strip()
                if line:
                    yield line
                else:
                    time.sleep(0.01)
        finally:
            self.close()
