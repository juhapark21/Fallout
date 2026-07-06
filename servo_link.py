# Serial communication with XIAO RP2040 over USB-C 

# ls /dev/tty.usb* or python servo_link.py --list 

import sys
import time

import serial
import serial.tools.list_ports


def list_ports():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        print(f"{p.device} - {p.description}")


class ServoLink:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # RP2040 resets on serial connect; give it time to boot

    def send(self, command: str):
        # Send a command line, e.g. 'ADJ:15', 'SET:90', 'AUTO:1' 
        line = command.strip() + "\n"
        self.ser.write(line.encode("utf-8"))

    def adjust(self, delta_degrees: int):
        self.send(f"ADJ:{delta_degrees}")

    def set_angle(self, angle_degrees: int):
        self.send(f"SET:{angle_degrees}")

    def set_auto_mode(self, enabled: bool):
        self.send(f"AUTO:{1 if enabled else 0}")

    def fire(self, enabled: bool): 
        self.send("FIRE") 

    def read_line(self):
        # Read any response/debug line the board sends back 
        if self.ser.in_waiting:
            return self.ser.readline().decode("utf-8", errors="ignore").strip()
        return None

    def close(self):
        self.ser.close()


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_ports()
        sys.exit(0)

    # Update this to match your board's port
    PORT = "/dev/tty.usbmodem1234"  

    link = ServoLink(PORT)
    print("Connected. Sending test commands...")

    link.set_auto_mode(True)
    time.sleep(0.5)
    link.set_angle(90)
    time.sleep(1)
    link.adjust(15)

    # Print anything the board sends back 
    time.sleep(0.5)
    while True:
        line = link.read_line()
        if line:
            print(f"Board: {line}")
        else:
            break

    link.close()
