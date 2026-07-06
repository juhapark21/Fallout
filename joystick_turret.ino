#include <Servo.h>

Servo verticalServo;
Servo horizontalServo; 
Servo launchServo; 

const int yPin = A1;   // VRx
const int xPin = A2; // VRy 
//int servoYAngle = 90;     // start centered
//int servoXAngle = 90; 

const int launchButtonPin = D10; 
int buttonState = 0; 
int lastButtonState = HIGH; 

// Auto-aiming and shoot command receiving 
bool firingNow = false; 
bool doFire = false; 
unsigned long fireEndTime = 0; 
int adjX = 0; 
int adjY = 0; 

String inputBuffer = "";

int standbyLaunchAngle = 1500;
int fireLaunchAngle = 1000;

void setup() {

  verticalServo.attach(D9);   // GPIO pin connected to the signal wire
  verticalServo.writeMicroseconds(1500);
  horizontalServo.attach(D5); 
  horizontalServo.writeMicroseconds(1500); 

  launchServo.attach(D6); 
  launchServo.writeMicroseconds(standbyLaunchAngle);

  pinMode(launchButtonPin, INPUT_PULLUP);

  Serial.begin(115200);
  Serial.println("Test test test");
}

void loop() {
  readSerialCommands();
  buttonState = digitalRead(launchButtonPin);
  if (buttonState == LOW && lastButtonState == HIGH) {
    // Fire 
    fire(); 
  } 
  lastButtonState = buttonState; 

  if (firingNow && millis() > fireEndTime) {
    firingNow = false; 
    // Reset to ready position 
    launchServo.writeMicroseconds(standbyLaunchAngle);
  } 

  if (!firingNow) {
    int raw = analogRead(yPin);        // 0–1023 by default on RP2040
    int angle = map(raw, 0, 1023, 0, 90); // 0
    int ms = map(angle, 0, 90, 1000, 2000); // for writems

    int rawX = analogRead(xPin); 
    int angleX = map(rawX, 0, 1023, 0, 90); 
    int msX = map(angleX, 0, 90, 1000, 2000); 

    verticalServo.writeMicroseconds(ms); 
    horizontalServo.writeMicroseconds(msX); 

    launchServo.writeMicroseconds(standbyLaunchAngle);
    
    Serial.println(angle);               // handy for debugging in Serial Monitor
    Serial.println(angleX);
  } 

  delay(20);
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line.indexOf("FIRE") != -1) {
    // Fire 
    fire(); 
  } else if (line.indexOf("ADJ") != -1) {
    // Adjust angle (according to what is sent via serial)
    Serial.println("Adjust");
  } else {
    return; 
  }

}

void fire() {
  // Fire 
  Serial.println("Fire!!");
  firingNow = true; 
  launchServo.write(fireLaunchAngle);
  fireEndTime = millis() + 300; 
}
