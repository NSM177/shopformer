/*
 * Firmware điều khiển Servo qua cổng USB Serial cho ESP32 và Arduino
 * 
 * Sơ đồ đấu nối Servo (SG90 / MG996R):
 *   - Dây Vàng / Cam (Tín hiệu): Chân GPIO 18 (ESP32) hoặc Pin 9 (Arduino Uno/Nano)
 *   - Dây Đỏ (Nguồn VCC): Chân 5V (khuyến nghị nguồn 5V riêng nếu dùng MG996R)
 *   - Dây Nâu / Đen (GND): Chân GND của bo mạch (và nối chung GND nguồn ngoài)
 * 
 * Giao thức giao tiếp qua USB Serial (Baudrate 115200):
 *   - Gửi: "UNLOCK" -> Servo quay về 0 độ (Mở khóa)
 *   - Gửi: "LOCK" hoặc "LOCK:45" -> Servo quay 90 độ (Khóa trong 45 giây)
 *   - Gửi: "ANGLE:xx" -> Servo quay đến góc xx (0 đến 180 độ)
 *   - Gửi: "STATUS" -> Trả về JSON trạng thái hiện tại
 * 
 * An toàn (Fail-safe):
 *   - Khi mới cắm điện: Luôn quay về góc 0 độ (MỞ).
 *   - Hết thời gian TTL khóa: Tự động quay về góc 0 độ.
 */

#if defined(ESP32)
  #include <ESP32Servo.h>
  #define SERVO_PIN 18
#else
  #include <Servo.h>
  #define SERVO_PIN 9
#endif

Servo doorServo;

const int ANGLE_UNLOCKED = 0;   // 0 độ: Mở cửa
const int ANGLE_LOCKED   = 90;  // 90 độ: Khóa cửa

bool isLocked = false;
int currentAngle = 0;
unsigned long lockDeadlineMs = 0;

void setServoAngle(int angle) {
  if (angle < 0) angle = 0;
  if (angle > 180) angle = 180;
  currentAngle = angle;
  doorServo.write(angle);
}

void printStatus() {
  Serial.print("{\"state\":\"");
  Serial.print(isLocked ? "locked" : "unlocked");
  Serial.print("\",\"angle\":");
  Serial.print(currentAngle);
  Serial.print(",\"ttl_remaining_sec\":");
  if (isLocked && lockDeadlineMs > millis()) {
    Serial.print((lockDeadlineMs - millis()) / 1000);
  } else {
    Serial.print(0);
  }
  Serial.println("}");
}

void unlockDoor(const char* reason) {
  isLocked = false;
  lockDeadlineMs = 0;
  setServoAngle(ANGLE_UNLOCKED);
  Serial.print("[SERVO] UNLOCKED (0 do) - Ly do: ");
  Serial.println(reason);
  printStatus();
}

void lockDoor(unsigned long ttlSec, const char* reason) {
  if (ttlSec <= 0) ttlSec = 45;
  if (ttlSec > 180) ttlSec = 180; // Giới hạn tối đa 3 phút
  
  isLocked = true;
  lockDeadlineMs = millis() + (ttlSec * 1000UL);
  setServoAngle(ANGLE_LOCKED);
  Serial.print("[SERVO] LOCKED (90 do) - TTL: ");
  Serial.print(ttlSec);
  Serial.print("s - Ly do: ");
  Serial.println(reason);
  printStatus();
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    // Chờ cổng USB Serial sẵn sàng
  }

#if defined(ESP32)
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  doorServo.setPeriodHertz(50);
  doorServo.attach(SERVO_PIN, 500, 2400);
#else
  doorServo.attach(SERVO_PIN);
#endif

  // Mặc định luôn mở khóa khi khởi động
  unlockDoor("startup_init");
  Serial.println("[SYSTEM] Ready to receive commands via USB Serial.");
}

void loop() {
  unsigned long now = millis();

  // Kiểm tra hết hạn TTL -> Tự động mở cửa
  if (isLocked && lockDeadlineMs > 0 && (long)(now - lockDeadlineMs) >= 0) {
    unlockDoor("ttl_expired");
  }

  // Đọc lệnh từ USB Serial
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();

    if (cmd.length() == 0) return;

    if (cmd == "UNLOCK" || cmd == "OPEN") {
      unlockDoor("user_serial_cmd");
    } 
    else if (cmd.startsWith("LOCK")) {
      unsigned long ttl = 45;
      int colonIdx = cmd.indexOf(':');
      if (colonIdx > 0) {
        String ttlStr = cmd.substring(colonIdx + 1);
        ttl = ttlStr.toInt();
        if (ttl <= 0) ttl = 45;
      }
      lockDoor(ttl, "user_serial_cmd");
    }
    else if (cmd.startsWith("ANGLE:")) {
      int angle = cmd.substring(6).toInt();
      setServoAngle(angle);
      Serial.print("[SERVO] Manual angle set to: ");
      Serial.println(angle);
      printStatus();
    }
    else if (cmd == "STATUS") {
      printStatus();
    }
    else {
      Serial.print("[ERROR] Lenh khong hop le: ");
      Serial.println(cmd);
      Serial.println("[HELP] Cac lenh ho tro: UNLOCK | LOCK | LOCK:45 | ANGLE:0..180 | STATUS");
    }
  }
}
