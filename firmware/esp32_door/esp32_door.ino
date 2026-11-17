/*
 * GD5: Firmware ESP32 điều khiển khóa cửa qua MQTT bằng Động cơ Servo (SG90 / MG996R)
 *
 * Hợp đồng MQTT:
 *   - Subscribe: store/{store_id}/door/cmd
 *       JSON: {"action": "lock"|"unlock", "reason": "...", "event_id": "...", "ttl_sec": 45}
 *   - Publish (retained): store/{store_id}/door/status
 *       JSON: {"state": "locked"|"unlocked", "ts": "...", "event_id": "...", "servo_angle": 90|0}
 *
 * Hành vi Động cơ Servo:
 *   - MỞ (UNLOCKED): Servo quay góc 0 độ (Mở chốt cửa).
 *   - ĐÓNG/KHÓA (LOCKED): Servo xoay góc 90 độ (Đóng chốt khóa cửa).
 *
 * Ràng buộc an toàn (Fail-safe):
 *   1. Khởi động hoặc mất mạng: Servo luôn quay về góc 0 độ (MỞ).
 *   2. Mọi lệnh lock bắt buộc có ttl_sec (mặc định 45s, tối đa 120s); hết hạn tự xoay về 0 độ.
 *   3. Lệnh unlock luôn được ưu tiên và mở cửa ngay lập tức.
 *   4. Mất kết nối WiFi hoặc MQTT -> xoay về 0 độ ngay lập tức trước khi thử kết nối lại.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// ---------------- Cấu hình phần cứng & mạng ----------------
#ifndef WIFI_SSID
#define WIFI_SSID "YOUR_WIFI_SSID"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#endif

#ifndef MQTT_HOST
#define MQTT_HOST "192.168.1.100"
#endif

#ifndef MQTT_PORT
#define MQTT_PORT 1883
#endif

#ifndef STORE_ID
#define STORE_ID "store_01"
#endif

// Chân GPIO điều khiển động cơ Servo
#ifndef SERVO_PIN
#define SERVO_PIN 18
#endif

// Góc xoay Servo chốt cửa
#define SERVO_UNLOCKED_ANGLE 0    // 0 độ: Mở cửa
#define SERVO_LOCKED_ANGLE   90   // 90 độ: Đóng/Khóa chốt cửa

// Chân GPIO Relay phụ trợ (tùy chọn nếu dùng kết hợp khóa nam châm)
#ifndef DOOR_RELAY_PIN
#define DOOR_RELAY_PIN 26
#endif

#ifndef RELAY_LOCK_LEVEL
#define RELAY_LOCK_LEVEL HIGH
#endif
#define RELAY_UNLOCK_LEVEL (RELAY_LOCK_LEVEL == HIGH ? LOW : HIGH)

static const unsigned long DEFAULT_TTL_SEC = 45;
static const unsigned long MAX_TTL_SEC = 120;

WiFiClient espClient;
PubSubClient mqttClient(espClient);
Servo doorServo;

String cmdTopic = String("store/") + STORE_ID + "/door/cmd";
String statusTopic = String("store/") + STORE_ID + "/door/status";

bool isLocked = false;
unsigned long lockDeadlineMs = 0;
String currentEventId = "";
unsigned long lastReconnectAttemptMs = 0;

void applyHardwareState(bool lockState) {
  isLocked = lockState;
  int targetAngle = isLocked ? SERVO_LOCKED_ANGLE : SERVO_UNLOCKED_ANGLE;
  
  // Xoay động cơ servo
  doorServo.write(targetAngle);
  digitalWrite(DOOR_RELAY_PIN, isLocked ? RELAY_LOCK_LEVEL : RELAY_UNLOCK_LEVEL);

  Serial.printf("[SERVO MOTOR] Xoay sang goc: %d do | Trang thai: %s\n",
                targetAngle, isLocked ? "LOCKED (DONG CHOT CUA)" : "UNLOCKED (MO CHOT CUA)");
}

void publishDoorStatus() {
  if (!mqttClient.connected()) {
    return;
  }
  JsonDocument doc;
  doc["state"] = isLocked ? "locked" : "unlocked";
  doc["ts"] = String(millis() / 1000) + "s";
  doc["event_id"] = currentEventId;
  doc["servo_angle"] = isLocked ? SERVO_LOCKED_ANGLE : SERVO_UNLOCKED_ANGLE;

  char buffer[256];
  serializeJson(doc, buffer, sizeof(buffer));
  mqttClient.publish(statusTopic.c_str(), buffer, true);  // retained = true
}

void failSafeUnlock(const char* reason) {
  bool changed = isLocked;
  applyHardwareState(false);
  lockDeadlineMs = 0;
  Serial.printf("[AUDIT] action=unlock | reason=%s | event_id=%s | state=unlocked | servo=%d do\n",
                reason, currentEventId.c_str(), SERVO_UNLOCKED_ANGLE);
  if (changed) {
    publishDoorStatus();
  }
}

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    failSafeUnlock("json_parse_error");
    return;
  }

  const char* action = doc["action"] | "";
  const char* reason = doc["reason"] | "mqtt_cmd";
  const char* eventId = doc["event_id"] | "";
  long ttlSec = doc["ttl_sec"] | (long)DEFAULT_TTL_SEC;

  // Ưu tiên lệnh unlock tuyệt đối
  if (strcmp(action, "unlock") == 0) {
    currentEventId = String(eventId);
    applyHardwareState(false);
    lockDeadlineMs = 0;
    Serial.printf("[AUDIT] action=unlock | reason=%s | event_id=%s | state=unlocked | servo=%d do\n",
                  reason, currentEventId.c_str(), SERVO_UNLOCKED_ANGLE);
    publishDoorStatus();
    return;
  }

  if (strcmp(action, "lock") == 0) {
    if (ttlSec <= 0) {
      failSafeUnlock("invalid_ttl");
      return;
    }
    if ((unsigned long)ttlSec > MAX_TTL_SEC) {
      ttlSec = MAX_TTL_SEC;
    }
    currentEventId = String(eventId);
    applyHardwareState(true);
    lockDeadlineMs = millis() + ((unsigned long)ttlSec * 1000UL);
    Serial.printf("[AUDIT] action=lock | reason=%s | event_id=%s | ttl_sec=%ld | state=locked | servo=%d do\n",
                  reason, currentEventId.c_str(), ttlSec, SERVO_LOCKED_ANGLE);
    publishDoorStatus();
    return;
  }

  // Lệnh không hợp lệ -> Fail-safe mở cửa
  failSafeUnlock("unknown_action");
}

bool reconnectMqtt() {
  String clientId = String("esp32-door-") + STORE_ID;
  const char* lwtPayload = "{\"state\":\"unlocked\",\"ts\":\"\",\"event_id\":\"\",\"servo_angle\":0}";
  if (mqttClient.connect(clientId.c_str(), nullptr, nullptr,
                         statusTopic.c_str(), 1, true, lwtPayload)) {
    mqttClient.subscribe(cmdTopic.c_str(), 1);
    publishDoorStatus();
    Serial.println("[MQTT] Connected and subscribed to door/cmd");
    return true;
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  pinMode(DOOR_RELAY_PIN, OUTPUT);

  // Khởi tạo thư viện Servo cho ESP32
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  doorServo.setPeriodHertz(50);             // Tần số PWM chuẩn 50Hz cho Servo SG90/MG996R
  doorServo.attach(SERVO_PIN, 500, 2400);   // Xung điều khiển 500us - 2400us

  // Khởi động luôn ở trạng thái MỞ (0 độ) (Fail-safe)
  applyHardwareState(false);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  mqttClient.setServer(MQTT_HOST, MQTT_PORT);
  mqttClient.setCallback(onMqttMessage);
}

void loop() {
  unsigned long nowMs = millis();

  // 1. Kiểm tra hết hạn TTL -> tự động xoay servo mở cửa về 0 độ
  if (isLocked && lockDeadlineMs > 0 && (long)(nowMs - lockDeadlineMs) >= 0) {
    failSafeUnlock("ttl_expired");
  }

  // 2. Fail-safe khi mất kết nối WiFi hoặc MQTT -> xoay servo mở khóa ngay lập tức
  if (WiFi.status() != WL_CONNECTED || !mqttClient.connected()) {
    if (isLocked) {
      failSafeUnlock("network_or_mqtt_lost");
    }
    if (WiFi.status() == WL_CONNECTED && (nowMs - lastReconnectAttemptMs > 3000UL)) {
      lastReconnectAttemptMs = nowMs;
      reconnectMqtt();
    }
    return;
  }

  mqttClient.loop();
}
