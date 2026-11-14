/*
 * GD5: Firmware ESP32 điều khiển khóa từ cửa hàng qua MQTT (Arduino / PlatformIO)
 *
 * Hợp đồng MQTT:
 *   - Subscribe: store/{store_id}/door/cmd
 *       JSON: {"action": "lock"|"unlock", "reason": "...", "event_id": "...", "ttl_sec": 45}
 *   - Publish (retained): store/{store_id}/door/status
 *       JSON: {"state": "locked"|"unlocked", "ts": "...", "event_id": "..."}
 *
 * Ràng buộc an toàn (Fail-safe):
 *   1. Dùng khóa từ (maglock): khi nhả relay (mất điện / khởi động / mất mạng), cửa ở trạng thái MỞ.
 *   2. Mọi lệnh lock bắt buộc có ttl_sec (mặc định 45s, tối đa 120s); hết hạn tự mở.
 *   3. Lệnh unlock luôn được ưu tiên và mở cửa ngay lập tức.
 *   4. Mất kết nối WiFi hoặc MQTT -> mở khóa ngay lập tức trước khi thử kết nối lại.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

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

// Chân GPIO điều khiển Relay khóa từ (có thể cấu hình qua build flag)
#ifndef DOOR_RELAY_PIN
#define DOOR_RELAY_PIN 26
#endif

// Mức logic kích hoạt khóa từ (Fail-safe: LOW = nhả khóa từ = MỞ, HIGH = cấp điện khóa = KHÓA)
#ifndef RELAY_LOCK_LEVEL
#define RELAY_LOCK_LEVEL HIGH
#endif
#define RELAY_UNLOCK_LEVEL (RELAY_LOCK_LEVEL == HIGH ? LOW : HIGH)

static const unsigned long DEFAULT_TTL_SEC = 45;
static const unsigned long MAX_TTL_SEC = 120;

WiFiClient espClient;
PubSubClient mqttClient(espClient);

String cmdTopic = String("store/") + STORE_ID + "/door/cmd";
String statusTopic = String("store/") + STORE_ID + "/door/status";

bool isLocked = false;
unsigned long lockDeadlineMs = 0;
String currentEventId = "";
unsigned long lastReconnectAttemptMs = 0;

void applyHardwareState(bool lockState) {
  isLocked = lockState;
  digitalWrite(DOOR_RELAY_PIN, isLocked ? RELAY_LOCK_LEVEL : RELAY_UNLOCK_LEVEL);
}

void publishDoorStatus() {
  if (!mqttClient.connected()) {
    return;
  }
  JsonDocument doc;
  doc["state"] = isLocked ? "locked" : "unlocked";
  doc["ts"] = String(millis() / 1000) + "s";
  doc["event_id"] = currentEventId;

  char buffer[256];
  serializeJson(doc, buffer, sizeof(buffer));
  mqttClient.publish(statusTopic.c_str(), buffer, true);  // retained = true
}

void failSafeUnlock(const char* reason) {
  bool changed = isLocked;
  applyHardwareState(false);
  lockDeadlineMs = 0;
  Serial.printf("[AUDIT] action=unlock | reason=%s | event_id=%s | state=unlocked\n",
                reason, currentEventId.c_str());
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
    Serial.printf("[AUDIT] action=unlock | reason=%s | event_id=%s | state=unlocked\n",
                  reason, currentEventId.c_str());
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
    Serial.printf("[AUDIT] action=lock | reason=%s | event_id=%s | ttl_sec=%ld | state=locked\n",
                  reason, currentEventId.c_str(), ttlSec);
    publishDoorStatus();
    return;
  }

  // Lệnh không hợp lệ -> Fail-safe mở cửa
  failSafeUnlock("unknown_action");
}

bool reconnectMqtt() {
  String clientId = String("esp32-door-") + STORE_ID;
  // Last Will: nếu ESP32 mất kết nối/mất nguồn, broker cập nhật door/status về unlocked (retained)
  const char* lwtPayload = "{\"state\":\"unlocked\",\"ts\":\"\",\"event_id\":\"\"}";
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
  // Khởi động luôn ở trạng thái MỞ (Fail-safe)
  applyHardwareState(false);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  mqttClient.setServer(MQTT_HOST, MQTT_PORT);
  mqttClient.setCallback(onMqttMessage);
}

void loop() {
  unsigned long nowMs = millis();

  // 1. Kiểm tra hết hạn TTL -> tự động mở cửa
  if (isLocked && lockDeadlineMs > 0 && (long)(nowMs - lockDeadlineMs) >= 0) {
    failSafeUnlock("ttl_expired");
  }

  // 2. Fail-safe khi mất kết nối WiFi hoặc MQTT -> mở khóa ngay lập tức
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
