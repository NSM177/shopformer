# Hướng dẫn đấu nối & điều khiển Động cơ Servo qua Raspberry Pi

Thư mục này chứa mã nguồn để biến một chiếc **Raspberry Pi (3, 4, hoặc 5)** thành **bộ điều khiển chốt cửa Servo thông minh** cho hệ thống an ninh cửa hàng Shopformer.

---

### 1. Sơ đồ nối chân (Pinout) giữa Servo và Raspberry Pi

Các dòng động cơ Servo thông dụng như **SG90 (nhựa)** hoặc **MG996R (kim loại)** đều có 3 dây màu:

| Dây của Động cơ Servo | Chức năng | Chân trên Raspberry Pi (40-Pin Header) | Ghi chú |
| :--- | :--- | :--- | :--- |
| **Vàng / Cam** | Tín hiệu PWM | **GPIO 18** (Chân vật lý số **12**) | Hỗ trợ Hardware PWM0 (chính xác, không bị giật rung) |
| **Đỏ** | Nguồn dương VCC (+5V) | **Chân 5V** (Chân vật lý số **2** hoặc **4**) | Khuyên dùng nguồn 5V riêng nếu dùng servo lớn MG996R |
| **Nâu / Đen** | Nguồn âm (GND) | **Chân GND** (Chân vật lý số **6, 9, 14, 20**) | Nối chung mass GND |

> ⚠️ **Lưu ý quan trọng về nguồn điện:**
> - Với servo nhỏ **SG90**, bạn có thể lấy nguồn 5V trực tiếp từ chân số 2 của Raspberry Pi để test thử.
> - Với servo lực kéo lớn **MG996R**, dòng khởi động có thể lên tới 1.5A–2.5A, cắm trực tiếp vào Pi dễ gây sập nguồn Pi. Khuyên dùng **nguồn tổ ong/adapter 5V ngoài** cấp cho servo và nối chung chân GND với Pi.

---

### 2. Cài đặt thư viện trên Raspberry Pi

Mở terminal trên Raspberry Pi (qua SSH hoặc màn hình trực tiếp) và chạy:

```bash
# Cập nhật hệ thống
sudo apt update

# Cài đặt thư viện GPIO và MQTT
sudo apt install -y python3-gpiozero python3-paho-mqtt
```

---

### 3. Kiểm tra hoạt động của Động cơ Servo

Chạy chế độ tự kiểm tra góc quay (0° -> 90° -> 0°):

```bash
python3 rpi_servo_door.py --test
```
Nếu Servo quay mượt mà từ 0 độ sang 90 độ rồi quay lại 0 độ là phần cứng đã chuẩn!

---

### 4. Kết nối Raspberry Pi với Hệ thống Shopformer trên Laptop

Khi Dashboard Shopformer đang chạy trên Laptop (ví dụ laptop có địa chỉ IP trong mạng WiFi là `192.168.1.15`):

Chạy lệnh trên Raspberry Pi:
```bash
python3 rpi_servo_door.py --mqtt-host 192.168.1.15 --store store_01
```

Lúc này:
1. Khi trên Dashboard laptop bạn bấm **Khóa cửa (45s)** hoặc AI phát hiện hành vi trộm cắp: Laptop gửi lệnh qua MQTT, Raspberry Pi nhận được và xoay Servo **90°** (Khóa chốt).
2. Sau 45 giây (hoặc khi bấm **Mở cửa ngay**): Raspberry Pi tự động xoay Servo về **0°** (Mở chốt).
3. Nếu Raspberry Pi mất mạng hoặc khởi động lại: Servo luôn giữ ở góc **0°** để đảm bảo an toàn thoát hiểm (Fail-safe).

---

### 5. Thiết lập tự khởi động cùng Raspberry Pi (Systemd Service)

Để Raspberry Pi tự chạy chương trình khóa cửa mỗi khi cắm điện:

1. Tạo file service:
```bash
sudo nano /etc/systemd/system/shopformer-door.service
```

2. Dán nội dung sau (thay đổi đường dẫn và IP laptop của bạn):
```ini
[Unit]
Description=Shopformer Door Controller via Servo on Raspberry Pi
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Shopformer/firmware/raspberry_pi_door
ExecStart=/usr/bin/python3 /home/pi/Shopformer/firmware/raspberry_pi_door/rpi_servo_door.py --mqtt-host 192.168.1.15 --store store_01
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

3. Kích hoạt service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable shopformer-door.service
sudo systemctl start shopformer-door.service
```
