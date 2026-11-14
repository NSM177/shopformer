# Kế hoạch & prompt: Hệ thống camera phát hiện trộm + thông báo admin + khóa cửa

Nov 14, 2026 · @Dung

## Mục tiêu và phạm vi

Hệ thống dùng camera phát hiện người trong cửa hàng, nhận diện hành vi nghi trộm, báo admin kèm clip bằng chứng, rồi cho admin khóa cửa có thời hạn để nhân viên kịp xử lý. Đây là đồ án demo ở trường: model VideoMAE dùng giấy phép CC BY-NC 4.0 nên không dùng thương mại.

**Nguyên tắc an toàn (không thương lượng):**

- Mặc định admin xem clip rồi mới bấm khóa. Chế độ tự khóa là tùy chọn và tắt sẵn.
- Khóa có thời hạn 30–60 giây rồi tự mở lại. Luôn có nút mở cửa và lối thoát hiểm.
- Mất điện hoặc mất mạng thì cửa mở (fail-safe), nên dùng khóa từ thay vì khóa chốt.
- Model sẽ báo nhầm, nên kết quả của nó chỉ là gợi ý cho người, không phải bằng chứng để giữ người.

**Trong phạm vi:** phát hiện và theo dõi người, nhận diện hành vi, cảnh báo, dashboard, điều khiển cửa (ESP32 hoặc giả lập). **Ngoài phạm vi:** nhận diện khuôn mặt, định danh người, lưu trữ dài hạn.

## Kiến trúc và hợp đồng MQTT

&#91;embedded content: kiến trúc hệ thống · 4 bước pipeline, 1 broker, 2 thiết bị đầu cuối\]

Pipeline chỉ gửi cảnh báo lên broker. Dashboard là nơi duy nhất phát lệnh khóa, cửa chỉ nghe lệnh và báo lại trạng thái.

| Topic | Chiều | Payload |
| --- | --- | --- |
| `store/{store_id}/cam/{cam_id}/alert` | pipeline → admin | `{event_id, ts, cam_id, track_id, score, threshold, clip_path}` (thêm `track_id`) |
| `store/{store_id}/cam/{cam_id}/status` | pipeline → broker | `online` / `offline` (retained, Last Will) |
| `store/{store_id}/door/cmd` | dashboard → cửa | `{action: lock\|unlock, reason, event_id, ttl_sec}` |
| `store/{store_id}/door/status` | cửa → dashboard | `{state: locked\|unlocked, ts, event_id}` (retained) |

**Máy trạng thái của mỗi camera:**

1. IDLE: không có nghi vấn.
2. SUSPECT: điểm vượt ngưỡng ở 1 cửa sổ.
3. ALERT: K cửa sổ liên tiếp vượt ngưỡng, gửi alert kèm clip.
4. CHỜ ADMIN: đợi lệnh; hết N giây không phản hồi thì về IDLE (chỉ tự khóa nếu bật chế độ tự động).
5. LOCKED: cửa khóa, đếm ngược `ttl_sec`.
6. RELEASED: admin mở hoặc hết hạn, về IDLE và vào cooldown.

## Kế hoạch theo giai đoạn

Làm theo thứ tự dưới đây; mỗi giai đoạn phải xong thì mới sang giai đoạn sau, vì các bước sau dựa vào đầu ra của bước trước.

| # | Giai đoạn | Đầu ra | Xong khi |
| --- | --- | --- | --- |
| 0 | Dọn cấu trúc dự án | `data/raw/` chứa PoseLift, `notebooks/` chứa notebook, test đổi tên theo module, `*.zip` vào `.gitignore` | `pytest` chạy được và `git status` không còn file dữ liệu lạ |
| 1 | Quay dữ liệu | `clips/normal`, `clips/shoplifting`, tên file bắt đầu bằng mã người | tối thiểu 4 người, 40–60 clip mỗi lớp, có cả clip dễ nhầm |
| 2 | Phát hiện người và cắt clip | module `vision/person_clips`: YOLO + tracking, cắt vùng quanh từng người | trên video thử, mỗi người giữ được một `track_id` và clip cắt được lưu ra |
| 3 | Fine-tune VideoMAE trên clip đã cắt | bundle model (`train_videomae.ipynb` đã có, chỉ đổi nguồn clip) | có AUC và ma trận nhầm lẫn trên tập test tách theo người |
| 4 | Bộ quyết định và cảnh báo | máy trạng thái, `demo_videomae.py` nối với YOLO, gửi `alert` qua MQTT | video thử có 1 sự kiện cho ra đúng 1 cảnh báo (không lặp) |
| 5 | Điều khiển cửa | `door_controller` + script giả lập ESP32 (sau này thay bằng ESP32 thật) | nhận `door/cmd`, đổi trạng thái, gửi `door/status`, tự mở khi hết `ttl_sec` |
| 6 | Dashboard và Telegram | nút Khóa / Mở, xem clip cảnh báo, hiển thị trạng thái cửa | bấm nút thì trạng thái cửa trên dashboard đổi theo |
| 7 | Kiểm thử đầu cuối | `docker-compose up` chạy cả hệ thống, kịch bản demo | chạy trọn luồng camera → cảnh báo → khóa → mở không cần can thiệp bằng tay ngoài nút bấm |

Giai đoạn 1 là việc lâu nhất và quyết định chất lượng, nên bắt đầu quay sớm, song song với giai đoạn 0 và 2.

## Prompt cho trợ lý lập trình AI

Dán nguyên khối dưới đây vào trợ lý (Claude Code, Cursor, Copilot) khi đã mở sẵn thư mục dự án. Prompt yêu cầu làm từng giai đoạn và dừng để bạn xác nhận, tránh việc trợ lý viết một mạch rồi sai hướng.

```text
Bạn là kỹ sư Python/IoT làm việc trong repo hiện tại của tôi. Hãy đọc cấu trúc repo trước, rồi đề xuất kế hoạch ngắn và chờ tôi đồng ý trước khi sửa code.

# Bối cảnh
Đây là đồ án demo ở trường, không dùng thương mại. Hệ thống: camera -> phát hiện + theo dõi người (YOLO) -> nhận diện hành vi nghi trộm (VideoMAE fine-tune trên clip tôi tự quay, 2 lớp: normal / shoplifting) -> gửi cảnh báo qua MQTT cho admin -> admin bấm nút trên dashboard để khóa cửa ra vào cửa hàng (có thời hạn) -> cửa (ESP32, ban đầu là script giả lập) báo lại trạng thái.

Repo đã có: src/shoplift_iot/{capture,vision,model,alerting,messaging,dashboard}, tests/, configs/, docker/, docker-compose.yml. Đã có sẵn: train_videomae.ipynb, videomae_lite.py (Predictor, đọc clip, tiền xử lý), demo_videomae.py (vòng lặp thời gian thực, buffer cửa sổ, K lần liên tiếp, cooldown, publish MQTT). Tái sử dụng, không viết lại từ đầu.

# Hợp đồng MQTT (không được đổi tên topic hay trường)
- store/{store_id}/cam/{cam_id}/alert  : JSON {event_id, ts, cam_id, track_id, score, threshold, clip_path}, QoS 1
- store/{store_id}/cam/{cam_id}/status : "online" | "offline", retained, dùng Last Will
- store/{store_id}/door/cmd            : JSON {action: "lock"|"unlock", reason, event_id, ttl_sec}, QoS 1
- store/{store_id}/door/status         : JSON {state: "locked"|"unlocked", ts, event_id}, retained

# Ràng buộc an toàn (bắt buộc)
1. Chỉ admin mới phát lệnh lock qua dashboard/Telegram. Thêm cờ cấu hình auto_lock, MẶC ĐỊNH false.
2. Mọi lệnh lock phải có ttl_sec (mặc định 45); cửa tự mở khi hết hạn. Luôn có nút unlock và lệnh unlock được ưu tiên hơn lock.
3. Fail-safe: mất kết nối MQTT hoặc lỗi phần mềm thì cửa về trạng thái MỞ.
4. Mọi lệnh cửa ghi log có thời gian, ai phát, event_id.
5. Không nhận diện khuôn mặt, không định danh người. Chỉ dùng track_id tạm thời.

# Nhiệm vụ (làm từng giai đoạn, xong mỗi giai đoạn thì chạy test, tóm tắt ngắn rồi DỪNG chờ tôi xác nhận)
GD0 Dọn cấu trúc: dữ liệu vào data/raw, notebook vào notebooks/, đổi tên tests/test_stage_*.py theo module, thêm *.zip vào .gitignore, gộp phụ thuộc vào pyproject.toml. Dùng git mv để giữ lịch sử.
GD2 vision/person_clips: YOLO (ultralytics) + ByteTrack, cắt vùng quanh từng track_id (thêm lề ~15%), giữ buffer khung hình theo từng người, xuất clip 16 khung đều nhau đúng định dạng videomae_lite.Predictor. Có script offline cắt clip từ video quay sẵn để tạo dữ liệu train, dùng chung code với lúc chạy thời gian thực.
GD4 Bộ quyết định: máy trạng thái IDLE -> SUSPECT -> ALERT -> WAIT_ADMIN -> LOCKED -> RELEASED (mỗi track/camera), K cửa sổ liên tiếp, cooldown, WAIT_ADMIN hết N giây thì về IDLE. Nối demo_videomae.py với person_clips, gửi alert theo hợp đồng.
GD5 src/shoplift_iot/actuators/door_controller.py: subscribe door/cmd, xử lý lock/unlock/ttl, publish door/status (retained), đọc cấu hình từ configs/. Kèm scripts/fake_door.py giả lập ESP32 (in trạng thái ra màn hình). Kèm firmware/esp32_door/ (Arduino/PlatformIO) dùng cùng hợp đồng, relay/servo ở chân cấu hình được, fail-safe khi mất WiFi/MQTT.
GD6 Dashboard (giữ công nghệ dashboard hiện có): danh sách cảnh báo kèm clip, trạng thái cửa theo thời gian thực, nút Khóa cửa / Mở cửa (có hộp xác nhận cho Khóa). Tùy chọn: bot Telegram gửi ảnh/clip cảnh báo và nhận lệnh Khóa/Mở, token đọc từ biến môi trường.
GD7 docker-compose: mosquitto + pipeline + dashboard + fake_door; README nêu cách chạy demo từng bước và các ràng buộc an toàn.

# Yêu cầu chất lượng
- Python 3.10+, type hints, không hard-code đường dẫn/IP/ngưỡng (để trong configs/*.yaml + biến môi trường).
- Mỗi giai đoạn có pytest: máy trạng thái (test chuyển trạng thái, ttl, cooldown, unlock ưu tiên), door_controller (dùng broker giả hoặc mock), hợp đồng JSON.
- Không đổi hợp đồng MQTT, không xóa code đang chạy, không tải model lớn vào git.
- Nói thẳng những gì chưa kiểm thử được (ví dụ chưa có ESP32 thật, chưa có GPU) thay vì giả vờ đã chạy.

Bắt đầu bằng: liệt kê cấu trúc repo bạn thấy, nêu những chỗ lệch so với mô tả trên, rồi đề xuất kế hoạch cho GD0.
```

## Kịch bản demo và rủi ro

**Kịch bản demo trên lớp:**

1. Chạy `docker-compose up`, mở dashboard.
2. Cho camera xem cảnh mua sắm bình thường: không có cảnh báo.
3. Diễn cảnh giấu đồ: dashboard hiện cảnh báo kèm clip.
4. Admin bấm Khóa cửa: cửa giả lập (hoặc ESP32) chuyển sang locked và báo lại trạng thái.
5. Bấm Mở cửa, hoặc đợi hết `ttl_sec`: cửa tự mở.
6. Tắt broker đột ngột: cửa tự về trạng thái mở, để chứng minh fail-safe.

**Rủi ro cần nói rõ khi bảo vệ:**

- Model sẽ báo nhầm. Trình bày AUC, tỷ lệ báo nhầm theo ngưỡng và giải thích vì sao cửa không tự khóa.
- Dữ liệu tự quay ít nên model chỉ chắc chắn trong cảnh quay tương tự; nêu rõ số người và số clip.
- Giấy phép CC BY-NC 4.0 của model gốc chỉ cho phép dùng phi thương mại.
- Quyền riêng tư: chỉ quay người đồng ý, không đưa clip lên mạng, không nhận diện khuôn mặt.
- Chưa kiểm thử với ESP32 thật; phần giả lập chỉ chứng minh luồng lệnh.
