# Kiểm tra Dataset và phân tích 2 notebook: `train_videomae_2.ipynb`, `train_shopformer.ipynb`

> Ngày: 2026-11-17. Thư mục dữ liệu kiểm tra: `Datasets/` (7.6 GB, 382 video `.mp4`, 1 file CSV).
> Cả 2 notebook **đã được sửa trực tiếp** để chạy lại trên Kaggle. Mọi đường dẫn Kaggle cũ được giữ nguyên
> (chỉ **thêm** đường dẫn dò tìm, không xóa). Bản gốc `train_shopformer.ipynb` xem bằng `git diff notebooks/train_shopformer.ipynb`.
> `train_videomae_2.ipynb` chưa được commit; bản gốc của nó có nội dung giống hệt `shopformer1.ipynb` đã phân tích trong `docs/phan_tich_shopformer1.md`.

---

## 1. Kiểm tra Dataset (`Datasets/`)

### 1.1. Tổng quan

| Thư mục | Số video | Độ dài (min / trung vị / max) | Tổng | Độ phân giải | FPS |
|---|---|---|---|---|---|
| `ShopLiftingVideoDataset/normal` | 90 | 5.2 / 11.2 / 20.7 s | 16.9 phút | 640×480 (88), 1920×1080 (2) | 30 |
| `ShopLiftingVideoDataset/shoplifting` | 92 | 7.0 / 10.9 / 13.5 s | 16.8 phút | 640×480 (86), 1920×1080 (6) | 30 |
| `ucf-shopp/.../Normal` | 150 | 7 s / 56 s / **60 phút** | 360 phút | 320×240 | 30 |
| `ucf-shopp/.../Shoplifting` | 50 | 14.9 s / 93 s / **37 phút** | 180 phút | 320×240 | 30 |
| `ucf-shopp/annotations_fixed (2).csv` | 9 dòng | 7 video có nhãn giờ + 2 `skip` | | | |

Không có file hỏng: video nào cũng đọc được khung đầu.

### 1.2. Vấn đề phát hiện

| # | Vấn đề | Bằng chứng | Ảnh hưởng |
|---|---|---|---|
| D1 | **Video trùng hệt nhau** (cùng MD5) | `normal-44 = normal-52`, `normal-50 = normal-51`, `normal-55 = normal-56`, `Normal_Videos_936 = Normal_Videos_937` | Một bản vào train, bản kia vào val/test, nên điểm bị thổi phồng |
| D2 | **Clip Kaggle quay cùng vài cảnh/camera, cùng người diễn** | Gom theo dHash 3 khung: 182 clip chỉ còn **48 cụm cảnh**. Cụm lớn nhất 32 clip (10 normal / 22 trộm); có cụm 11 normal / 0 trộm, cụm 6 trộm / 0 normal | Chia ngẫu nhiên theo clip thì cùng cảnh và cùng người nằm ở cả train lẫn val. Cụm lệch nhãn cũng cho phép model đoán nhãn theo **bối cảnh** |
| D3 | **Chỉ 7/50 video UCF Shoplifting có nhãn giờ** | CSV có 7 video trộm + 2 `skip`; 41 video bị notebook bỏ qua | Dữ liệu trộm "camera thật" quá ít: train chỉ có khoảng 41 cửa sổ dương UCF |
| D4 | **Clip Kaggle trộm không có nhãn giờ**, nên cả clip bị coi là trộm | 92 clip ≤ 15 s: mọi cửa sổ 3 s đều dương, kể cả đoạn đi lại | Nhãn nhiễu, val AUC Kaggle bị "chặn trần" ở ~0.84 |
| D5 | `Shoplifting031_x264.mp4` (UCF) dài 14.9 s | Dưới ngưỡng `AUTO_CLIP_MAX=15`, nên bị tự gán cả clip là trộm | Nên ghi nhãn giờ cho clip này |
| D6 | **Hai nguồn khác hẳn độ phân giải**: Kaggle 640×480, UCF 320×240 | | Model có thể học "ảnh mờ = UCF" (phần lớn âm), tức là học nguồn chứ không học hành vi |
| D7 | Video UCF rất dài | `Normal_Videos_924/935` dài 60 phút, `Shoplifting014` dài 37 phút | Notebook đã giới hạn (`NEG_RANGE=90 s`, `CTX=40 s`), nên phần lớn video dài không được dùng |
| D8 | **Không có dữ liệu PoseLift** trong `Datasets/` | Không có `.pkl` / `.npy` | `train_shopformer.ipynb` **không chạy được** với dataset này. Cần Add Input dataset PoseLift riêng trên Kaggle (xem mục 5.1) |

Các nhãn giờ hiện có đều nằm trong độ dài video (đã kiểm tra `end_frame ≤ số khung`):

| Video | Khoảng trộm | Độ dài vụ | Có `px,py` |
|---|---|---|---|
| Shoplifting015 | 67.0 – 72.0 s | 5.0 s | không |
| Shoplifting027 | 36.0 – 38.7 s và 49.0 – 57.0 s | 2.7 s + 8.0 s | không |
| Shoplifting028 | 19.0 – 28.0 s | 9.0 s | có |
| Shoplifting029 | 34.0 – 49.0 s | 15.0 s | có |
| Shoplifting039 | 73.0 – 78.0 s | 5.0 s | có |
| Shoplifting044 | 369.0 – 375.0 s | 6.0 s | có |

---

## 2. `train_videomae_2.ipynb`: phân tích

Notebook này có nội dung giống hệt `shopformer1.ipynb` đã phân tích trong `docs/phan_tich_shopformer1.md`.
Nó fine-tune VideoMAE trên crop từng người (YOLOv8s + ByteTrack, cửa sổ 3 s, bước 1 s).
**Output đang lưu trong notebook là của bản CŨ** (trước khi sửa), nên cần chạy lại.

### 2.1. Kết quả lần chạy trước

| | Giá trị |
|---|---|
| Train / val / test | 737 / 5649 / 1561 cửa sổ |
| Val AUC Kaggle / UCF | 0.843 / 0.937 (UCF chỉ có **15** cửa sổ dương) |
| Val theo sự kiện | bắt 16/25 vụ (64%), 0.18 báo nhầm/phút |
| Test | **0 cửa sổ dương**, AUC = NaN, nên không đo được khả năng bắt trộm |

### 2.2. Lỗi và điểm không hợp lý

| # | Lỗi | Mức |
|---|---|---|
| V1 | Tự chọn test bằng `v not in _fixed_vids`, mà toàn bộ video UCF có nhãn lại nằm trong `annotations_fixed`, nên test không có video trộm | **Cao** |
| V2 | Không loại video trùng (D1) | **Cao** |
| V3 | Chia val theo từng clip, không theo cảnh (D2), nên rò rỉ cảnh và người diễn | **Cao** |
| V4 | Nhãn Kaggle cả clip là trộm (D4) | **Cao**: phải sửa bằng nhãn giờ |
| V5 | Overfit: loss giảm từ 0.42 xuống 0.07 mà val không đổi; mở khóa toàn bộ 86 M tham số với 737 cửa sổ | Trung bình |
| V6 | Độ phân giải lệch giữa hai nguồn (D6) | Trung bình |
| V7 | Tăng cường dùng `default_rng()` không seed | Thấp |
| V8 | Pipeline trong repo cắt ảnh khác notebook (chữ nhật 15%, không che người khác, `yolov8n-pose`) | Trung bình, **ngoài phạm vi**: lần này chỉ sửa notebook |

### 2.3. Đã sửa trong notebook

| Ô | Thay đổi | Sửa lỗi |
|---|---|---|
| 0 (markdown) | Thêm bảng "Bản sửa ngày 2026-11-17" | |
| 1 (CFG) | `AUTO_TEST_POS` 3 → **2**, `EPOCHS` 4 → **6**; thêm `FREEZE=4`, `LOWRES_P=0.3`, `SCENE_DHASH=10` (đều chỉnh được bằng biến môi trường) | V1, V5, V6, V3 |
| 4 | **Loại video trùng hệt** (so kích thước + MD5 của 2 MB đầu) trước khi đọc nhãn | V2 |
| 5 | Bỏ điều kiện `not in _fixed_vids` khi tự chọn video test | V1 |
| 5 | **Chia val theo nhóm cảnh**: dHash 3 khung (20/50/80%), gom clip ngắn cùng nguồn có khoảng cách ≤ 10; cả nhóm vào val hoặc cả nhóm ở train; luôn để lại ít nhất 1 nhóm cho train | V3 |
| 5 | In cảnh báo nếu test không có cửa sổ trộm; in số nhóm cảnh dùng chung train/val (phải là 0) | V1, V3 |
| 6 | Đóng băng embedding + `FREEZE` block đầu, in số tham số được train | V5 |
| 6 | `WinDS`: seed theo chỉ số mẫu; thêm tăng cường **thu nhỏ rồi phóng lại** (56–112 px → 224) với xác suất `LOWRES_P` | V7, V6 |
| 7 | Optimizer chỉ nhận tham số `requires_grad` | V5 |
| 9 | `meta.json` ghi thêm danh sách video trùng đã loại, `freeze`, `lowres_p`, số nhóm cảnh; mọi `open(..., "w")` có `encoding="utf-8"` | |

Đã kiểm tra: mọi ô code qua kiểm tra cú pháp. Phần chia theo nhóm cảnh đã chạy thử trên 182 clip Kaggle thật của `Datasets/`:
48 nhóm; val 49 video (32 có trộm), train 133 video (60 có trộm); **0 nhóm dùng chung**.
Chưa chạy được cả notebook trên máy này (cần GPU, YOLO, tải model từ HuggingFace).

> Sửa V3 có thể làm **val AUC Kaggle giảm** so với 0.843. Đây là con số trung thực hơn, không phải model kém đi.

---

## 3. `train_shopformer.ipynb`: phân tích

Notebook **chưa có output** (chưa chạy), nên chỉ phân tích mã. Mô hình là GCAE (ST-GCN) tạo 2 token, rồi Transformer tái tạo;
điểm bất thường là MSE tái tạo; train chỉ trên dữ liệu bình thường của PoseLift.

### 3.1. Lỗi và điểm không hợp lý

| # | Lỗi | Bằng chứng | Mức |
|---|---|---|---|
| P1 | **Khớp không phát hiện được giữ nguyên tọa độ (0, 0)**, độ tin cậy bị bỏ đi. Sau khi trừ tâm, khớp (0,0) thành điểm rất xa, nên sai số tái tạo bị chi phối bởi khớp thiếu thay vì hành vi | Bundle cũ `checkpoints/threshold.json`: trung vị 0.22 nhưng p99 36.7, max 111, tức phân phối đuôi cực dài, dấu hiệu điển hình của khớp (0,0) | **Cao** |
| P2 | **Ngưỡng và đánh giá khác thang điểm**: ngưỡng = p99 MSE từng **đoạn thô**, còn AUC tính trên điểm **khung hình đã làm mượt** (gaussian σ = 8) | Ô 7 và ô 8 | **Cao** |
| P3 | Chấm điểm test dùng bước trượt 12 (bằng lúc train), nên mỗi khung chỉ có 1–2 đoạn phủ, điểm khung thô; bài báo dùng bước 1 | `video_scores` dùng `seg_stride` | Trung bình |
| P4 | GCAE dùng **epoch cuối**, không giữ bản tốt nhất theo val | Ô 5 | Thấp |
| P5 | Đường dẫn dữ liệu chỉ có kiểu Kaggle cũ `/kaggle/input/shopfomer/...`; Kaggle mới gắn dataset vào `/kaggle/input/datasets/<user>/<slug>/...` (thấy trong output của `train_videomae_2`) | Ô 1 | Trung bình (vẫn có dự phòng `rglob` trên `/kaggle/input`) |
| P6 | Histogram điểm dùng trục tuyến tính với phân phối đuôi dài, nên gần như không đọc được | Ô 7 | Thấp |
| P7 | Bundle của notebook (`shopformer_lite_v1`, đoạn 24, kiến trúc tự viết) **không cùng kiến trúc** với bundle pipeline đang dùng (`checkpoints/shopformer_v1`, đoạn 12, `sagc`) | `checkpoints/shopformer_v1/meta.json` | Ghi chú: muốn dùng trong pipeline phải viết thêm bộ nạp |
| P8 | Không có dữ liệu PoseLift trong `Datasets/` (D8) | | **Chặn**: phải Add Input |
| P9 | `fps=15` ghi cứng trong `preprocess.json`, notebook không kiểm tra fps thật của PoseLift | Ô 8 | Thấp: cần xác minh |

### 3.2. Đã sửa trong notebook

| Ô | Thay đổi | Sửa lỗi |
|---|---|---|
| 0 (markdown) | Ghi chú bản sửa | |
| 1 | **Thêm** đường dẫn dò `/kaggle/input/datasets/*/shopfomer/PoseLift/Pickle_files` và `/kaggle/input/datasets/*/*/PoseLift/Pickle_files` (giữ nguyên các đường cũ); thêm `EVAL_STRIDE=4`, `MIN_CONF=0.1`, `MAX_MISSING=0.3` | P5, P3, P1 |
| 2 (`shopformer_lite.py`) | Thêm `fill_missing()`: nội suy tuyến tính theo thời gian từng khớp, khớp thiếu cả đoạn thì lấy tâm khung; `normalize_segment` gọi nó trước khi chuẩn hóa; `make_segments(..., max_missing)` bỏ đoạn thiếu > 30% hoặc còn NaN | P1 |
| 3 | `parse_pkl` giữ cột độ tin cậy: khớp có `conf < MIN_CONF` hoặc tọa độ (0,0) thì thành NaN; đếm tỉ lệ khớp thiếu | P1 |
| 4 | In tỉ lệ khớp thiếu; truyền `max_missing` khi tạo đoạn | P1 |
| 5 | Giữ GCAE tốt nhất theo val loss | P4 |
| 7 | Chấm điểm test với `EVAL_STRIDE`; tính thêm `eer_threshold`, `fpr_at_10fnr`, `threshold_at_10fnr` (giống `checkpoints/eval_metrics.json`, đánh dấu "chỉ tham khảo"); histogram trục log | P3, P6 |
| 8 | Thêm `frame_threshold` = p99 điểm khung hình đã làm mượt trên **video val**, cùng thang với lúc đánh giá; báo tỉ lệ bắt / báo nhầm trên test; `preprocess.json` ghi `eval_stride`, `min_conf`, `max_missing`; `encoding="utf-8"` | P2 |

Đã kiểm tra: chạy **toàn bộ notebook trên CPU với dữ liệu PoseLift giả** (6 video train + 3 test, định dạng `[bbox, (y,x,conf)]`,
khoảng 14% khớp thiếu). Mọi ô đều chạy hết và xuất đủ `model.pt`, `preprocess.json`, `calibration.json`, `meta.json`.
Chỉ số trên dữ liệu giả không có ý nghĩa.

> Giả định cần xác minh khi chạy thật: cột thứ 3 của keypoint trong `.pkl` PoseLift là độ tin cậy. Ô 3 in cấu trúc file (`inspect_pkl`),
> nên hãy xem dòng `(17, 3)` và giá trị cột cuối nằm trong [0, 1]. Nếu không phải, đặt `MIN_CONF=0` để chỉ coi (0,0) là thiếu.

---

## 4. Việc bạn phải tự làm (notebook không làm thay được)

| # | Việc | Vì sao | Công sức |
|---|---|---|---|
| M1 | **Ghi nhãn giờ cho 41 video UCF Shoplifting còn thiếu** + `Shoplifting031` | D3, D5: đây là việc cải thiện kết quả nhiều nhất | 3–5 giờ |
| M2 | Ghi nhãn giờ cho 92 clip Kaggle trộm (file mới `annotations_kaggle.csv`) | D4 | 1–2 giờ |
| M3 | Add Input dataset PoseLift (`Pickle_files` gồm `Train/`, `Test/`, `.pkl` + `.npy`) cho notebook Shopformer | D8 | 10 phút |
| M4 | Sau khi chạy: xem các track normal có điểm > 0.5 (vd. `Normal_Videos_892`) và đánh dấu làm âm khó, hoặc `skip` nếu nhãn sai | Giảm báo nhầm | 30 phút |

Định dạng CSV (notebook tự gộp mọi file `annotations*.csv` trong dataset đã Add Input):

```
video,start,end,label,px,py,start_frame,end_frame
Shoplifting001_x264.mp4,23.0,29.5,shoplifting,160,120,,
Shoplifting003_x264.mp4,,,shoplifting,,,1530,1710
shoplifting-1.mp4,3.5,8.0,shoplifting,,,,
Shoplifting010_x264.mp4,,,skip,,,,
```

- Ghi **giây** (`start,end`) hoặc **khung** (`start_frame,end_frame`; video đều 30 fps).
- `px,py`: tọa độ pixel một điểm trên người trộm tại thời điểm `start`. Video UCF là 320×240 nên x ∈ [0, 320), y ∈ [0, 240).
  Nên ghi khi trong cảnh có nhiều người.
- File tên bắt đầu bằng `annotations_fixed` sẽ **ghi đè** dòng của file khác cho cùng video. Hãy thêm vào đúng file đó,
  hoặc đặt file Kaggle tên `annotations_kaggle.csv`.

---

## 5. Hướng dẫn từng bước trên Kaggle

### 5.1. Chuẩn bị dataset

1. **(M1, M2)** Ghi nhãn theo mục 4. Mở video bằng VLC (Ctrl+T để xem thời gian; phím `E` để tiến từng khung).
2. Trên Kaggle, mở dataset `ucf-shopp`, chọn **New Version**, upload file `annotations_fixed (2).csv` đã bổ sung, và thêm `annotations_kaggle.csv` nếu có.
   **Giữ nguyên tên dataset và cấu trúc thư mục** để đường dẫn trong notebook không đổi.
3. **(M3)** Với Shopformer: Add Input dataset chứa `PoseLift/Pickle_files` (dataset cũ tên `shopfomer`).
   Notebook tự dò `/kaggle/input/shopfomer/PoseLift/Pickle_files`, `/kaggle/input/datasets/*/shopfomer/PoseLift/Pickle_files`,
   rồi đến mọi `.pkl` dưới `/kaggle/input`.

### 5.2. Chạy `train_videomae_2.ipynb`

1. Trên Kaggle, mở notebook cũ và chọn **File → Import Notebook**, rồi chọn file `notebooks/train_videomae_2.ipynb` đã sửa.
   File vẫn mang metadata Kaggle (2 dataset input, GPU T4), nhưng hãy mở tab **Input** để chắc cả `shoplifting-video-dataset` và `ucf-shopp` (bản mới) đều được gắn.
2. Settings: **Accelerator = GPU T4**, **Internet = On** (để tải `yolov8s.pt` và VideoMAE từ HuggingFace).
3. (Tùy chọn) Chỉ định video test thay vì để notebook tự chọn: thêm ô đầu tiên
   ```python
   import os
   os.environ["TEST_VIDEOS"] = "Shoplifting028_x264.mp4,Shoplifting039_x264.mp4,Normal_Videos_892_x264.mp4,Normal_Videos_042_x264.mp4"
   ```
   Có thể đặt thêm `FREEZE`, `LOWRES_P`, `EPOCHS`, `SCENE_DHASH` theo cùng cách.
4. **Run All**. Thời gian: lần đầu khoảng 1.5 giờ (tạo cache crop), sau đó khoảng 7 phút mỗi epoch.
5. Kiểm tra trong output:

   | Ô | Dòng cần thấy | Ý nghĩa |
   |---|---|---|
   | 4 | `Loại 4 video trùng hệt ...` | D1 đã được xử lý |
   | 5 | `nhóm cảnh: ~50 nhóm ...` và `nhóm cảnh dùng chung train/val: 0` | Chia train/val không rò rỉ cảnh |
   | 5 | `test : ... | dương N` với **N > 0**, không có dòng `CẢNH BÁO: test KHÔNG có cửa sổ trộm` | Test đo được khả năng bắt trộm |
   | 6 | `tham số được train: ~57 triệu` (với FREEZE=4) | Đóng băng có hiệu lực |
   | 8 | `AUC test theo nguồn`, `test_caught / test_missed`, `test_fa_per_min` | **Kết quả chính để báo cáo** |

6. Tiêu chí chấp nhận gợi ý: trên test bắt ≥ 70% vụ, với ≤ 0.2 báo nhầm/phút. Nếu chưa đạt:
   - Ghi thêm nhãn (M1). Đây là cách hiệu quả nhất.
   - Thử `FREEZE=6` nếu val AUC đứng yên trong khi loss vẫn giảm.
   - Thử `LOWRES_P=0.5` nếu AUC UCF thấp hơn nhiều so với AUC Kaggle.
7. Tải `/kaggle/working/checkpoints/videomae_crops_v1` về `models/videomae_crops_v1/` trên máy.

### 5.3. Chạy `train_shopformer.ipynb`

1. Import notebook đã sửa lên Kaggle. Add Input dataset PoseLift (bước 5.1.3). GPU là tùy chọn: model nhỏ, CPU cũng chạy được.
2. **Run All**.
3. Kiểm tra trong output:

   | Ô | Dòng cần thấy | Ý nghĩa |
   |---|---|---|
   | 3 | `... -> list [(4,), (17, 3)]` | Đúng định dạng `[bbox, (y, x, conf)]` |
   | 4 | `khớp thiếu / độ tin cậy < 0.1: X%` | Nếu X > 40% thì giảm `MIN_CONF` hoặc tăng `MAX_MISSING` |
   | 4 | `video: train ~93 | val ~10 | test ...` | PoseLift có 104 video train |
   | 5–6 | val loss giảm dần; `GCAE val tốt nhất`, `val tốt nhất` | |
   | 7 | `auc_roc`, `auc_pr`, `eer` so với bài báo (69.15 / 44.49 / 0.38) và bundle cũ (74.33 / 49.13 / 0.33) | Kết quả chính |
   | 7 | Histogram trục log: hai phân phối tách nhau | |
   | 8 | `ngưỡng khung hình (p99 val)` và tỉ lệ bắt / báo nhầm trên test | Ngưỡng để dùng thực tế |

4. So sánh: chạy thêm một lần với `EVAL_STRIDE=1` (chậm hơn) để biết bước trượt ảnh hưởng thế nào.
   Chạy một lần với `MIN_CONF=0` và `MAX_MISSING=1` (gần với hành vi cũ) để thấy tác dụng của việc xử lý khớp thiếu.
5. Tải `/kaggle/working/checkpoints/shopformer_lite_v1` về máy.
   Lưu ý P7: pipeline hiện **chưa** nạp được bundle này. Muốn dùng phải viết thêm bộ nạp (ngoài phạm vi lần sửa này).

### 5.4. Ghi lại kết quả

Sau mỗi lần chạy, chép vào báo cáo:
- Với VideoMAE: các dòng `test : ...`, `AUC test theo nguồn`, JSON `metrics`.
- Với Shopformer: JSON `metrics` và các dòng ngưỡng ở ô 8.

Ghi rõ các tham số môi trường đã dùng (`FREEZE`, `LOWRES_P`, `EVAL_STRIDE`, ...).
