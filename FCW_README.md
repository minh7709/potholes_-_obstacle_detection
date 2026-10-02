# RoadGuard — Hệ thống cảnh báo va chạm phía trước (FCW)

Prototype cảnh báo va chạm cho ô tô, chạy trên **CPU không cần GPU**, dùng một camera
góc nhìn phía trước.

```
Video/Camera
  → Resize (giảm về ≤1280x720)
  → Frame sampling (giữ 10 FPS trên nguồn 30/60 FPS)
  → YOLO detect ─┐
  → ByteTrack ────┴→ Depth (mỗi 5 frame) → Distance (median, EMA)
  → Interpolate → Closing speed → TTC → WARNING/DANGER → Audio + Overlay
  → annotated_video.mp4 + events.csv + summary.json
```

> **Bản chất tốc độ — đọc mục 6.** Từ camera đơn, hệ thống chỉ ước lượng được
> **relative closing speed**, không phải tốc độ thật của xe máy. Đây là giới hạn
> của bài toán, không phải của cách cài đặt.

---

## 1. Cài đặt

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-fcw.txt \
    --index-url https://download.pytorch.org/whl/cpu
```

Bỏ `--index-url` nếu máy có GPU. Model tải tự động ở lần chạy đầu:
`yolov8n.pt` (6 MB) và `Depth-Anything-V2-Metric-Outdoor-Small-hf` (100 MB).

**Dùng `.venv` bắt buộc.** `venv` cũ chạy Python 3.14, không có `torch`/`pygame`.

---

## 2. Chạy

```bash
# Video điện thoại (1080p 30 hoặc 60 FPS đều được, không cần convert)
.venv/bin/python main_fcw.py --source video.mp4

# Camera
.venv/bin/python main_fcw.py --source 0

# Ảnh đơn (chỉ đo khoảng cách, không có TTC)
.venv/bin/python main_fcw.py --source image.jpg
```

Phím (chế độ camera): `Q`/`ESC` thoát · `P` tạm dừng · `M` tắt/bật âm.

### Web app

```bash
./run_all.sh          # http://localhost:8000
```

Web app dùng pipeline FCW, ghi cảnh báo vào SQLite, hiển thị lịch sử và thống kê.

### Các tuỳ chọn thường dùng

```bash
# Tối ưu tốc độ (giảm độ chính xác một chút)
.venv/bin/python main_fcw.py --source v.mp4 \
    --depth-resolution 266 --depth-interval 8 --ego-interval 8

# Hiệu chỉnh depth bằng một khoảng cách đã biết
.venv/bin/python main_fcw.py --source v.mp4 --calibrate 10.0
# → đọc "Model reported: 8.4 m" → chạy lại với --depth-scale 1.1905

# Đo trần hiệu năng (tắt adaptive)
.venv/bin/python main_fcw.py --source v.mp4 --benchmark --no-adaptive

# Ghi đè bất kỳ khóa config nào
.venv/bin/python main_fcw.py --source v.mp4 --set collision.danger_ttc=1.5
```

---

## 3. Output

```
output/
├── annotated_video.mp4   # overlay, ghi ở FPS xử lý
├── events.csv           # mọi frame × mọi object
└── summary.json         # tổng hợp + theo từng track
```

**`events.csv`** — đúng thứ tự cột bạn yêu cầu, thêm `distance_source`:

```
timestamp,frame,track_id,class,confidence,distance_m,distance_source,
closing_speed_mps,ttc_seconds,status,in_roi,ego_speed_mps,depth_computed
12.40,372,5,motorcycle,0.94,8.20,measured,1.40,5.86,NORMAL,1,8.20,1
13.10,393,5,motorcycle,0.95,6.90,interpolated,2.00,3.45,WARNING,1,8.20,0
13.80,414,5,motorcycle,0.96,5.10,measured,2.60,1.96,DANGER,1,8.20,1
```

`distance_source` là cột quan trọng nhất trong file này:

| Giá trị | Nghĩa |
|---|---|
| `measured` | Có chạy Depth Anything V2 thật tại frame này |
| `interpolated` | Nội suy tuyến tính giữa hai lần đo thật |
| `held` | Giữ giá trị hợp lệ gần nhất (frame không đo được, hoặc nội suy quá dốc) |

Dòng `interpolated` vẫn tham gia hồi quy closing speed, nên **đọc kết quả là ước
lượng, không phải đo đạc**.

**`annotated_video.mp4` chỉ chứa frame đã phân tích** — không thể dựng lại frame bị
bỏ. Vì vậy nó được ghi ở *FPS xử lý*, không phải FPS gốc. Clip 30 s ở 30 FPS với
step 3 cho file 23 s. Dùng `events.csv` khi cần đối chiếu thời gian chính xác.

---

## 4. Cách tối ưu (và con số đo thật)

Máy đo: **Intel i5-8250U, 4 nhân / 8 luồng, 8 GB RAM, không GPU**.

Chi phí từng stage, đo trực tiếp (1280x720):

| Stage | Thời gian |
|---|---|
| YOLO track @640 | 49–55 ms |
| Depth Anything V2 Small @224 | 250–285 ms |
| Depth @350 | 841 ms |
| Depth @518 | 2580 ms |
| Optical flow (ego) | 14 ms |
| Resize 1080p → 720p | 11.5 ms |
| Overlay | <1 ms |

**Depth chiếm 80% thời gian.** Đó là lý do `depth.interval` là đòn bẩy số một,
và lý do `processing_height` mặc định là 224 chứ không phải 518 (chậm 9×).

Bốn đòn bẩy theo thứ tự hiệu quả:

**1. `depth.interval: 5`** — giảm chi phí depth xuống 1/5 (285 → 57 ms/frame).
Khoảng cách giữa các frame bị nội suy, đánh dấu `interpolated` trong CSV.

**2. `processing.target_fps: 10`** — giảm số frame đưa vào pipeline. Video 60 FPS
và 30 FPS tốn **gần như bằng nhau**: đã đo 174 ms/frame ở 60 FPS (step 6) so với
160 ms/frame ở 30 FPS (step 3). Không có tính năng này, video 60 FPS tốn gấp đôi.

**3. `input.max_width/height: 1280x720`** — resize 4K trước khi suy luận, giữ tỉ lệ
khung hình (`INTER_AREA` để tránh răng cương trên xe ở xa). Bounding box tự động
đúng vì YOLO trả về tọa độ trong không gian của frame đã resize.

**4. `ego_motion.interval`** — optical flow chạy theo nhịp depth. Vận tốc xe đã được
làm mượt mạnh và thay đổi chậm, nên giảm 14 ms → ~3 ms/frame gần như không mất
độ chính xác.

### Số đo thật (1080p)

Video 1080p, `--max-frames 300`, máy đang có tải nền (load ~3):

| Nguồn | Step | Xử lý | Thời gian xử lý | Tốc độ |
|---|---|---|---|---|
| 30 FPS | 3–4 | 232 frame / 29.9 s | 38.4 s | **0.78× realtime** |
| 60 FPS | 6 | 200 frame / 19.9 s | 34.8 s | **0.57× realtime** |

Video 90 giây mất khoảng **115–160 giây**. Máy không tải sẽ nhanh hơn.

### Adaptive

Khi processing FPS < 5 trong 25 frame liên tiếp, hệ thống tự đi xuống bậc thang
`10→8→6→4 FPS`, `depth_interval 5→6→8→10`, và **ghi log**:

```
[FCW] ADAPTIVE: throughput below floor, stepping down to level 1
       (target_fps=8, depth_interval=6) because processing_fps 3.0 < 5.0.
       TTC thresholds unchanged.
```

Hai ràng buộc có chủ đích:

- **Không bao giờ** nới lỏng ngưỡng TTC để đổi lấy FPS. Mất một cảnh báo nhìn thấy
  được để lấy một hỏng học không nhìn thấy là hướng đánh đổi sai.
- Chỉ đi xuống, không đi lên trong cùng một lần chạy, và có `cooldown 8 s`.

---

## 5. Camera realtime: bỏ frame cũ

Khi camera nhanh hơn khả năng xử lý, queue không giới hạn sẽ làm độ trễ tăng vô hạn —
cảnh báo đến muộn vài chục giây, tệ hơn không cảnh báo.

`LiveRunner` dùng **hộp thư 1 chỗ, frame mới ghi đè frame cũ**. Độ trễ luôn bị chặn
ở khoảng một chu kỳ xử lý, dù tụt bao nhiêu. Bấm `P` để tạm dừng.

---

## 6. Closing speed là *relative*, không phải tốc độ thật

Đây là điểm cần nói rõ nhất về hệ thống này.

Với một camera đơn, khoảng cách `d(t)` tới một mục tiêu thay đổi vì **cả hai**
vật thể đều chuyển động:

```
-d(d)/dt = v_xe (xe tiến lại gần xe ô tô) + v_xm (xe máy chạy tới gần ô tô)
```

Camera đơn **không** tách được hai thành phần này. Hệ thống đo `-(d(d)/dt)` rồi trừ
ước lượng vận tốc của xe ô tô, thu được:

```
closing_speed ≈ v_xm     (tốc độ *tương đối* của xe máy)
```

Phần còn lại là sai số, và nó không nhỏ:

| Nguồn sai số | Đại lượng |
|---|---|
| Ego speed từ optical flow | 10–30% — nhiễu LK, giả định cảnh tĩnh |
| Monocular depth | 10–25% |
| Xe rẽ / cắt ngang | Không mô hình hoá được, TTC vô nghĩa |
| Va chạm vào biên | Ảnh hư |

Nên **TTC là ước lượng của thời gian tới va chạm, không phải thời gian tới va chạm thật.**
Đừng dùng nó để quyết định phanh.

Cách giảm sai số (đã làm):

- Hồi quy least-squares trên cửa sổ thời gian thay vì hiệu hai frame
- EMA trên cả khoảng cách lẫn closing speed
- Bù ego-motion bằng optical flow của nền xa
- Median + loại outlier khi lấy depth
- Yêu cầu N frame liên tiếp cùng mức mới cảnh báo
- TTC tính bằng khoảng cách từ chính đường hồi quy, tránh trễ của EMA

### Nâng lên tốc độ tuyệt đối (chưa implement)

Kiến trúc đã chỗ cho việc này:

- **GPS / OBD-II CAN**: đặt `ego_motion.bias_gain` theo tỉ lệ giữa vận tốc ước lượng
  và vận tốc thật, hoặc tắt estimator (`ego_motion.enabled: false`) và cấp vận tốc
  thật qua API. `EgoMotionEstimator` chỉ cần nhận một con số thay vì tự tính.
- **Camera stereo**: loại bỏ hoàn toàn nguồn sai số depth.
- **Radar / LiDAR**: không còn phụ thuộc monocular depth.

---

## 7. Chỉ cảnh báo vật cản chính giữa khung hình

Camera gắn trên kính xe gần vị trí **mắt tài xế**, nên tâm khung hình là xấp xỉ nơi
tài xế đang nhìn. Vì vậy hệ thống có hai vùng:

```
┌────────────────────────────────┐
│                                │
│      ROAD ROI (rộng)          │  x 0.20 → 0.80   theo dõi, đo khoảng cách
│      ┌──────────────┐          │
│      │  CANH BAO    │          │  x 0.33 → 0.67   DUY NHẤT được phát âm
│      │   (hẹp)      │          │  y 0.32 → 0.97
│      └──────────────┘          │
└────────────────────────────────┘
```

Vật cản **ngoài CANH BAO** vẫn được phát hiện, vẽ box, đo khoảng cách, tính TTC — nhưng
không được nâng lên WARNING/DANGER.

**Vì sao phải vậy.** Xe máy dừng cách 2 m ở làn bên cạnh **không phải** mối nguy hiểm
cho xe bạn. Báo động với nó chỉ khiến tài xế tắt tiếng, và mất hiệu lực cảnh báo thật.
Xe ngược chiều đi sát ở Việt Nam rất phổ biến — không có cổng này, hệ thống sẽ hú
liên tục.

Mức cảnh báo là mức nghiêm trọng nhất của hai nguồn, **chỉ khi trong CANH BAO**:

```
TTC ≤ ngưỡng  ─┐
               ├─→ nghiêm trọng nhất ─→ (nếu trong CANH BAO) → WARNING/DANGER + âm
Khoảng cách  ─┘                        (ngoài CANH BAO)      → NORMAL + im lặng
```

Video này cho thấy tác dụng: warning giảm từ **10 frame → 2 frame** sau khi bật cổng.

Chỉnh độ rộng:

```bash
# Camera lệch khỏi trục mắt, muốn rộng ra
--set roi.warn_x_min=0.25 --set roi.warn_x_max=0.75
```

### Căn camera

Cổng này giả định camera nằm gần trục mắt tài xế. Nếu gắn lệch sang một bên, xe ở
làn trước sẽ rơi ra ngoài CANH BAO và hệ thống im lặng. Kiểm tra bằng cách xem vạch
đường: nếu vạch xe chạy nằm ngay giữa vạch vàng trong ảnh thì lệch. Khi đó nới
`warn_x_min/x_max` hoặc dịch camera.

---

## 8. Monocular depth: metric nhưng chưa hiệu chuẩn

Dùng `Depth-Anything-V2-**Metric-Outdoor**` (biến thể huấn luyện cho cảnh giao
thông). Output **có đơn vị mét**. Kiểm chứng trên ảnh thật:

```
Mặt đường phía trước :  3.2 m
Bầu trời              : 37.4 m
Vùng giữa khung hình  : 55.9 m
```

Metric ≠ chính xác. Sai số 10–25% là bình thường, tệ hơn khi xe nhỏ ở xa, bị che,
trời tối, mưa nặng. Ba lớp bảo vệ:

**1. Lấy median trên center-crop.** Không lấy một pixel, cũng không lấy cả bbox
(motorcycle phần lớn diện tích là trời và đường). Cắt 50% trung tâm, loại NaN/±inf/
≤0, loại pixel lệch quá 25% so với median (nền lọt qua kẽ tay), lấy median phần còn
lại, kiểm tra khoảng `[1 m, 120 m]`. Không hợp lệ → `distance = None`, không crash.

**2. Calibration một điểm.**

```bash
.venv/bin/python main_fcw.py --source video.mp4 --calibrate 10.0
#   Model reported : 8.40 m
#   True distance  : 10.00 m
#   Scale factor   : 1.1905
.venv/bin/python main_fcw.py --source video.mp4 --depth-scale 1.1905
```

Sửa sai số hệ thống, không sửa được sai số từng vật thể.

**3. Không giả định độ chính xác.** Chuỗi TTC được xây để chịu sai số chứ không để
giấu nó: interpolation có trần tốc độ giả định, và `None` truyền thẳng xuống dưới
thay vì ép thành số.

### Camera calibration (scaffold)

```bash
# 15 ảnh bàn cờ, đầy khung hình, nhiều góc
.venv/bin/python main_fcw.py --calibrate-cam 0 --calib-pattern 9x6
```

Lưu `config/calibration.yaml` (camera_matrix, dist_coeffs, FOV). Pipeline đọc khi có.
Chưa implement undistort trong đường chính — cần bộ ảnh bàn cờ, MVP không yêu cầu.

---

## 9. Model detector: `assets/best.pt`

Hệ thống dùng **model 15 lớp bạn tự train** (`assets/best.pt`), không dùng COCO:

```
stop_sign, person, bicycle, bus, truck, car, motorbike, reflective_cone,
ashcan, warning_column, spherical_roadblock, pothole, dog, tricycle, fire_hydrant
```

Đo trên máy này (đã warm-up, 1280x720):

| Model | Thời gian/frame | Ghi chú |
|---|---|---|
| `assets/best.pt` | **44.7 ms** | 15 lớp vật cản đường, train trên footage Việt Nam |
| `yolov8n.pt` (COCO) | 51.8 ms | chỉ 80 lớp COCO, không có ổ gà / cột cảnh / rào chắn |

Tức là `best.pt` **nhanh hơn** và đúng domain. Đổi sang COCO chỉ có ý nghĩa khi cần
lớp mà model của bạn không có (ví dụ `traffic light` — `best.pt` chỉ có `stop_sign`).

```bash
.venv/bin/python main_fcw.py --source v.mp4 --yolo-model yolov8n.pt   # COCO
```

### Bẫy tên lớp — đã sửa

Model của bạn gọi xe máy là **`motorbike`**, COCO gọi là **`motorcycle`**. Lọc theo
tên chữ cái khiến lớp quan trọng nhất cho cảnh báo va chạm **không khớp gì cả**, và
lỗi này **im lặng**: pipeline vẫn chạy, chỉ là không bao giờ báo xe máy.

Đã sửa bằng `normalize + alias` (`app/fcw/detector.py`), và giờ:

- `motorcycle` trong config → `motorbike` của model (id 6) ✓
- `stop sign` ↔ `stop_sign` ✓
- `cone` → `reflective_cone`, `trash can` → `ashcan`, `manhole` → `pothole` ✓
- Class yêu cầu mà model không có → **cảnh báo to** lúc khởi động, không im lặng

```bash
# Kiểm tra model có những lớp gì:
.venv/bin/python -c "from ultralytics import YOLO; print(YOLO('assets/best.pt').names)"
```

## 10. Mở rộng sang 14 vật cản

Đã xong: `assets/best.pt` có sẵn 15 lớp và `config.yaml` đang bật **đủ cả 15**:

```yaml
classes:
  targets: [motorbike, car, bus, truck, bicycle, person,
            pothole, reflective_cone, warning_column, spherical_roadblock,
            ashcan, stop_sign, tricycle, fire_hydrant, dog]
  primary: [motorbike]
```

Tách `targets` và `primary` là cố ý:

- **`targets`** — được phát hiện, vẽ box, đo khoảng cách, tính TTC. Đủ 15 lớp.
- **`primary`** — lớp **được phát âm cảnh báo**. Chỉ `motorbike`.

Không nên để `primary` gồm cả `pothole` và `reflective_cone`: hệ thống sẽ hú khi xe
tới gần ổ gà hay cột cảnh, vô nghĩa và gây nhiễu. Mở rộng `primary` khi muốn:

```yaml
primary: [motorbike, car, bus, truck]    # thêm ô tô, xe buýt, xe tải
```

Lưu ý `pothole` vẫn hữu ích ở chỗ khác: nếu web app cần hiển thị cả ổ gà cũ, lớp đó
đã được vẽ box và ghi khoảng cách vào CSV.

---

## 11. Cấu trúc mã nguồn

```
main_fcw.py               CLI: parse, calibration, banner, report
config.yaml               MỌI tham số tinh chỉnh
requirements-fcw.txt
run_fcw.sh

app/fcw/
├── config_loader.py   Đọc YAML + env, truy cập dotted-key
├── frame_sampler.py   Decimation + resize, timestamp chuẩn
├── detector.py        YOLO + ByteTrack
├── depth.py           Depth Anything V2, median, calibration
├── tracking.py        Lịch sử, interpolation, closing speed, ego-motion
├── risk.py            TTC, mức cảnh báo, ROI, chọn target
├── audio.py           Sinh WAV, phát qua thread nền, cooldown
├── visualizer.py      Overlay + HUD hiệu năng
├── monitor.py         Đo throughput + adaptive controller
├── output.py          CSV, summary.json, annotated video
├── runner.py          VideoRunner + LiveRunner
├── calibration.py     Camera intrinsics scaffold
└── source.py          Mở camera / video / URL

app/                    Web app (giữ nguyên kiến trúc gốc)
├── api/routes.py
├── database/models.py
├── services/detector.py
└── static/

tests/                  94 test
output/                 Kết quả chạy
```

---

## 12. Kiểm thử

```bash
.venv/bin/python -m pytest tests/ -v      # 94 passed
```

| File | Số test | Phủ |
|---|---|---|
| `test_fcw_kinematics.py` | 26 | closing speed, ego-motion, TTC, ROI, depth, calibration, audio |
| `test_fcw_warning_path.py` | 7 | Leo thang NORMAL→WARNING→DANGER, cooldown, DB round-trip |
| `test_fcw_optimization.py` | 55 | Config, decimation, timestamp, NaN/inf, interpolation, EMA, CSV, adaptive, cảnh báo quá gần, CANH BAO |

Hai test quan trọng vì chúng bắt được bug thật trong quá trình làm:

- **Timestamp dưới decimation.** 60 FPS → 10 FPS phải lấy frame gốc 0, 6, 12…
  với timestamp 0.0, 0.1, 0.2 s. Sai chỗ này thì mọi tốc độ sai mà vẫn trông hợp lý.
- **Depth interval.** Điều kiện ban đầu đếm *số lần chạy depth* thay vì *số frame đã
  trôi qua*, nên depth chạy đúng một lần rồi **không bao giờ chạy lại** — khoảng cách
  đóng băng im lặng. Test bắt được vì khoảng cách không đổi.

Nhóm `test_fcw_warning_path` dùng perception giả tất định vì các video trong repo chỉ
có ô tô, không có xe máy tiến gần.

---

## 13. Giới hạn

Prototype. **Không phải thiết bị an toàn, không dùng thay phán đoán người lái.**

1. **TTC là ước lượng, và `closing_speed` là tương đối** — xem mục 6.
2. **TTC = distance / closing speed** giả định mục tiêu chạy cùng hướng. Xe ngược
   chiều, xe rẽ, người đi bộ vắt qua làm TTC vô nghĩa mà hệ thống không biết.
3. **Hai frame thay vì một** cho khoảng cách xuyên tắc.
4. **Ban đêm / mưa** — độ chính xác giảm rõ rệt. Đây là điều kiện khó nhất.
5. **Không có phân loại ý định.** Đèn đỏ phía trước bị báo như vật thể.
6. **Xe cắt ngang góc nhìn** có thể thoát ROI rồi quay lại, gây gián đoạn lịch sử.
7. **~0.6–0.8× realtime** trên CPU 4 nhân có tải nền.
8. **Chỉ cảnh báo `motorcycle`.** `classes.primary` trong config đổi được sang lớp khác.

---

## 14. Tham số hay dùng

| Khóa config | Mặc định | Ý nghĩa |
|---|---|---|
| `processing.target_fps` | 10 | FPS đưa vào pipeline |
| `depth.interval` | 5 | Chạy depth mỗi N frame |
| `depth.processing_height` | 224 | 224 nhanh · 350 cân bằng · 518 chính xác |
| `depth.interpolation` | linear | `linear` hoặc `hold` |
| `input.max_width/height` | 1280/720 | Giới hạn resize |
| `distance.smoothing` / `alpha` | ema / 0.3 | EMA ↓ = mượt hơn, trễ hơn |
| `speed.ema_alpha` | 0.5 | EMA cho closing speed |
| `collision.warning_ttc` | 4.0 | Ngưỡng WARNING (không tự đổi) |
| `collision.danger_ttc` | 2.0 | Ngưỡng DANGER (không tự đổi) |
| `collision.danger_distance_m` | 3.5 | Cảnh báo khi quá gần, không cần TTC |
| `collision.warning_distance_m` | 8.0 | |
| `roi.warn_x_min/max` | 0.33 / 0.67 | CANH BAO — vùng duy nhất được phát âm |
| `collision.confirm_frames` | 3 | Frame liên tiếp để xác nhận |
| `ego_motion.interval` | 1 | Nhịp chạy optical flow |
| `tracking.max_age` | 30 | Frame giữ ID khi mất dấu |
| `performance.adaptive_enabled` | true | Tự giảm khi chậm |
| `warning.*_cooldown_sec` | 1.5 / 1.0 | Cooldown âm thanh |

Tương tác:

- `depth.interval ↑` → nhanh hơn, nhưng nhiều dòng `interpolated` hơn
- `distance.alpha ↑` → phản ứng nhanh hơn, nhiễu hơn
- `confirm_frames ↑` → ít cảnh báo giả, chậm hơn
- `ego_motion.enabled=false` → **chỉ để kiểm chứng**, không dùng khi chạy thật