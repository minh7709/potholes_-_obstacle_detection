# MIGRATION_PLAN.md — Python FCW → Android Native (Kotlin)

Kế hoạch chuyển hệ thống FCW Python sang Android. Viết sau khi **inspect và kiểm chứng thực
nghiệm** trên source Python hiện tại, không suy đoán từ tài liệu.

Ngày: 2026-10-04
Nguồn: `potholes_-_obstacle_detection` @ branch `nguyentam`, commit `d1b9035`

---

## 0. Sai lệch giữa spec và source thực tế

| Spec nói | Source thực tế | Xử lý |
|---|---|---|
| `app/fcw/risk_mod.py` | File này **không tồn tại**. Logic ở `app/fcw/risk.py` (328 dòng) | Port từ `risk.py` |
| Class gồm 14 lớp, không `pothole` | Model có **15 lớp, có `pothole` (id 11)**, không có `pole` | Lấy từ ONNX metadata, không hard-code |
| Tài liệu cũ nói class `pole` | `pole` không tồn tại ở bất kỳ đâu trong repo | Bỏ hoàn toàn |

---

## 1. YOLO — đã kiểm chứng đầy đủ

### 1.1 Export

```bash
.venv/bin/python -c "from ultralytics import YOLO; YOLO('assets/best.pt').export(format='onnx', imgsz=640, simplify=True, opset=17)"
```

Kết quả thực tế:

| Thuộc tính | Giá trị |
|---|---|
| Input | `images` `[1, 3, 640, 640]` — **fixed, không dynamic** |
| Output | `output0` `[1, 19, 8400]` |
| Kích thước fp32 | 10.6 MB |
| Opset | 17 |
| `dynamic` | `False` |

`19 = 4 (cx,cy,w,h) + 15 (class scores)`, `8400 = 80×80 + 40×40 + 20×20`.

### 1.2 Ngữ nghĩa output — đã kiểm chứng bằng ảnh thật

**Tọa độ là PIXEL của ảnh input 640×640, KHÔNG phải normalized 0..1.**

Đo trên frame thật: `cx` ∈ [2.1, 637.5], `w` ∈ [2.2, 639.8].

Class scores **đã qua sigmoid** (max thấy 0.8889).

### 1.3 Decode + NMS — đã đối chiếu khớp với Ultralytics

Decode thủ công + NMS class-aware IoU=0.5 cho kết quả **giống hệt** `model.predict()`:

```
stop_sign  conf 0.889  xyxy=[557.6, 146.3, 595.1, 236.2]   ← khớp chính xác
car        conf 0.865  xyxy=[398.9, 330.0, 446.6, 372.1]
car        conf 0.770  xyxy=[485.5, 331.3, 506.6, 353.2]
car        conf 0.675  xyxy=[128.3, 334.1, 168.8, 348.7]
```

32 anchor > 0.45 → NMS còn 4 box.

### 1.4 NMS chưa có trong graph

Kiểm tra op: **không có** `NonMaxSuppression`, `metadata['nms'] = None`.
→ **Phải tự implement NMS trên Android.** ByteTracker cũng cần điểm này.

### 1.5 Class names — nguồn chuẩn là ONNX metadata

```
names = {0:'stop_sign', 1:'person', 2:'bicycle', 3:'bus', 4:'truck', 5:'car',
         6:'motorbike', 7:'reflective_cone', 8:'ashcan', 9:'warning_column',
         10:'spherical_roadblock', 11:'pothole', 12:'dog', 13:'tricycle',
         14:'fire_hydrant'}
```

→ `YoloDetector.kt` **phải đọc `names` từ metadata lúc load**. Nếu thiếu → fail sớm
kèm message, không được fallback sang danh sách hard-code.

### 1.6 Preprocess

Giống Ultralytics: resize giữ tỉ lệ vào 640, pad **114,114,114**, BGR→RGB, ÷255,
CHW, thêm batch dim. Cần giữ `scale` + `pad` để un-letterbox tọa độ về frame gốc.

### 1.7 Quantization (khuyến nghị cho mobile)

| | fp32 | int8 dynamic |
|---|---|---|
| Kích thước | 10.6 MB | **3.0 MB** |
| Inference (desktop 4 nhân) | 60 ms | 120 ms |
| max score | 0.889 | 0.885 (chấp nhận được) |
| Số anchor > 0.45 | 32 | 32 |
| Metadata `names` | có | **có** (giữ được) |

int8 giữ nguyên metadata `names`. Trên mobile NNAPI có thể nhanh hơn fp32, dù CPU
thuần thì chậm hơn. **Quyết định ở Phase 2 sau khi benchmark trên thiết bị thật.**

---

## 2. ByteTrack — cấu hình thực tế

Đọc từ `ultralytics/cfg/trackers/bytetrack.yaml` và khởi tạo runtime:

```
track_high_thresh = 0.25
track_low_thresh  = 0.10
new_track_thresh  = 0.25
track_buffer      = 30
match_thresh      = 0.80
fuse_score        = True
```

### 2.1 Hạn chế có sẵn: nhánh stage 2 không nhận dữ liệu

`byte_tracker.py:331-332` tách detection thành 2 nhóm:

```python
remain_inds = valid & (scores >= self.args.track_high_thresh)   # >= 0.25
inds_low    = valid & (scores >  self.args.track_low_thresh)    # 0.10 .. 0.25
              & (scores <  self.args.track_high_thresh)
```

Nhưng pipeline truyền `conf=0.45` cho `model.track()`. Ultralytics lọc trước khi
ByteTrack kịp thấy, nên **mọi score < 0.45 bị loại và `inds_low` luôn rỗng.**

**Cần nói đúng mức độ.** ByteTrack khớp 2 lần:

```python
strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)  # gồm cả track ĐÃ MẤT
_first_association (strack_pool, detections, ...)        # stage 1: track đã mất ↔ conf cao
_second_association(strack_pool, u_track, detections_second, ...)  # stage 2: conf thấp
```

Nên **track mất vẫn khôi phục được ID ở stage 1** nếu nó hiện lại rõ ràng.
Đo thực tế trên video:

```
xử lý 195 frame — 195 frame đều có track
ID 2 : 128 lần xuất hiện, liên tục frame 0..127
ID 92:  45 lần, frame 149..194
```

Chỉ mất đúng nhánh recovery bằng detection **confidence thấp**:

| Tình huống | Hiện tại |
|---|---|
| Track mất rồi hiện lại rõ (conf > 0.45) | Khôi phục được, stage 1 |
| Track bị che một phần, conf rơi ~0.2 | **Mất ID**, tạo ID mới |
| Track mất > `track_buffer` (30) | ID mới — đúng thiết kế |

Mức độ: **thấp**. Không phải lỗi nghiêm trọng.

**Quyết định đã chốt:** giữ nguyên hành vi Python (không đổi thuật toán).
Hạ detector xuống 0.25 sẽ *cải thiện* ID stability khi che khuất, nhưng không
bắt buộc — để sau, bật bằng cờ.

## 3. Depth — phần khó nhất, đã kiểm chứng kỹ

### 3.1 Vấn đề: Depth Anything V2 không export dynamic được

DPT có position embedding phụ thuộc kích thước input. Export với `dynamic_axes`
fail runtime:

```
ONNXRuntimeError: Got invalid dimensions for input: pixel_values
index: 2 Got: 224 Expected: 518
```

Model ONNX cộng đồng (`77ukhtar/depth-anything-v2-metric-onnx`) **cố định 518×518**,
không resize được.

### 3.2 Giải pháp: export ở kích thước CỐ ĐỊNH nhỏ hơn — đã thành công

Dynamic axes mới là thứ fail. Export từng kích thước cố định **chạy được**:

| Kích thước | fp32 | int8 |
|---|---|---|
| 196×196 | 143 ms | — |
| 252×252 | 267 ms | **188 ms** |
| 350×350 | 680 ms | 634 ms |
| 518×518 (model cộng đồng) | 840 ms | — |

### 3.3 Sai số so với PyTorch 518 — đo thật, đoán không được

Sai số % trên 3 vùng, cùng một ảnh:

| Vùng | ref 518 | ONNX 196 | ONNX 252 | ONNX 350 |
|---|---|---|---|---|
| Đường gần | 3.22 m | **+139.0%** | **+78.4%** | **+21.3%** |
| Tầm trung | 55.93 m | −72.9% | +17.6% | **−0.1%** |
| Bầu trời | 36.92 m | −61.8% | +5.2% | +2.9% |

**Kết luận quan trọng:** giảm kích thước làm **sai số metric lớn lên ở vật thể gần** —
đúng vùng quan trọng nhất cho FCW. 196 và 252 **không phải** bản thay thế drop-in.
350 giữ được ở xa nhưng vẫn lệch +21% ở gần.

→ Dù 350 vẫn cần `DepthCalibration` đúng như Python.

### 3.4 Kích thước model

Export sinh **2 file** (graph `.onnx` + weights `.data` 99 MB) do
`torch.onnx.export` tự externalize. Gộp bằng
`onnx.external_data_helper.convert_model_from_external_data()` → **1 file 99.1 MB**,
load OK. Sau đó dynamic quantization:

| | fp32 | int8 |
|---|---|---|
| Kích thước | 99.1 MB | **27.0 MB** |
| 252 inference | 267 ms | 188 ms |
| 252 road / sky | 5.68 / 38.03 m | 5.63 / 38.59 m |

int8 gần như không đổi kết quả, nhỏ hơn 3.7×.

### 3.5 Tương thích ONNX Runtime Mobile

Ops rủi ro trong graph: `LayerNormalization` (28), `Resize` (5).

Cả hai thường **không được NNAPI hỗ trợ** → CPU fallback hoặc cần custom op.
Đây là rủi ro lớn nhất của toàn bộ kế hoạch, **bắt buộc benchmark Phase 4 trên máy thật**
trước khi cam kết realtime.

### 3.6 Abstraction bắt buộc

`depth/DepthEstimator.kt` là `interface`, pipeline không biết model cụ thể:

```kotlin
interface DepthEstimator {
    fun infer(frame: ImageProxy): DepthResult
    fun close()
}

data class DepthResult(
    val depth: FloatArray,     // metres, [h][w]
    val width: Int, val height: Int,
    val inferenceTimeMs: Float,
    val valid: Boolean
)
```

Cho phép thay bằng: Depth Anything 350 int8 / model nhẹ khác / stub (TTC = ∞).

### 3.7 Model đề xuất nếu Depth Anything không đạt

Theo thứ tự ưu tiên:

1. **Depth Anything V2 Metric Outdoor @350 int8** (27 MB) — giữ metric, đã đo
2. **FastDepth / MiDaS-small int8** (5–15 MB) — nhanh hơn nhiều nhưng là **relative
   depth**, phải cấu hình mốc từ chối khoảng cách định kỳ → không đạt yêu cầu metric
3. **Bỏ depth, dùng bbox height làm proxy** — nhanh nhất, **sai hoàn toàn**, chỉ để
   demo đường ống, phải ghi rõ trong README

Không tự chọn. Quyết định sau Phase 4.

---

## 4. Thuật toán cần port — tham chiếu file và giá trị

### 4.1 Distance estimation (`depth.py` + `risk.py`)

| Tham số | Giá trị |
|---|---|
| center_crop | 0.5 |
| min_valid_samples | 16 |
| outlier_tolerance | 0.25 |
| valid range | 1.0 – 120.0 m |

Thứ tự: clip box → center crop → loại NaN/±inf/≤0 → median → loại pixel lệch quá
25% → median lại → kiểm tra khoảng hợp lệ → nhân `DepthCalibration.scale`.

Fallback: nếu center crop quá nhỏ → lấy dải dưới–trung tâm (60% bề ngang, từ giữa
chiều cao xuống đáy).

### 4.2 Smoothing (`tracking.py`)

| Tham số | Giá trị |
|---|---|
| smoothing | ema |
| alpha | 0.3 |
| history_length | 20 |

### 4.3 Closing speed — hồi quy least-squares

| Tham số | Giá trị |
|---|---|
| min_samples | 4 |
| min_window_sec | 0.6 |
| max_closing_speed_mps | 25.0 |
| ema_alpha | 0.5 |

Công thức, giữ nguyên:

```
t_rel = ts - ts[-1]                      # recenter chống mất chính xác số
slope = Σ((t_rel - mean)(d - mean)) / (var(t_rel) · n)
speed = -slope - egoBias
fitDistance = intercept                  # dùng CHO TTC, không dùng EMA
```

Vì sao không dùng hiệu 2 frame: sai số depth 10% ở 10 m là ~1 m; ở 30 fps
`dt = 0.033 s` → sai số ~30 m/s nếu tính từ 2 frame.

### 4.4 TTC (`risk.py`)

```
distance = fitDistance (không phải EMA — tránh trễ)
TTC = distance / closingSpeed     khi closingSpeed > 1e-3
TTC = Infinity                    khi closingSpeed <= 1e-3
```

**Không** gộp "chưa đủ dữ liệu" với "không closing" (`speedValid` tách riêng).

### 4.5 Risk levels (`risk.py`)

```
classifyTtc:     TTC > 4.0 → NORMAL | 2.0 < TTC <= 4.0 → WARNING | TTC <= 2.0 → DANGER
classifyDistance: d > 8.0  → NORMAL | 3.5 < d <= 8.0  → WARNING | d <= 3.5  → DANGER
rawLevel = worse(ttcLevel, distanceLevel)
```

Distance trigger độc lập TTC — trường hợp xe đứng yên 3 m cho
`closingSpeed = 0 → TTC = ∞ → NORMAL` nhưng `distanceLevel = DANGER → DANGER`.

### 4.6 ROI và Warning Corridor

| | x_min | y_min | x_max | y_max |
|---|---|---|---|---|
| ROI (theo dõi) | 0.20 | 0.30 | 0.80 | 0.97 |
| Corridor (được cảnh báo) | 0.33 | 0.32 | 0.67 | 0.97 |

Ngoài corridor: vẫn detect / track / đo / vẽ telemetry, nhưng `level = NORMAL`.
Telemetry giữ `levelUnrestricted` để UI giải thích được.

`center_tolerance = 0.35`, chỉ ảnh hưởng ranking score (0.10 trọng số), không
ảnh hưởng cổng cứng.

### 4.7 Primary target (`risk.py`)

Chỉ 1 object. Thứ tự: DANGER → WARNING → TTC nhỏ hơn → center score cao hơn →
distance nhỏ hơn. Lọc `class ∈ primary` và `inCorridor`.

### 4.8 Confirmation

`confirm_frames = 3`, `min_hits = 3`. Streak phải **liên tiếp** cùng một mức;
DANGER → NORMAL → DANGER không tính là sustained. Reset khi `primary == null`.

### 4.9 Ego motion (`tracking.py`)

Công thức, giữ nguyên — **không nhân focal length**:

```
v_ego = Z · dy / (yr · dt)        yr = y - H/2
```

Chỉ dùng: `Z >= background_depth_m (15.0)` và `|yr| >= 0.12·H`.
Fit least-squares một lần → loại outlier bằng MAD (3.0) → fit lại.
Smoothing 0.6, giới hạn gia tốc 8.0 m/s², max 33 m/s, `bias_gain = 1.0`.

> Focal đã chứa trong toạ độ pixel của phép chiếu `u = f·X/Z`. Nhân thêm là
> sai số ~1300×. Đây là lỗi đã gặp và đã sửa; giữ nguyên cảnh báo khi review.

### 4.10 Audio (`audio.py`)

| | WARNING | DANGER |
|---|---|---|
| cooldown | 1.5 s | 1.0 s |

Phát trên thread nền, queue bounded 4, drop cũ khi đầy.
`warning.wav` = 2 hồi 660 Hz, `danger.wav` = 3 hồi 950 Hz (đã có trong assets).

---

## 5. Cấu hình đầy đủ để port

```yaml
collision:  warning_ttc 4.0, danger_ttc 2.0,
            warning_distance_m 8.0, danger_distance_m 3.5,
            min_distance_m 0.5, confirm_frames 3
roi:        0.20/0.30/0.80/0.97, corridor 0.33/0.32/0.67/0.97,
            center_tolerance 0.35
distance:   ema, alpha 0.3, history 20
speed:      min_samples 4, min_window_sec 0.6, max 25.0, ema_alpha 0.5
depth:      interval 5, interpolation linear, max_interp_rate 12.0
tracking:   max_age 30, min_hits 3, buffer 30
processing: target_fps 10
input:      max 1280x720
classes:    conf 0.45, iou 0.5, min_area_ratio 0.0008
warning:    cooldown 1.5 / 1.0
ego_motion: enabled, background_depth 15.0, min_row_offset 0.12,
            smoothing 0.6, bias_gain 1.0
```

`classes.primary = [motorbike]`, `classes.targets` = đủ 15 lớp.

---

## 6. Test — chuyển từ Python sang Kotlin

Python hiện có **94 test**. Cần port **tối thiểu** (spec §27):

| # | Test |
|---|---|
| 1 | TTC ≤ danger → DANGER |
| 2 | TTC giữa 2 khoảng → WARNING |
| 3 | TTC > warning → NORMAL |
| 4 | closingSpeed ≤ 0 → TTC infinity |
| 5 | distance ≤ dangerDistance → DANGER |
| 6 | distance ≤ warningDistance → WARNING |
| 7 | distance gần nhưng closingSpeed = 0 → **vẫn DANGER** |
| 8 | ngoài warning corridor → NORMAL |
| 9 | primary target chọn DANGER trước WARNING |
| 10 | TTC nhỏ hơn được ưu tiên |
| 11 | confirmation frames hoạt động |
| 12 | track history tính closing speed đúng |
| 13 | interpolation không tạo distance âm |
| 14 | distance source đúng |
| 15 | ROI scale đúng với nhiều resolution |

Thêm từ Python hiện có:
- ego-motion: vật đứng yên + ego 2 m/s → closing ≈ 0 (**test chống hồi quy**)
- depth loại NaN/inf
- `motorbike` ↔ `motorcycle` alias matching (11 ca)
- calibration scale

> `tests/` trong cấu trúc Android đặt sai chỗ — Gradle chỉ chạy test trong
> `app/src/test/java/...`. Đề xuất dùng `app/src/test/java/com/example/fcw/`.

---

## 7. Điểm không thể hoàn thành / rủi ro

| # | Vấn đề | Mức | Xử lý |
|---|---|---|---|
| 1 | **Depth ONNX dùng `LayerNormalization` + `Resize`** — NNAPI thường không hỗ trợ | **cao** | Phải benchmark Phase 4 trên máy thật |
| 2 | Depth 99 MB fp32 | **đã xử lý** | int8 → 27 MB |
| 3 | Depth ONNX **fixed size**, không resize | trung bình | Chọn 1 kích thước khi export |
| 4 | Sai số metric +21% ở vật thể gần với ONNX 350 | trung bình | `DepthCalibration` bắt buộc |
| 5 | ByteTrack nhánh stage 2 không nhận dữ liệu (lỗi có sẵn) | thấp | Chấp nhận, giữ nguyên hành vi Python |
| 6 | Optical flow trên mobile | trung bình | Phase 10, abstraction trước |
| 7 | Chưa đo được FPS thực tế trên thiết bị | — | Phải có máy thật từ Phase 2 |
| 8 | **Chưa đo được FPS trên thiết bị thật** | cao | Cần máy test để xác nhận realtime |

---

## 8. Thứ tự thực hiện

Đúng thứ tự spec §28, kèm điểm kiểm chứng:

| Phase | Nội dung | Điều kiện qua |
|---|---|---|
| 1 | Project + CameraX preview | App chạy, thấy camera |
| 2 | YOLO ONNX | Box đúng, **đo FPS thật** |
| 3 | ByteTrack | ID giữ nguyên khi xe di chuyển |
| 4 | Depth abstraction | **Benchmark 3 kích thước + int8 trên máy** |
| 5 | TrackHistory + closing speed | Khớp Python trong test |
| 6 | RiskEngine + ROI + corridor | 15 test business logic pass |
| 7 | Primary target + confirmation | — |
| 8 | Compose overlay | Vẽ đủ 12 thành phần |
| 9 | Audio + vibration | Không spam |
| 10 | Ego motion | Xe đứng yên không báo |
| 11 | Optimization + adaptive | Không nới threshold TTC |

**Ưu tiên correctness.** Không sang phase sau khi chưa qua phase trước.

---

## 9. Quyết định đã chốt

| # | Quyết định | Ghi chú |
|---|---|---|
| 1 | **Vẫn dùng Depth Anything V2 Metric** | Không đổi model. Dùng bản ONNX 350 int8 |
| 2 | **Dùng ONNX int8 cho YOLO** | 3.0 MB thay vì 10.6 MB |
| 3 | **Phải xuất APK để test thủ công** | Báo lỗi thật, không giả định |
| 4 | **Không tách repo** | `FCW-Android/` nằm trong repo Python: 1 web + 1 mobile |

### Hệ quả

**Depth 350 int8:** 27 MB, đo trên desktop 634 ms. Trên mobile **chưa đo được** —
phải benchmark Phase 4. Nếu quá nặng:

- Tăng `depthInterval` (đã có cơ chế adaptive)
- Xuống ONNX 252 int8 (27 MB, 188 ms desktop) — chấp nhận sai số +78% ở vật thể gần
- Bật `NNAPI`/`XNNPACK` và đo lại

**Sai số +21% ở vật thể gần** của bản 350 → `DepthCalibration` bắt buộc, giữ nguyên
cơ chế Python (một điểm tham chiếu đã biết). Không được bỏ.

**Test thủ công:** bắt buộc có `README` ghi rõ:

- bấm nút phân tích, log tag để đọc lỗi
- cách kiểm tra FPS thật
- `adb logcat -s FCW` để gửi log khi báo lỗi

### Còn tồn đọng

Thiết bị test. Không có máy thật thì mục tiêu "realtime" không kiểm chứng được —
chỉ chứng minh được build + chạy được.

## 10. Toolchain

Đã cài trên máy build:

| Thành phần | Phiên bản |
|---|---|
| JDK | OpenJDK 21.0.8 |
| Android SDK platform | android-35 |
| Build tools | 35.0.0 |
| SDK path | `/home/void/android-sdk` |
| Gradle | qua wrapper (`./gradlew`) |

Khoảng 458 MB. `local.properties` trỏ `sdk.dir`, **không** commit file này.
