# FCW Demo — Hướng dẫn chạy

## Yêu cầu

- Python 3.12+ với môi trường ảo `.venv` đã cài dependency
- Điện thoại cài **IP Webcam** (Android) cùng mạng Wi-Fi với máy tính
- Hoặc một file video để chạy offline

## Cài đặt

```bash
# Cài dependency (nếu chưa)
pip install -r requirements-fcw.txt
```

## Cách xác định URL camera

1. Mở ứng dụng **IP Webcam** trên điện thoại
2. Bấm **Start Server**
3. Ghi lại địa chỉ hiển thị, ví dụ: `http://192.168.1.42:8080`
4. URL luồng video sẽ là: `http://192.168.1.42:8080/video`

> **Quan trọng**: Thay IP bằng IP thực của điện thoại. Không dùng IP công khai.

## Lệnh chạy

```bash
# Chạy với camera điện thoại
python tools/live.py --url "http://192.168.1.42:8080/video"

# Giảm resolution để tăng tốc
python tools/live.py --url "http://192.168.1.42:8080/video" --width 960

# Chạy với file video (dự phòng khi không có camera)
python tools/live.py --url "path/to/video.mp4"

# Ghi benchmark CSV
python tools/live.py --url "video.mp4" --benchmark

# Xem tất cả tùy chọn
python tools/live.py --help
```

## Phím tắt

| Phím    | Hành động |
|---------|-----------|
| `q`     | Thoát     |
| `Esc`   | Thoát     |
| `Ctrl+C`| Thoát (terminal) |

## Overlay hiển thị

Cửa sổ demo hiển thị các thông tin:
- **Loop FPS**: tốc độ xử lý thực tế
- **Pipeline ms**: thời gian xử lý mỗi frame
- **YOLO / Depth ms**: thời gian suy luận từng module
- **Source / Processed resolution**: kích thước ảnh gốc và sau resize
- **Level**: trạng thái cảnh báo (NORMAL / WARNING / DANGER)
- **Primary target**: vật thể chính đang theo dõi, khoảng cách và TTC

## Reconnect

Khi mất kết nối camera, hệ thống sẽ tự thử kết nối lại (mặc định 5 lần, mỗi lần cách 2 giây).

```bash
# Tùy chỉnh reconnect
python tools/live.py --url "http://..." --reconnect 10 --reconnect-delay 3
# Tắt reconnect
python tools/live.py --url "http://..." --reconnect 0
```

## Khắc phục lỗi

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| "Cannot open camera/video" | IP sai hoặc camera chưa bật | Kiểm tra IP Webcam đã Start Server |
| Ảnh đen / lag | Wi-Fi yếu | Di chuyển gần router |
| FPS thấp | CPU yếu | Giảm `--width`, hoặc tăng `depth.interval` trong config.yaml |
| Không có cửa sổ | Headless mode | Bỏ `--no-display` |

## Benchmark

Khi dùng `--benchmark`, file CSV sẽ được lưu trong `runs/` với tên dạng `live_benchmark_YYYYMMDD_HHMMSS.csv`, chứa:
- Frame index, timestamp
- Resolution nguồn và xử lý
- FPS, pipeline/YOLO/depth timing
- Số detection, tracks
- Warning level, primary target info

## Giới hạn

- Đây là **nguyên mẫu demo**, không phải hệ thống ADAS sản phẩm
- Độ trễ xử lý trên CPU quá cao để dùng làm cảnh báo an toàn thực tế
- Khoảng cách depth chỉ là ước lượng từ monocular depth, cần hiệu chuẩn
- Kết quả phụ thuộc điều kiện ánh sáng, góc camera và tốc độ di chuyển
