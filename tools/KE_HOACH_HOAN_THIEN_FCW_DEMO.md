# Kế hoạch hoàn thiện demo hệ thống phát hiện vật cản và cảnh báo sớm (FCW)

> **Mục tiêu:** hướng dẫn AI lập trình khác kiểm tra và hoàn thiện project hiện có để tạo bản demo ổn định, có thể lặp lại, có số đo hiệu năng đáng tin cậy và trình bày rõ giới hạn an toàn.
>
> **Phạm vi đã biết:** script `tools/live.py` nhận luồng hình ảnh từ camera điện thoại qua mạng, đưa từng frame vào `app.fcw.pipeline.CollisionWarningSystem`, rồi hiển thị frame đã vẽ trong một cửa sổ OpenCV. Pipeline hiện có được mô tả là chịu trách nhiệm phát hiện YOLO, tracking, depth và risk/collision warning.
>
> **Ràng buộc quan trọng:** chưa có toàn bộ repository, `config.yaml` và mã `app/fcw/pipeline.py` trong tài liệu đầu vào này. AI thực hiện cần đọc các file thật trước khi chỉnh sửa. Không được đoán cấu trúc dữ liệu trả về, viết lại pipeline hoặc tuyên bố tính năng đã hoạt động nếu chưa kiểm thử.

---

## 1. Kết luận đánh giá hiện trạng

`tools/live.py` là một nguyên mẫu demo hợp lý vì có vai trò điều phối rõ ràng: nhận camera → gọi pipeline đang có → hiển thị kết quả → hiển thị một số chỉ số. Tuy nhiên, từ đoạn code được cung cấp, chưa thể xác nhận toàn bộ hệ thống hoạt động đúng vì thiếu các thành phần được import như `CollisionWarningSystem`, `load_config` và file `config.yaml`.

### Điểm tốt

- Không nhân bản logic phát hiện, tracking, depth và cảnh báo; gọi `pipe.process_frame(frame, now)`.
- Có hỗ trợ giảm chiều rộng khung hình để giảm chi phí xử lý.
- Có hiển thị thời gian YOLO và depth nếu pipeline trả các trường `yolo_ms`, `depth_ms`.
- Có hiển thị khung kết quả và phím `q`/`Esc` để thoát.
- Có chủ ý cố định thư mục gốc project để các đường dẫn tương đối dễ hoạt động hơn.

### Vấn đề cần khắc phục trước demo

| Mức độ | Vấn đề | Hệ quả | Hướng xử lý |
|---|---|---|---|
| P0 | URL được viết dưới dạng Markdown link bên trong chuỗi | Nếu đúng như đoạn trích trong file thật, URL truyền cho `cv2.VideoCapture` không hợp lệ | Truyền URL thuần, ví dụ `http://192.168.1.42:8080/video`; không để cú pháp `[text](url)` trong chuỗi |
| P0 | URL ở docstring, URL mặc định và URL người chạy truyền vào không thống nhất | Người demo có thể kết nối nhầm IP/luồng | Dùng `argparse`, một nguồn cấu hình URL duy nhất; in URL cuối cùng đã chọn |
| P0 | Chưa có xác nhận kết quả risk/cảnh báo thực tế từ pipeline | Có thể hiển thị detection nhưng chưa chắc cảnh báo đúng | Đọc hợp đồng trả về của `process_frame` và kiểm thử cảnh báo end-to-end |
| P0 | Địa chỉ IP công khai xuất hiện trong lệnh ví dụ | Có thể vô tình phơi luồng camera ra Internet nếu port được mở | Ưu tiên LAN/VPN; không để IP công khai cố định; kiểm tra xác thực và firewall |
| P1 | Công thức FPS trượt đang dùng `len(times)/(last-first)` | FPS cửa sổ trượt bị cao hơn công thức chuẩn do đếm N mốc thời gian thay vì N-1 khoảng | Dùng `(N-1)/(t_last-t_first)` hoặc đo thông lượng bằng đồng hồ đơn điệu |
| P1 | Không xử lý reconnect khi luồng bị ngắt | Demo dừng khi camera chớp tắt hoặc Wi-Fi chập chờn | Thêm cơ chế thử nối lại có giới hạn và thông báo trạng thái rõ ràng |
| P1 | Chưa bảo đảm `cap.release()` và `destroyAllWindows()` luôn chạy khi ngoại lệ | Tài nguyên camera/cửa sổ có thể không được giải phóng | Dùng `try/finally`; xử lý `KeyboardInterrupt`/ngoại lệ phù hợp |
| P1 | Kích thước lấy từ `cap.get()` có thể bằng 0 với một số backend luồng mạng | Không giảm kích thước đúng hoặc hiển thị thông tin sai | Lấy `height, width` từ frame đầu tiên đã giải mã; dùng giá trị đó làm nguồn đáng tin cậy |
| P1 | `parse_width()` tự tìm số nguyên trong tất cả đối số | Khó kiểm soát cách gọi, thiếu kiểm tra chiều rộng dương và thông báo lỗi | Thay bằng `argparse` với `--url`, `--width`, `--config`, `--reconnect` |
| P1 | Không phân biệt thoát chủ động với luồng lỗi | Log kết thúc có thể khiến người vận hành hiểu sai | Log reason kết thúc; trả mã thoát phù hợp nếu nguồn video hỏng |
| P1 | `res` chỉ in keys một lần, nhưng chưa hiển thị trạng thái cảnh báo có ngữ nghĩa | Giao diện demo có thể thiếu thông tin quan trọng | Kiểm tra schema thật rồi hiển thị trạng thái warning/risk và object fields sẵn có, không tự đoán tên khóa |
| P2 | Dùng `time.time()` cho đo khoảng thời gian | Đồng hồ hệ thống có thể bị điều chỉnh | Dùng `time.perf_counter()`/`time.monotonic()` cho benchmark; chỉ thay timestamp truyền vào pipeline sau khi xác minh pipeline yêu cầu gì |
| P2 | Một dòng overlay gộp FPS, YOLO và depth | Chữ có thể quá dài, khó đọc | Chia overlay thành nhiều dòng, nền tương phản và kích thước phù hợp |
| P2 | Chưa có log benchmark và thống kê p50/p95 | Khó so sánh các lần demo hoặc cấu hình | Ghi kết quả CSV/JSON, thời gian xử lý, FPS và dropped/reconnect frames nếu đo được |

**Lưu ý cú pháp:** đoạn code người dùng gửi có các biểu hiện escape/format như `**ROOT**`, `\_\_file\_\_`, `app.fcw\.config_loader` và URL dạng Markdown. Có thể đây là lỗi định dạng khi sao chép vào chat. AI triển khai phải mở file Python thật để xác định nội dung thực tế, không được tự kết luận file chắc chắn có lỗi cú pháp nếu chưa xem file.

---

## 2. Nguyên tắc thiết kế bản demo

1. **Giữ `CollisionWarningSystem` làm nguồn chân lý cho logic nghiệp vụ.** Không tái viết YOLO, tracking, depth hoặc thuật toán risk trong `tools/live.py`.
2. **Tách input camera, pipeline và UI/telemetry.** Script live chỉ điều phối các thành phần này.
3. **Ưu tiên tin cậy trước tính năng mới.** Một demo ổn định, có thể giải thích số đo sẽ tốt hơn một demo nhiều overlay nhưng khó xác minh.
4. **Không bịa trường dữ liệu.** Chỉ sử dụng các khóa và kiểu dữ liệu đã xác nhận từ `process_frame` hoặc tài liệu trong repository.
5. **Không tuyên bố hệ thống bảo đảm tránh va chạm.** Đây là nguyên mẫu hỗ trợ nhận biết/cảnh báo; hiệu năng và giới hạn cần được đo và nêu rõ.
6. **Giữ nguyên các thay đổi nhỏ, có kiểm thử.** Không refactor rộng toàn repository khi chưa cần thiết.
7. **Bảo vệ luồng camera.** Mặc định ưu tiên mạng nội bộ; không ghi thông tin nhạy cảm trong log hoặc tài liệu chia sẻ.

---

## 3. Kế hoạch triển khai theo từng giai đoạn

## Giai đoạn 0 — Kiểm tra repository và dựng baseline

### Nhiệm vụ

- Liệt kê cấu trúc repository và các dependency.
- Mở và đọc ít nhất:
  - `tools/live.py`
  - `app/fcw/pipeline.py`
  - `app/fcw/config_loader.py`
  - `config.yaml`
  - module detector/YOLO được pipeline gọi
  - module tracking, depth estimation và risk/collision warning được pipeline gọi
  - README hoặc hướng dẫn chạy hiện có
- Ghi lại schema thực tế của `CollisionWarningSystem.process_frame`:
  - Kiểu dữ liệu đầu vào.
  - Kiểu và số phần tử trả về.
  - Ý nghĩa của `result[0]` và `result[1]`.
  - Những khóa thực có trong `res` và đơn vị của chúng.
  - Timestamp đầu vào được dùng để làm gì.
- Kiểm tra phiên bản Python, OpenCV, Ultralytics/PyTorch và cách tải model.
- Chạy script hiện tại với camera có thể truy cập được và ghi nhận lỗi/log, không sửa logic trước khi có baseline.
- Kiểm tra trạng thái Git và tạo branch/commit an toàn trước khi chỉnh sửa.

### Đầu ra bắt buộc

- `docs/demo_baseline.md` hoặc báo cáo trong phản hồi AI: cấu trúc, lệnh chạy, lỗi đã tái hiện, schema `process_frame` và cấu hình máy.
- Bảng phân biệt: lỗi đã xác nhận, rủi ro cần kiểm tra, giả định chưa được xác nhận.

### Điều kiện hoàn thành

AI có thể giải thích chính xác luồng từ `VideoCapture.read()` tới kết quả `process_frame()` và biết rõ cách pipeline biểu diễn warning/risk.

---

## Giai đoạn 1 — Sửa đường dẫn, CLI và kết nối camera (P0)

### Nhiệm vụ

- Dùng `argparse`, không tự quét đối số bằng `parse_width()`.
- Hỗ trợ tối thiểu:
  - `--url`: URL luồng camera, có thể bắt buộc hoặc lấy từ biến môi trường/cấu hình riêng.
  - `--width`: chiều rộng đầu ra mong muốn; chỉ nhận số nguyên dương.
  - `--config`: đường dẫn file cấu hình, mặc định trỏ đến `config.yaml` ở root.
  - `--reconnect`: bật/tắt reconnect hoặc đặt số lần thử.
  - `--reconnect-delay`: thời gian chờ giữa hai lần kết nối, có giới hạn hợp lý.
  - `--help`: hiển thị trợ giúp và thoát, không khởi tạo model.
- URL phải là URL thuần, không phải Markdown link.
- Không hard-code IP công khai. Có thể yêu cầu người chạy truyền `--url`, hoặc dùng ví dụ LAN trong README với nhãn placeholder.
- Kiểm tra URL rỗng, scheme không hỗ trợ và `width <= 0`.
- Sau khi nhận frame đầu tiên, đọc kích thước thật bằng `frame.shape`.
- Nếu không mở được nguồn, hiển thị hướng dẫn cụ thể: kiểm tra app camera đang phát, IP/port/path, Wi-Fi và quyền truy cập.

### Gợi ý giao diện CLI

```bash
python tools/live.py --url "http://192.168.1.42:8080/video" --width 960
```

Địa chỉ trên chỉ là ví dụ LAN. Người triển khai phải thay bằng URL thực của điện thoại. Không commit IP công khai, mật khẩu hoặc token.

### Điều kiện hoàn thành

- `python tools/live.py --help` hoạt động mà không tải model.
- URL hợp lệ có thể mở luồng camera.
- URL không hợp lệ trả thông báo dễ hiểu và mã thoát khác 0.
- Kích thước ảnh sau resize đúng giới hạn mong muốn và giữ tỷ lệ khung hình.

---

## Giai đoạn 2 — Vòng lặp frame ổn định và giải phóng tài nguyên (P0/P1)

### Nhiệm vụ

- Bọc tài nguyên trong `try/finally` để luôn gọi `cap.release()` và `cv2.destroyAllWindows()`.
- Xử lý có chủ ý các trường hợp:
  - Người dùng nhấn `q`/`Esc`.
  - Camera phát frame lỗi hoặc `None`.
  - Pipeline ném exception.
  - Người dùng nhấn Ctrl+C.
- Khi mất nguồn, thử reconnect với thời gian chờ và số lần thử giới hạn; log số lần thử.
- Tránh vòng lặp reconnect bận liên tục.
- Không dùng frame cũ để phát một cảnh báo mới nếu dữ liệu đầu vào đã stale.
- Không thêm luồng/thread bất đồng bộ trước khi đo chứng minh đang có vấn đề buffer/latency.
- Nếu phát hiện độ trễ luồng tăng do backlog, thiết kế một chế độ “latest frame” ở tầng capture (buffer 1 frame), nhưng phải bảo đảm pipeline vẫn được gọi tuần tự nếu nó giữ state tracking.

### Điều kiện hoàn thành

- Thoát bằng phím tắt luôn đóng cửa sổ và giải phóng camera.
- Camera rớt kết nối có thông báo; reconnect đúng giới hạn.
- Exception có log đủ để gỡ lỗi và không bị nuốt âm thầm.
- Không làm đổi thứ tự frame đưa vào pipeline nếu tracking phụ thuộc temporal state.

---

## Giai đoạn 3 — Đo FPS và độ trễ có ý nghĩa (P1)

### Phân biệt các đại lượng

- **YOLO inference time:** thời gian suy luận của detector, chỉ lấy từ `res` nếu đơn vị và ý nghĩa đã được xác minh.
- **Depth time:** thời gian xử lý depth, nếu pipeline thực sự trả về giá trị này.
- **Pipeline time:** thời gian gọi hoàn tất `process_frame()`.
- **Loop throughput:** số frame đã xử lý chia cho thời gian chạy.
- **Sliding FPS:** tốc độ trong cửa sổ thời gian gần nhất.
- **Camera-to-display latency:** thời gian từ lúc frame được tạo/đánh dấu timestamp tại nguồn đến lúc hiển thị. Không thể tuyên bố đo đại lượng này chính xác nếu timestamp tại nguồn không được cung cấp hoặc chưa hiệu chỉnh đồng hồ.
- **Warning latency:** thời gian từ lúc pipeline xác định warning đến khi ứng dụng hiển thị/phát cảnh báo. Cần instrument chính xác các mốc, không suy ra từ FPS.

### Nhiệm vụ

- Dùng `time.perf_counter()` hoặc `time.monotonic()` để đo duration.
- Nếu có `N` timestamp trong một cửa sổ, công thức FPS trượt là `(N - 1) / (t_last - t_first)` với `N >= 2` và mẫu số dương.
- Tính thời gian pipeline mỗi frame bằng `end - start`.
- Làm rõ timestamp `now` truyền cho `process_frame`: đọc code pipeline để xác nhận nó cần wall-clock hay elapsed/monotonic time; không đổi tùy tiện vì có thể ảnh hưởng tracker/risk.
- Tính trung bình theo cửa sổ và lưu được p50/p95 sau một đợt đo đủ dài.
- Nếu pipeline trả `yolo_ms`/`depth_ms`, xác minh giá trị là milliseconds và không cộng lặp thời gian.
- Đặt tên overlay rõ ràng, ví dụ `Loop FPS`, `Pipeline ms`, `YOLO ms`, `Depth ms`; chỉ hiển thị chỉ số có dữ liệu thật.
- Xuất CSV benchmark có timestamp, source resolution, processed resolution, loop fps, pipeline ms và các trường đã xác minh.

### Điều kiện hoàn thành

- Công thức số đo có chú thích.
- Có thể chạy benchmark lặp lại với cùng nguồn/cấu hình.
- Không ghi nhãn “end-to-end latency” nếu chỉ mới đo thời gian model/pipeline.

---

## Giai đoạn 4 — Xác nhận detection, tracking, depth và cảnh báo (P0)

### Nhiệm vụ

- Đọc mã của `CollisionWarningSystem` và các module liên quan.
- Vẽ sơ đồ luồng thực tế: frame → detector → tracker → depth → risk logic → annotated frame/warning output.
- Xác nhận hệ thống thực sự thực hiện những gì trong danh sách trên; nếu một module là placeholder hoặc chưa dùng, ghi rõ.
- Xác định schema kết quả trả về và tài liệu hóa bằng `TypedDict`/dataclass nếu phù hợp, nhưng không đổi API rộng nếu không cần thiết.
- Kiểm tra đơn vị depth. Nếu là monocular relative depth, không hiển thị như mét hoặc khoảng cách thực nếu chưa hiệu chuẩn.
- Xác minh ngưỡng warning/risk nằm trong cấu hình hay hard-code; tránh có nhiều nguồn giá trị mâu thuẫn.
- Xác minh cảnh báo không kích hoạt liên tục không kiểm soát trên cùng một đối tượng; nếu thêm debounce/cooldown thì đưa vào module nghiệp vụ/config phù hợp, không nhét logic risk vào `live.py`.
- Đánh giá trường hợp mất detection trong vài frame, object ID đổi, vật thể bị che, camera rung và ánh sáng biến đổi.
- Bảo đảm UI chỉ trình bày đúng trạng thái mà pipeline đã tính, không tự suy diễn risk từ tên lớp.

### Điều kiện hoàn thành

- Có mô tả schema output.
- Ít nhất có test/smoke test cho đường đi warning từ input đến output.
- Các giá trị depth/risk có đơn vị/ý nghĩa được ghi rõ hoặc được đánh dấu chưa hiệu chuẩn.
- `tools/live.py` không chứa bản sao của logic collision-risk.

---

## Giai đoạn 5 — Hoàn thiện UI dành cho demo (P1/P2)

### Nhiệm vụ

- Đặt tên cửa sổ dễ nhận diện, ví dụ `Road Obstacle & FCW Demo`.
- Overlay theo nhiều dòng, không đè lên đối tượng quan trọng:
  - FPS/throughput.
  - Thời gian pipeline.
  - YOLO/depth duration khi có.
  - Trạng thái pipeline/camera.
  - Warning/risk status chỉ khi lấy được từ schema thật.
- Chọn cách trình bày màu sắc nhất quán; cảnh báo cần rõ nhưng không che phần lớn khung hình.
- Hiển thị resolution nguồn và resolution sau resize.
- Khi reconnect, hiện trạng thái “Reconnecting” để người xem không tưởng rằng hệ thống vẫn nhận frame mới.
- Cho phép tắt overlay hoặc chỉ số phụ bằng CLI nếu thực sự cần.
- Tránh quá nhiều chữ nhỏ; thử ở độ phân giải màn hình demo thực tế.

### Điều kiện hoàn thành

Người xem có thể phân biệt trạng thái camera, detection, risk/warning và chỉ số hiệu năng mà không cần đọc terminal liên tục.

---

## Giai đoạn 6 — Đánh giá mô hình và kịch bản demo (P0/P1)

### Bối cảnh cần lưu ý

Kết quả YOLO11n đã được chia sẻ trước đó có mAP@50 xấp xỉ 72,1%; một số AP@50 theo lớp thấp hơn, chẳng hạn `dog` khoảng 56,1%, `bicycle` 62,7% và `pothole` 63,2%. Vì vậy, demo phải chọn kịch bản đại diện và không tuyên bố độ tin cậy cao cho mọi vật thể. Nếu đã có lần huấn luyện mới sau khi sửa nhãn, cập nhật các số liệu theo checkpoint thật được dùng trong demo.

### Nhiệm vụ

- Khóa rõ checkpoint model được dùng (đường dẫn, kích thước file, hash hoặc version tag).
- Ghi cấu hình, confidence threshold, input resolution và các tham số liên quan.
- Chọn video/cảnh demo có ánh sáng và khoảng cách khác nhau, đồng thời có các lớp dễ và khó.
- Tạo bộ demo cố định và ghi lại những tình huống đúng, bỏ sót, false positive và cảnh báo không hợp lý.
- Không lựa chọn riêng những frame đẹp rồi dùng làm bằng chứng đại diện cho cả hệ thống.
- Đánh giá AP theo lớp trên tập test độc lập, không chỉ ảnh quay trong buổi demo.
- Phân biệt rõ chỉ số mô hình (mAP) với chỉ số vận hành (FPS/latency) và chỉ số cảnh báo (false alarms/missed warnings).
- Nếu tuyên bố phát hiện va chạm hoặc khoảng cách an toàn, cần định nghĩa ground truth phù hợp và quy trình đánh giá riêng; object detection đơn thuần không chứng minh được điều đó.

### Điều kiện hoàn thành

Có báo cáo ngắn ghi rõ model version, nguồn dữ liệu, kịch bản đã thử, giới hạn quan sát và kết quả đo. Không dùng số liệu ước đoán.

---

## Giai đoạn 7 — Kiểm thử, tài liệu và đóng gói demo (P1)

### Nhiệm vụ

- Thêm unit test nhỏ cho `parse_args`/validation nếu tiện.
- Thêm test FPS với các timestamp giả lập để xác nhận công thức.
- Mock pipeline để kiểm thử UI/loop mà không cần tải model thật nếu kiến trúc hiện tại cho phép.
- Thêm smoke test chạy một video mẫu từ file, để có thể demo khi camera Wi-Fi không ổn định.
- Thêm test luồng camera bị ngắt và reconnect.
- Viết `README_DEMO.md` gồm cài đặt, cách khởi động camera điện thoại, cách xác định URL, lệnh chạy, phím thoát, khắc phục lỗi và cách chạy video dự phòng.
- Thêm `.env.example` hoặc file config ví dụ nếu cần; không commit IP công khai, mật khẩu camera, API token hay khóa cá nhân.
- Ghi phiên bản Python và dependency đã được kiểm thử.

### Điều kiện hoàn thành

Một người khác có thể làm theo README để chạy demo từ đầu, biết URL nào phải thay, biết cách thoát và có phương án dự phòng khi mất Wi-Fi.

---

## 4. Kịch bản kiểm thử nghiệm thu

| ID | Tình huống | Kết quả mong đợi |
|---|---|---|
| T01 | Chạy `--help` | In trợ giúp rồi thoát; không tải model hoặc mở camera |
| T02 | URL hợp lệ | Mở luồng, lấy frame và xử lý qua pipeline |
| T03 | URL sai/stream không tồn tại | Báo lỗi rõ ràng, không treo |
| T04 | `--width 960` | Giữ tỷ lệ khung hình và giảm chiều rộng xuống tối đa theo yêu cầu |
| T05 | `--width 0` hoặc âm | Báo lỗi tham số |
| T06 | Nhấn `q`/`Esc` | Thoát sạch, giải phóng tài nguyên |
| T07 | Camera ngắt trong khi chạy | Hiện mất kết nối; reconnect trong giới hạn hoặc kết thúc có kiểm soát |
| T08 | Pipeline phát sinh lỗi | Log có traceback phù hợp; tài nguyên được giải phóng |
| T09 | Không có khóa `yolo_ms`/`depth_ms` | UI vẫn chạy, hiện `N/A` hoặc ẩn chỉ số; không crash |
| T10 | Cửa sổ chạy ít nhất 10 phút | Không tăng bộ nhớ vô hạn, FPS và telemetry tiếp tục cập nhật |
| T11 | Một số tình huống warning đã biết | UI phản ánh đúng trạng thái trả về từ pipeline |
| T12 | Chạy video dự phòng | Có thể demo detector/pipeline khi camera Wi-Fi không khả dụng |
| T13 | Kiểm tra network | Không cần mở camera điện thoại công khai ra Internet |
| T14 | Benchmark lặp lại | Có CSV và cấu hình đủ để so sánh các lần chạy |

---

## 5. Đề xuất tổ chức code sau khi hoàn thiện

Không cần tách nhiều file nếu project nhỏ. Tổ chức tối thiểu được khuyến nghị:

```text
project-root/
├── app/
│   └── fcw/
│       ├── pipeline.py          # Giữ logic nghiệp vụ hiện có
│       ├── config_loader.py
│       └── ...                  # Detector/tracker/depth/risk hiện có
├── tools/
│   └── live.py                  # CLI, camera source, loop, display
├── config.yaml
├── tests/
│   ├── test_live_args.py
│   ├── test_fps_metrics.py
│   └── ...
├── docs/
│   ├── demo_baseline.md
│   └── README_DEMO.md
└── runs/                         # Kết quả benchmark; tránh commit file lớn
```

Chỉ tạo helper module riêng cho camera/benchmark nếu sau khi đọc project thấy cần thiết. Đừng tạo lại detector, depth model hay risk engine chỉ để làm demo.

---

## 6. Giao việc cho AI lập trình khác

Hãy thực hiện theo thứ tự các giai đoạn trên và tuân theo các quy tắc sau:

1. Bước đầu tiên chỉ đọc repository và báo cáo cấu trúc, hợp đồng API cùng những lỗi tái hiện được. Không thay đổi file ở bước này.
2. Tạo checklist với các lỗi P0/P1/P2 và liên kết mỗi thay đổi với một kiểm thử nghiệm thu.
3. Sửa P0 trước: URL/CLI, kiểm chứng schema pipeline, bảo vệ luồng camera và xử lý lỗi tài nguyên.
4. Sửa P1 kế tiếp: reconnect, resize an toàn, đo FPS/latency có định nghĩa rõ và hiển thị trạng thái.
5. Giữ nguyên hành vi nghiệp vụ của `CollisionWarningSystem`. Nếu bắt buộc sửa pipeline, mô tả lý do, thay đổi API và ảnh hưởng trước khi áp dụng.
6. Không đoán tên trường trong `res`, không tuyên bố depth là mét nếu chưa xác minh, không tự đặt ngưỡng nguy hiểm không có cơ sở.
7. Không hard-code URL camera, credential hoặc IP công khai. Dùng CLI/config/local environment.
8. Sau mỗi thay đổi, chạy test liên quan và báo cáo lệnh thực tế đã chạy cùng kết quả. Nếu không thể chạy do thiếu camera/model/dependency, ghi rõ là chưa kiểm chứng.
9. Không tự nâng cấp thư viện toàn project nếu không cần thiết. Ghi lại thay đổi dependency nếu có.
10. Kết thúc bằng báo cáo: file đã sửa, lỗi đã khắc phục, test pass/fail, lệnh chạy demo, chỉ số hiệu năng đã đo thật và các hạn chế còn tồn tại.

### Tiêu chí hoàn thành toàn dự án

- Demo khởi động bằng một lệnh documented.
- URL và kích thước frame được cấu hình rõ ràng.
- Camera failure và pipeline error có xử lý.
- Tài nguyên luôn được giải phóng.
- Các số FPS/latency có định nghĩa và được đo đúng.
- Trạng thái cảnh báo hiển thị đúng theo pipeline đã xác minh.
- Có video dự phòng, README và kịch bản test.
- Không có tuyên bố an toàn/khả năng tránh va chạm vượt quá bằng chứng đo được.
