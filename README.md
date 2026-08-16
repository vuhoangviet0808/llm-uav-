# UAV Point-to-Point Pathfinding with a Fine-Tuned Small LLM

Bài toán UAV đơn giản nhất từ paper "Learning to Recharge": bỏ hết coverage
target, pin, sạc, chỉ còn một UAV bay từ ô `S` đến ô `G` trên lưới NxN,
tránh vật cản/no-fly-zone `#`. BFS cho biết đường đi ngắn nhất tuyệt đối
(ground truth), dùng để (1) sinh dữ liệu fine-tuning và (2) làm mốc so sánh
khi chấm điểm model, y hệt cách paper gốc dùng Greedy Heuristic + RPD để so
với agent RL.

Ý tưởng: thay vì train PPO, ta lấy một LLM nhỏ (~1B tham số, ví dụ
Llama-3.2-1B), fine-tune bằng LoRA trên các cặp (bản đồ dạng text, chuỗi
nước đi tối ưu từ BFS), rồi đánh giá xem model tự suy luận đường đi trên lưới
mới (chưa thấy khi train) tốt tới đâu.

## Cấu trúc

```
uav-llm-pathfinding/
  configs/
    default.yaml       # config chạy thật: model ~1B + GPU
    smoke_test.yaml     # config chạy thử: model tí hon, CPU, offline
  src/
    env.py              # lưới NxN, BFS optimal path, mô phỏng/kiểm tra 1 chuỗi nước đi
    prompts.py           # text hoá lưới -> prompt; parse output model -> chuỗi nước đi
    data_gen.py          # sinh train/val/test.jsonl từ BFS
    tokenizer.py          # get_tokenizer(): HF AutoTokenizer thật, hoặc tokenizer ký tự
                           # tí hon (tiny-debug) để chạy thử không cần internet
    model.py               # load base model + gắn LoRA (peft), tự đoán target_modules
    sft_dataset.py           # dataset PyTorch: mask loss phần prompt, chỉ học phần completion
    train.py                  # vòng lặp fine-tune (transformers.Trainer)
    evaluate.py                 # sinh nước đi, mô phỏng, tính success_rate / length_ratio / ...
    visualize.py                 # vẽ 1 lưới + đường đi optimal vs model ra PNG
  scripts/
    smoke_test.sh          # chạy thử toàn bộ pipeline, tí hon, CPU, không cần mạng
    run_full_pipeline.sh     # chạy thật với model 1B, cần GPU + internet
  colab_quickstart.ipynb  # notebook clone repo -> cài đặt -> train -> eval trên Colab
  requirements.txt
  .gitignore              # data/ và outputs/ không commit (chỉ giữ .gitkeep)
```

## 1. Chạy thử nhanh (bắt buộc làm trước) — không cần GPU, không cần mạng

Bước này KHÔNG dùng model 1B thật, chỉ dùng một model GPT-2-tí-hon khởi tạo
ngẫu nhiên + tokenizer ký tự tự chế, để kiểm tra toàn bộ pipeline (sinh dữ
liệu -> tokenize -> LoRA -> train -> generate -> parse -> chấm điểm) chạy
đúng cơ chế, không lỗi, trước khi đụng vào model thật tốn thời gian/tiền GPU.

```bash
pip install -r requirements.txt
bash scripts/smoke_test.sh
```

Đã tự chạy thử toàn bộ script này (mất ~5 phút trên CPU): loss giảm từ ~4.4
(gần random) xuống ~0.68, và `unparseable_rate` giảm từ 100% (model chưa
train) xuống 0% sau fine-tune — chứng minh cơ chế "mask loss theo
completion -> generate -> parse -> chấm điểm" chạy đúng. `success_rate` trên
scenario chưa từng thấy vẫn thấp/0% ở quy mô tí hon này (model 3 lớp/96
chiều, vài trăm mẫu) — bình thường, vì bài test này không đủ capacity để
tổng quát hoá suy luận không gian; nó chỉ chứng minh pipeline không có bug,
chưa nói lên gì về khả năng thật của LLM 1B.

## 2. Chạy thật với LLM ~1B tham số (cần máy có GPU + internet)

Sửa `configs/default.yaml` nếu muốn đổi model, kích thước lưới, độ dày vật
cản, số lượng mẫu train... rồi:

```bash
pip install -r requirements.txt
bash scripts/run_full_pipeline.sh configs/default.yaml
```

Script sẽ: sinh dataset -> đánh giá model gốc (chưa fine-tune) làm baseline
zero-shot -> fine-tune LoRA -> đánh giá lại model đã fine-tune -> so sánh 2
báo cáo JSON (`outputs/eval_report_base.json` vs
`outputs/eval_report_finetuned.json`).

Model mặc định trong `default.yaml` là `Qwen/Qwen2.5-1.5B-Instruct` —
model mở hoàn toàn (Apache 2.0), chạy được ngay, không cần xin quyền hay
token gì cả. Đây là lựa chọn nên dùng để chạy thử nhanh.

Nếu vẫn muốn dùng đúng `meta-llama/Llama-3.2-1B-Instruct` (~1B, đúng như đề
xuất ban đầu): model này bị Meta "gate" trên Hugging Face, sẽ báo lỗi
`403 GatedRepoError` cho tới khi bạn (1) đăng nhập huggingface.co, vào
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct bấm "Agree and
access repository" (Meta duyệt thủ công, có thể ngay lập tức hoặc mất vài
giờ), (2) tạo token tại huggingface.co/settings/tokens, và (3) trong Colab
thực sự chạy ô `from huggingface_hub import login; login()` với token đó
*trước khi* chạy data_gen/train/evaluate. Sau đó đổi lại `model.name` trong
config. Nếu chỉ muốn thử ngay không đợi duyệt quyền, sửa nhanh 1 dòng bằng:
`!sed -i 's|Qwen/Qwen2.5-1.5B-Instruct|meta-llama/Llama-3.2-1B-Instruct|' configs/default.yaml`
trong 1 ô Colab, sau khi đã login.

LoRA r=16/alpha=32 trên 1B tham số chạy được trên GPU ~8GB VRAM, kể cả GPU T4
miễn phí của Colab (config mặc định đã để `fp16: true, bf16: false` vì T4
không hỗ trợ tốt bf16 — nếu chạy trên A100/GPU đời mới hơn thì đảo lại). Nếu
máy không có GPU, train sẽ rất chậm trên CPU.

## Đẩy lên GitHub và chạy trên Google Colab

1. `git init && git add -A && git commit -m "init"` trong thư mục này, tạo
   repo trên GitHub rồi `git push`. `.gitignore` đã loại `data/`, `outputs/`
   (dữ liệu/model sinh ra lúc chạy, không cần commit).
2. Mở `colab_quickstart.ipynb` trên Colab (Colab -> GitHub -> dán link
   notebook trong repo, hoặc File -> Upload notebook), sửa biến `REPO_URL` ở
   ô đầu tiên thành link repo vừa tạo.
3. Runtime -> Change runtime type -> chọn GPU (T4 miễn phí là đủ).
4. Chạy lần lượt từng ô: clone repo -> cài `requirements.txt` -> (nếu dùng
   Llama-3.2-1B-Instruct thì đăng nhập Hugging Face để qua "gate") -> sinh
   dữ liệu -> đánh giá model gốc -> fine-tune LoRA -> đánh giá model đã
   fine-tune -> so sánh -> vẽ hình 1 scenario.

## 3. Xem trực quan 1 scenario

```bash
python -m src.visualize --config configs/default.yaml --split test --index 0 \
    --eval_report outputs/eval_report_finetuned.json --out outputs/example.png
```

Vẽ lưới, đường BFS-optimal (xanh) và đường model thực sự bay (đỏ, lấy từ
`model_moves` đã lưu trong eval report) chồng lên nhau.

## Các chỉ số đánh giá (giống tinh thần RPD trong paper gốc)

- `success_rate`: tỉ lệ scenario model bay tới đúng G mà không đâm vật cản/ra
  khỏi lưới.
- `avg_length_ratio_on_success`: (số bước model dùng) / (số bước BFS tối ưu),
  chỉ tính trên các scenario thành công — 1.0 nghĩa là bằng tối ưu, càng lớn
  càng lãng phí quãng đường, y hệt cách RPD so agent với heuristic trong
  paper.
- `invalid_move_rate`: tỉ lệ model chọn 1 nước đi phạm luật (đâm `#` hoặc ra
  ngoài lưới) trước khi tới G.
- `unparseable_rate`: tỉ lệ output của model không tách được nước đi hợp lệ
  nào — dấu hiệu model chưa học được đúng định dạng output.

## Mở rộng tiếp (nếu muốn quay lại gần bài toán gốc hơn)

- Thêm pin (`budget`) và action `charge`/`land`/`take off` như trong
  `GridGym.Params` của repo uavSim -> quay lại đúng bài CPP-with-recharge.
- Thêm target zone cần quét thay vì chỉ 1 điểm G -> quay lại bài coverage.
- So sánh thêm với Greedy Heuristic thật của uavSim (`src/base/heuristics.py`)
  thay vì chỉ so với BFS, một khi đã thêm pin/coverage.
