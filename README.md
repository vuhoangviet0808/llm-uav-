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

Model mặc định trong `default.yaml` là `meta-llama/Llama-3.2-1B-Instruct`
(đúng ~1B tham số như bạn muốn). Model này bị "gate": cần vào
huggingface.co/meta-llama/Llama-3.2-1B-Instruct bấm "Agree and access", rồi
đăng nhập bằng token (`huggingface-cli login` hoặc set biến môi trường
`HF_TOKEN`). Nếu không muốn xin quyền, đổi `model.name` trong config sang
một model mở hoàn toàn cùng tầm cỡ, ví dụ `Qwen/Qwen2.5-0.5B-Instruct` hoặc
`HuggingFaceTB/SmolLM2-1.7B-Instruct` — code không cần sửa gì thêm.

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

## So sánh với RL cổ điển (PPO) — baseline thứ 3

Để trả lời câu hỏi "LLM fine-tune có tốt hơn cách làm truyền thống của chính
paper gốc không" (paper gốc dùng PPO), repo có thêm 1 baseline PPO huấn
luyện trực tiếp cho bài point-to-point pathfinding này (không cần GPU, train
được trên CPU trong vài phút vì môi trường rất nhẹ):

```bash
pip install gymnasium stable-baselines3   # đã có trong requirements.txt
python -m src.train_rl --config configs/default.yaml --timesteps 1200000
python -m src.evaluate_rl --config configs/default.yaml \
    --model_path outputs/ppo-pathfinding/ppo_model.zip \
    --report_path outputs/eval_report_ppo.json
```

`src/rl_env.py` định nghĩa 1 Gymnasium environment cho bài toán này. Lưu ý
thiết kế: agent chỉ quan sát 1 cửa sổ cục bộ (mặc định 5x5 ô quanh UAV) +
hướng/khoảng cách chuẩn hoá tới đích, KHÔNG quan sát toàn bộ bản đồ qua CNN
như kiến trúc PPO trong paper gốc — lựa chọn này để train nhanh, đơn giản,
không cần custom CNN feature extractor, nhưng vì vậy đây là một baseline RL
hợp lý chứ không phải tái hiện chính xác kiến trúc "Learning to Recharge".
`src/train_rl.py` train trên đúng các instance trong `data/train.jsonl` (lặp
lại nhiều lần qua hàng trăm nghìn/triệu bước môi trường, vì PPO là on-policy
RL chứ không phải SFT một lần như LLM). `src/evaluate_rl.py` đánh giá trên
đúng `data/test.jsonl`, cùng 4 chỉ số với `src.evaluate` (`unparseable_rate`
luôn = 0 vì PPO luôn xuất ra 1 trong 4 hành động hợp lệ).

**Kết quả thật** (1.2 triệu timesteps, ~5 phút trên CPU, cùng `test.jsonl`
200 mẫu với LLM fine-tuned): `success_rate = 0.78`, `invalid_move_rate =
0.005`, `unparseable_rate = 0.0`, `avg_length_ratio_on_success ≈ 1.01`. PPO
vượt trội rõ rệt so với cả LLM zero-shot (0%) và LLM fine-tuned (35%) trên
bài toán đơn giản này — dễ hiểu vì PPO được huấn luyện *chuyên biệt* hàng
triệu lượt tương tác trực tiếp với đúng phân phối bài toán, còn LLM chỉ học
gián tiếp qua vài nghìn cặp (prompt, completion) tĩnh và phải tổng quát hoá
suy luận không gian từ kiến trúc ngôn ngữ vốn không chuyên cho việc này.
Đây không phải bằng chứng "LLM vô dụng" mà là lời nhắc: với 1 bài toán hẹp,
có simulator rẻ để lấy hàng triệu mẫu, RL cổ điển vẫn là lựa chọn mạnh và rẻ
hơn nhiều so với fine-tune một LLM — điểm mạnh của LLM nằm ở khả năng
tổng quát/diễn giải/kết hợp ngôn ngữ tự nhiên, không phải ở việc thay thế
RL trên chính bài toán RL đã được thiết kế tốt.

## Mở rộng tiếp (nếu muốn quay lại gần bài toán gốc hơn)

- Thêm pin (`budget`) và action `charge`/`land`/`take off` như trong
  `GridGym.Params` của repo uavSim -> quay lại đúng bài CPP-with-recharge.
- Thêm target zone cần quét thay vì chỉ 1 điểm G -> quay lại bài coverage.
- So sánh thêm với Greedy Heuristic thật của uavSim (`src/base/heuristics.py`)
  thay vì chỉ so với BFS, một khi đã thêm pin/coverage.

## Troubleshooting

- **`ImportError: Found an incompatible version of torchao`** khi chạy
  `src.train`: `peft` dò các quantization backend tùy chọn (torchao,
  bitsandbytes...) kể cả khi không dùng tới; nếu môi trường (Colab/Kaggle) có
  sẵn bản torchao cũ, `peft` báo lỗi cứng thay vì bỏ qua. Đã pin
  `torchao>=0.16.0` trong `requirements.txt` để tự sửa khi cài lại từ đầu.
  Fix nhanh không cần cài lại toàn bộ: `!pip install -q -U torchao`.
- **`403 GatedRepoError`** khi tải `meta-llama/Llama-3.2-1B-Instruct`: model
  bị Meta "gate", cần chấp nhận license + đăng nhập bằng token HF trước. Mặc
  định `configs/default.yaml` đã đổi sang `Qwen/Qwen2.5-1.5B-Instruct` (không
  bị gate) để tránh vướng bước này.
- **Cell `evaluate`/`train` trông như bị treo (đứng yên rất lâu, không in gì
  thêm)**: `evaluate.py` đã có thanh tiến trình `tqdm`; `train.py` in ngay
  dòng `torch.cuda.is_available()` + có `StepTimerCallback` in thời gian của
  5 step đầu (không phụ thuộc `logging_steps`). Nếu log không hề nhích số dù
  đã chờ rất lâu (hàng chục phút tới hàng giờ), gần như chắc chắn là **không
  có GPU thật sự** trong phiên chạy đó, không phải một lỗi/deadlock.
- **Notebook chạy nền trên Kaggle ("Save & Run All"/commit) đứng ở `0/N`
  hàng giờ không nhích**: nguyên nhân phổ biến nhất là phiên chạy đó **không
  thật sự có GPU**, dù bạn có chọn Accelerator = GPU trong Settings — Kaggle
  có thể âm thầm trả về CPU nếu quota GPU tuần đã hết, hoặc nếu bạn đổi
  Accelerator sau khi đã tạo session mà chưa restart/factory-reset. Cách
  kiểm tra: xem lại output cell `!nvidia-smi` ngay đầu log của chính phiên
  đó (không phải chạy lại) — nếu báo lỗi/không có bảng GPU, hoặc log không
  có dòng `GPU detected: ...` từ `train.py`, thì đúng là chạy CPU. Fine-tune
  một model ~1.5B trên CPU có thể mất hàng chục phút tới hàng giờ **cho một
  step**, nên `0/564` đứng yên nhiều giờ liền không phải là bug của code —
  cần dừng phiên đó, kiểm tra Settings -> Accelerator (chọn lại GPU rồi bấm
  factory reset/restart session) và trang quota GPU tuần của tài khoản, rồi
  chạy lại.
- **`torch.AcceleratorError: CUDA error: no kernel image is available for execution
  on the device`** (thường kèm warning trước đó dạng `Tesla P100-PCIE-16GB with
  CUDA capability sm_60 is not compatible with the current PyTorch installation`):
  đây KHÔNG phải bug của repo này. Kaggle đôi khi gán GPU **Tesla P100**
  (kiến trúc Pascal, `sm_60`), nhưng bản PyTorch có sẵn trong image Kaggle
  hiện tại (`2.10+cu128` trở lên) chỉ build kernel cho `sm_70` (Volta) trở
  lên — PyTorch đã lên lộ trình bỏ hỗ trợ Pascal/Volta cho các bản build
  CUDA 12.8/12.9 (xem pytorch/pytorch#157517), và đây cũng là lỗi đã được
  người dùng khác báo lên chính Kaggle/docker-python#1546, hiện chưa được
  Kaggle xử lý dứt điểm. Cách sửa:
  1. **Ưu tiên**: vào Settings -> Accelerator, đổi từ "GPU P100" sang **"GPU
     T4 x2"** (nếu tài khoản có) rồi factory-reset/restart session. T4 dùng
     kiến trúc Turing (`sm_75`), nằm trong danh sách torch hiện tại vẫn hỗ
     trợ, không cần sửa gì thêm trong `requirements.txt`.
  2. Nếu bắt buộc dùng P100, ép cài một bản torch cũ hơn còn hỗ trợ Pascal
     TRƯỚC khi `pip install -r requirements.txt`, ví dụ:
     `!pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu121`
     (rủi ro: có thể lệch phiên bản với các gói khác Kaggle đã cài sẵn).
- **Cấu trúc thư mục git bị lồng nhau nhiều lớp** (ví dụ
  `.../repo/repo/repo/...`) sau khi `git clone`: thường do lỡ chạy
  `git init`/`git add -A` ở thư mục cha thay vì đúng trong
  `uav-llm-pathfinding`. Không làm hỏng việc chạy script (đường dẫn tương
  đối vẫn đúng), nhưng nên dọn lại: xoá `.git` ở thư mục cha nếu có, chỉ giữ
  `.git` đúng bên trong `uav-llm-pathfinding`, `git add -A && git commit` lại
  từ đúng cấp thư mục rồi push.
