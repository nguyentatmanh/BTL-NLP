import json
import os
import matplotlib.pyplot as plt

def plot_training_loss(checkpoint_dir="src/viet_qa/checkpoints/extractive", output_file="loss_chart.png"):
    state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    
    # Nếu không có ở thư mục gốc, tìm trong các thư mục checkpoint-* mới nhất
    if not os.path.exists(state_file):
        checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            state_file = os.path.join(latest_checkpoint, "trainer_state.json")

    if not os.path.exists(state_file):
        print(f"❌ Không tìm thấy file trainer_state.json ở {checkpoint_dir} hoặc các thư mục con.")
        return

    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    log_history = data.get("log_history", [])
    
    train_epochs = []
    train_loss = []
    
    val_epochs = []
    val_loss = []

    for log in log_history:
        # Lấy loss của quá trình Train
        if "loss" in log and "epoch" in log:
            train_epochs.append(log["epoch"])
            train_loss.append(log["loss"])
            
        # Lấy loss của quá trình Validation (Đánh giá)
        if "eval_loss" in log and "epoch" in log:
            val_epochs.append(log["epoch"])
            val_loss.append(log["eval_loss"])

    if not train_loss and not val_loss:
        print("⚠️ File log không chứa thông tin loss. Bạn kiểm tra lại quá trình train nhé.")
        return

    # TẠO BẢNG THỐNG KÊ (MARKDOWN STYLE)
    print("\n" + "="*50)
    print(" BẢNG TỔNG HỢP THÔNG SỐ ĐÀO TẠO QUA TỪNG EPOCH")
    print("="*50)
    print(f"| {'Epoch':^7} | {'Train Loss':^15} | {'Validation Loss':^15} |")
    print("|---------|-----------------|-----------------|")
    
    table_md = "| Epoch | Train Loss | Validation Loss |\n|-------|------------|-----------------|\n"
    
    # Gom nhóm theo epoch (lấy số tròn của Epoch VD: 1.0, 2.0)
    epochs_set = sorted(list(set([round(e) for e in train_epochs + val_epochs])))
    
    for ep in epochs_set:
        # Tìm loss gần với epoch này nhất
        t_loss = next((loss for e, loss in zip(train_epochs, train_loss) if round(e) == ep), "N/A")
        v_loss = next((loss for e, loss in zip(val_epochs, val_loss) if round(e) == ep), "N/A")
        
        t_str = f"{t_loss:.4f}" if isinstance(t_loss, float) else str(t_loss)
        v_str = f"{v_loss:.4f}" if isinstance(v_loss, float) else str(v_loss)
        
        print(f"| {ep:^7} | {t_str:^15} | {v_str:^15} |")
        table_md += f"| {ep} | {t_str} | {v_str} |\n"
        
    print("="*50 + "\n")
    
    # Lưu bảng ra file markdown để bạn copy vào báo cáo
    with open("loss_report.md", "w", encoding="utf-8") as fm:
        fm.write("## Bảng Thông Số Train/Validation Loss\n\n")
        fm.write(table_md)
    print("✅ Đã lưu bảng thống kê dạng text vào file: loss_report.md\n")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    
    if train_loss:
        plt.plot(train_epochs, train_loss, label='Train Loss', color='blue', marker='o', linestyle='-', linewidth=2)
    if val_loss:
        plt.plot(val_epochs, val_loss, label='Validation Loss', color='red', marker='s', linestyle='--', linewidth=2)

    plt.title('Đồ thị quá trình học của mô hình (Train/Val Loss)', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Độ lỗi / Sai số)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Lưu và hiển thị
    plt.savefig(output_file, dpi=300)
    print(f"✅ Đã lưu đồ thị thành công tại: {os.path.abspath(output_file)}")
    
    try:
        plt.show()  # Mở popup xem ảnh nếu máy hỗ trợ UI
    except Exception:
        pass

if __name__ == "__main__":
    plot_training_loss()
