import os
import torch

src_dir = "save_model"
mobilefacenet_dst_dir = "Android/app/src/main/assets"
mtcnn_dst_dir = "Android/app/src/main/assets/mtcnn"

os.makedirs(mtcnn_dst_dir, exist_ok=True)

for name in ["PNet", "RNet", "ONet"]:
    src = os.path.join(src_dir, "mtcnn", f"{name}.pth")
    dst = os.path.join(mtcnn_dst_dir, f"{name}.pt")

    # 关键点：强制映射到 CPU
    model = torch.jit.load(src, map_location="cpu")
    model = model.eval()
    model = torch.jit.freeze(model)

    # 保存为 Android 可加载的 CPU TorchScript
    torch.jit.save(model, dst)
    print(f"导出MTCNN模型完成: {dst}")


src = os.path.join(src_dir, "mobilefacenet.pth")
dst = os.path.join(mobilefacenet_dst_dir, "mobilefacenet.pt")

# 关键点：强制映射到 CPU
model = torch.jit.load(src, map_location="cpu")
model = model.eval()
model = torch.jit.freeze(model)

# 保存为 Android 可加载的 CPU TorchScript
torch.jit.save(model, dst)
print(f"导出MobileFaceNet模型完成: {dst}")