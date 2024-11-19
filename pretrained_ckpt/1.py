import torch
import os


def modema(ckpt=None):
    opath = os.path.join(os.path.dirname(ckpt), f"new_{os.path.basename(ckpt)}")
    _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
    _ckpt["model"] = _ckpt["model_ema"]
    torch.save(_ckpt, open(opath, "wb"))

if __name__ == "__main__":
    modema("./vssmsmall_dp03_ckpt_epoch_238.pth")