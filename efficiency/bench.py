"""Per-network GPU cost: ms per crop (batch 64) / per volume, fp16 weights, on the current GPU."""
import json, os, sys, time, torch
sys.argv += [] 
from pipeline import Models, load_recipe, CROP_BATCH, UNET_BATCH, SIDE_BATCH, VOL_DEPTH, VOL_SIZE
W = sys.argv[1] if len(sys.argv) > 1 else "weights_val"
names = sorted(os.listdir(os.path.join(W, "crops")))
recipe = {n: {"models": [n], "minus": None} for n in ["ACL"]}
recipe = {"ACL": {"models": names, "minus": None}}
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = False
M = Models(W, recipe, dev)
M.warmup()
def bench(fn, n=20):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n
with torch.no_grad():
    from side_cls import model_input
    x = torch.zeros((UNET_BATCH, 1, VOL_DEPTH, VOL_SIZE, VOL_SIZE), device=dev, dtype=torch.float16)
    print(f"unet   {bench(lambda: M.unet(x)) / UNET_BATCH * 1e3:7.2f} ms / volume")
    pl = torch.zeros(SIDE_BATCH, dtype=torch.long, device=dev)
    xs = model_input(torch.zeros((SIDE_BATCH, 1, VOL_DEPTH, VOL_SIZE, VOL_SIZE), device=dev), pl).half()
    print(f"side   {bench(lambda: M.side(xs)) / SIDE_BATCH * 1e3:7.2f} ms / volume")
    for m in M.crops:
        k = json.loads(m["z"]); s = k.get("size", 128)
        xc = torch.zeros((CROP_BATCH, k["depth"], s, s), device=dev, dtype=torch.float16)
        i = torch.zeros(CROP_BATCH, dtype=torch.long, device=dev)
        a = json.load(open(os.path.join(W, "crops", m["name"], "args.json")))
        print(f"{m['name']:8s} {bench(lambda: m['net'](xc, i, i)) / CROP_BATCH * 1e3:7.3f} ms / crop  "
              f"{a['backbone'][:24]:24s} depth {k['depth']} size {s} img {a.get('img_size')} planes {a['planes']} group {a['group']}")
