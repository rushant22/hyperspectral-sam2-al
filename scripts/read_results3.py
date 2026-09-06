import json, os

BASE = r"C:\Users\HP\Desktop\college\sem5\ML\Results\Result 3\results"

# AL Results
for strat in ['bald', 'entropy', 'random']:
    path = os.path.join(BASE, "al_results_" + strat + ".json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rounds = data.get("rounds", [])
    print("\n=== %s ===" % strat.upper())
    for r in rounds:
        miou_pct = r["miou"] * 100
        print("  R%d: labeled=%d  mIoU=%.4f (%.2f%%)  new=%d" % (
            r["round"], r["labeled_count"], r["miou"], miou_pct, r.get("num_new_labels", 0)))
    if rounds:
        best = max(rounds, key=lambda x: x["miou"])
        print("  BEST: R%d mIoU=%.4f (%.2f%%)" % (best["round"], best["miou"], best["miou"]*100))

# Adapter training log (last few epochs)
log_path = os.path.join(BASE, "adapter_training_log.json")
with open(log_path, "r", encoding="utf-8") as f:
    log = json.load(f)

epochs = log.get("epochs", [])
print("\n=== ADAPTER TRAINING LOG (last 5 epochs) ===")
for ep in epochs[-5:]:
    print("  Epoch %d: loss=%.4f  val_miou=%.4f (%.2f%%)" % (
        ep["epoch"], ep["train_loss"], ep["val_miou"], ep["val_miou"]*100))
if epochs:
    best_ep = max(epochs, key=lambda x: x["val_miou"])
    print("  BEST EPOCH: %d  val_miou=%.4f (%.2f%%)" % (best_ep["epoch"], best_ep["val_miou"], best_ep["val_miou"]*100))
    print("  FINAL EPOCH val_oa: %.4f (%.2f%%)" % (epochs[-1].get("val_oa", 0), epochs[-1].get("val_oa", 0)*100))

# Ablation
abl_path = os.path.join(BASE, "ablation_results.json")
with open(abl_path, "r", encoding="utf-8") as f:
    abl = json.load(f)
print("\n=== ABLATION RESULTS ===")
for k, v in abl.items():
    print("  %s: %s" % (k, v))
