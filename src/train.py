import argparse, json, os, time
from pathlib import Path
from model import train_from_csv, save_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--version", default=None)  # Jenkins build number hoặc git commit
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    version = args.version or str(int(time.time()))
    model_path = outdir / f"model_{version}.joblib"
    latest_path = outdir / "model_latest.joblib"
    meta_path = outdir / f"meta_{version}.json"

    model = train_from_csv(args.data)
    save_model(model, str(model_path))
    save_model(model, str(latest_path))

    meta = {
        "version": version,
        "data": args.data,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "artifact": str(model_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[TRAIN] Saved: {model_path}")
    print(f"[TRAIN] Latest: {latest_path}")
    print(f"[TRAIN] Meta:   {meta_path}")

if __name__ == "__main__":
    main()
