import torch
from phonerec.utils import load_model, load_yaml, visualize_with_feature
import soundfile as sf
import librosa
import argparse
import numpy as np

BEST_MODEL_PATH = "model-best.pt"
AUDIO_PATH = "./audio/Sumijo-oiseaux.wav"
SAVE_PATH = "./pt/model-best-inferred.pt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, nargs="?", default=BEST_MODEL_PATH)
    parser.add_argument("audio_path", type=str, nargs="?", default=AUDIO_PATH)
    parser.add_argument("save_path", type=str, nargs="?", default=SAVE_PATH)
    args = parser.parse_args()

    model, config, consts = load_model(args.model_path)
    paths = load_yaml("configs/paths.yaml")
    model.eval()

    x, sr = sf.read(args.audio_path)
    if x.ndim > 1:
        x = x[:, 0]
    x = librosa.resample(x, orig_sr=sr, target_sr=config.sample_rate)
    x_tensor = torch.from_numpy(x).float().to(config.device)

    with torch.no_grad():
        batch = dict()
        batch["audio"] = x_tensor
        predictions = model.run_on_batch(batch, cal_loss=False)
        batch["label"] = torch.argmax(predictions["frame"].squeeze(), dim=1)
        fig = visualize_with_feature(batch, model, config, {}, paths)
        # fig.savefig("output.png")

    torch.save(predictions["frame"].detach().to("cpu"), args.save_path)
