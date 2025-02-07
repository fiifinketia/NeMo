import os
import pandas as pd
# Log in to your W&B account
import wandb
from tqdm import tqdm
import json

# Use wandb-core
wandb.require("core")
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig

from TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs
import mutagen
import datetime
from pathlib import Path


# use temp_dir, and when done:
output_path = "/kaggle/working/tts_train_dir"
if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset_config = BaseDatasetConfig(
        formatter="ugspeech",
    dataset_name="ugspeech",
    path="/raid/datasets/LJSpeech-1.1_24khz/",
    meta_file_train="train.json",
    meta_file_val="test.json",
    language="gh_ak",
)

def formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    items = []
    with open(Path(meta_file).expanduser(), 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            file_info = {
                "audio_file": item["audio_filepath"],
                "text": item["text"],
                "root_path": "",
                "duration": item["duration"] if "duration" in item else None,
                "speaker_name": str(item["speaker"]) if "speaker" in item else None,
            }
    
            items.append(file_info)
    return items

audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)
# VitsConfig: all model related values for training, validating and testing.
config = VitsConfig(
    batch_size=80,
    eval_batch_size=32,
    batch_group_size=5,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    text_cleaner= "multilingual_cleaners",
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00af\u00b7\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u00ff\u0101\u0105\u0107\u0113\u0119\u011b\u012b\u0131\u0142\u0144\u014d\u0151\u0153\u015b\u016b\u0171\u017a\u017c\u01ce\u01d0\u01d2\u01d4\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451\u0454\u0456\u0457\u0491\u2013\u2107\u0254\u0190\u025B\u0186\u2019\u2183!'(),-.:;? ",
        punctuations="!'(),-.:;?’ ",
        phonemes="",
        is_unique=True,
        is_sorted=True,
    ),
    # text_cleaner="phoneme_cleaners",
    # use_phonemes=True,
    # phoneme_language="de",
    # phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    test_sentences=[
        "Nnipa nnum koto hɔ a nhwiren sisi wɔn anim. Wɔahyehyɛ wɔn nsa a wɔreyɛ nhwiren no ho adwuma. Nnipa no mu mmɔfra ɛyɛ mmɔfra emu mmienu nso yɛ mpanimfoɔ. Na sereɛ ɛdeda wɔn anim.",
        "Mmabunu bi a wɔhyehyɛ ntaadeɛ tuntum tuntum a wɔhyehyɛ ɛkyɛ kɔkɔɔ a wɔkuta adeɛ a wɔatwerɛtwerɛ so a wɔreyɛ yɛkyerɛ.",
        "Nnipa binom tete yoma so a wɔato santene wɔ akwaweɛ bi so.",
        "Nnipadɔm bebree nenam dwam hɔ.",
        "Nkwadaa pii redi agorɔ.",
    ],
    output_path=output_path,
    datasets=[dataset_config],
)

ap = AudioProcessor.init_from_config(config)
# Modify sample rate if for a custom audio dataset:
# ap.sample_rate = 22050

tokenizer, config = TTSTokenizer.init_from_config(config)
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    formatter=formatter
)

model = Vits(config, ap, tokenizer, speaker_manager=None)

wandb.init(
    project="train-vits",
    
)
wandb_callbacks = [
    WandbMetricsLogger(),
    WandbModelCheckpoint(filepath="my_model_{epoch:02d}"),
]

trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples, callbacks=wandb_callbacks
)


trainer.fit()

# temp_dir.cleanup()
wandb.finish()
