import json
import os
from dataclasses import dataclass, is_dataclass
from typing import Optional

import torch
import pandas as pd
from omegaconf import MISSING, OmegaConf, open_dict
from nemo.utils import logging
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.transcribe_utils import (
    PunctuationCapitalization,
    TextProcessingConfig,
    compute_metrics_per_sample,
)
from nemo.collections.common.metrics.punct_er import DatasetPunctuationErrorRate
from nemo.core.config import hydra_runner


@dataclass
class EvaluationConfig:
    dataset_manifest: str = MISSING
    output_filename: Optional[str] = "evaluation_transcripts.json"
    decoder_type: Optional[str] = None
    att_context_size: Optional[list] = None
    use_cer: bool = False
    use_punct_er: bool = False
    tolerance: Optional[float] = None
    only_score_manifest: bool = False
    scores_per_sample: bool = False
    text_processing: Optional[TextProcessingConfig] = TextProcessingConfig(
        punctuation_marks=".,?", separate_punctuation=False, do_lowercase=False, rm_punctuation=False,
    )


def transcribe_audio(asr_model, audio_dir):
    # List all WAV files in the directory
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

    # Prepare a list to store transcriptions
    transcriptions = []

    # Transcribe each audio file
    for audio_file in audio_files:
        audio_id = os.path.basename(audio_file).split('.')[0]
        transcription = asr_model.transcribe([audio_file], batch_size=1)[0]
        transcriptions.append({"audio": audio_id, "transcript": transcription})

    return transcriptions


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig):
    torch.set_grad_enabled(False)

    if not os.path.exists(cfg.dataset_manifest):
        raise FileNotFoundError(f"The dataset manifest file could not be found at path: {cfg.dataset_manifest}")

    if cfg.audio_dir is not None:
        raise RuntimeError(
            "Evaluation script requires ground truth labels to be passed via a manifest file. "
            "If manifest file is available, submit it via `dataset_manifest` argument."
        )

    # Transcribe speech into an output directory
    transcription_cfg = transcribe_speech.main(cfg) if not cfg.only_score_manifest else cfg

    # Release GPU memory if it was used during transcription
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Finished transcribing speech dataset. Computing ASR metrics..")

    ground_truth_text = []
    predicted_text = []
    invalid_manifest = False

    with open(transcription_cfg.output_filename, 'r') as f:
        for line in f:
            data = json.loads(line)

            if "pred_text" not in data:
                invalid_manifest = True
                break

            ground_truth_text.append(data[cfg.gt_text_attr_name])
            predicted_text.append(data["pred_text"])

    pc = PunctuationCapitalization(cfg.text_processing.punctuation_marks)

    if cfg.text_processing.separate_punctuation:
        ground_truth_text = pc.separate_punctuation(ground_truth_text)
        predicted_text = pc.separate_punctuation(predicted_text)
    if cfg.text_processing.do_lowercase:
        ground_truth_text = pc.do_lowercase(ground_truth_text)
        predicted_text = pc.do_lowercase(predicted_text)
    if cfg.text_processing.rm_punctuation:
        ground_truth_text = pc.rm_punctuation(ground_truth_text)
        predicted_text = pc.rm_punctuation(predicted_text)

    if invalid_manifest:
        raise ValueError(
            f"Invalid manifest provided: {transcription_cfg.output_filename} does not contain value for `pred_text`."
        )

    if cfg.use_punct_er:
        dper_obj = DatasetPunctuationErrorRate(
            hypotheses=predicted_text,
            references=ground_truth_text,
            punctuation_marks=list(cfg.text_processing.punctuation_marks),
        )
        dper_obj.compute()

    if cfg.scores_per_sample:
        metrics_to_compute = ["wer", "cer"]

        if cfg.use_punct_er:
            metrics_to_compute.append("punct_er")

        samples_with_metrics = compute_metrics_per_sample(
            manifest_path=cfg.dataset_manifest,
            reference_field=cfg.gt_text_attr_name,
            hypothesis_field="pred_text",
            metrics=metrics_to_compute,
            punctuation_marks=cfg.text_processing.punctuation_marks,
            output_manifest_path=cfg.output_filename,
        )

    # Compute the WER
    cer = word_error_rate(hypotheses=predicted_text, references=ground_truth_text, use_cer=True)
    wer = word_error_rate(hypotheses=predicted_text, references=ground_truth_text, use_cer=False)

    if cfg.use_cer:
        metric_name = 'CER'
        metric_value = cer
    else:
        metric_name = 'WER'
        metric_value = wer

    if cfg.tolerance is not None and metric_value > cfg.tolerance:
        raise ValueError(f"Got {metric_name} of {metric_value}, which was higher than tolerance={cfg.tolerance}")

    logging.info(f"Dataset WER/CER {wer:.2%}/{cer:.2%}")

    if cfg.use_punct_er:
        dper_obj.print()
        dper_obj.reset()

    # Inject the metric name and score into the config, and return the entire config
    with open_dict(cfg):
        cfg.metric_name = metric_name
        cfg.metric_value = metric_value

    return cfg


if __name__ == '__main__':
    # Initialize the ASR model
    asr_model = EncDecCTCModelBPE.restore_from(restore_path="ASR_squad.nemo")

    # Directory containing WAV files
    audio_dir = "/kaggle/input/aic-competition/mct-aic-2/test"

    # Transcribe and evaluate
    transcriptions = transcribe_audio(asr_model, audio_dir)

    # Save transcriptions to a CSV file
    output_df = pd.DataFrame(transcriptions)
    output_df.to_csv("transcriptions.csv", index=False, encoding='utf-8')

    print("Transcriptions saved to transcriptions.csv")

    # Run evaluation
    main()
