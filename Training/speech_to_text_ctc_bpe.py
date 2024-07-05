import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="/kaggle/working", config_name="conformer_ctc_bpe")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)  
    # Save the model
    final_model_path = '/kaggle/working/ASR_squad.nemo'
    asr_model.save_to(final_model_path)
    logging.info(f'Model saved at {final_model_path}')

if __name__ == '__main__':
    main()
