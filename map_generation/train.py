import pytorch_lightning as pl
from config import CHECKPOINTS_DIR, LOGS_DIR
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

    
def train_model(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    epochs: int = 50,
    name: str = "model",
    gpu: bool = True,
    verbose: bool = True,
) -> dict[str, float]:

    model_chkpt = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR,
        filename=name,
        monitor="train/loss",
        mode="min",
        verbose=verbose,
    )

    early_stopping = EarlyStopping(
        monitor="test/loss",
        mode="min",
        patience=100,
        strict=True,
        check_finite=True,
    )
    # bar = LitProgressBar()
    trainer = pl.Trainer(
        logger=TensorBoardLogger(
            save_dir=LOGS_DIR,
            name=name,
            default_hp_metric=False,
        ),
        callbacks=[model_chkpt, early_stopping],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        max_epochs=epochs,
        accelerator="gpu" if gpu else "cpu",
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
    )

    trainer.fit(model=model, datamodule=datamodule)