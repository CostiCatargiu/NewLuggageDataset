# train_yolov12s_custom.py
from ultralytics import YOLO
import torch
import time


# ---------------------------
# Epoch sync callback
# ---------------------------
def on_train_epoch_start(trainer):
    """Store current epoch in the global EpochTracker."""
    epoch = trainer.epoch

    try:
        from ultralytics.utils.loss import EPOCH_TRACKER
        EPOCH_TRACKER.set_epoch(epoch, trainer.epochs)
        if epoch % 10 == 0:
            print(f"[Epoch Sync] Global EpochTracker set to epoch={epoch}")
    except ImportError:
        if hasattr(trainer, 'model') and trainer.model is not None:
            trainer.model.current_epoch = epoch
            if epoch % 10 == 0:
                print(f"[Epoch Sync] Set model.current_epoch = {epoch}")


# ---------------------------
# Single experiment runner
# ---------------------------
def run_experiment(
    data_yaml,
    model_weights,
    project_dir,
    run_name,
    epochs,
    imgsz,
    batch,
    device,
    workers
    # alpha_start,
    # alpha_end,
    # center_loss_weight_init,
    # center_loss_weight_min,
):
    print(f"\n🚀 Starting experiment: {run_name}\n")

    # IMPORTANT: fresh model every time
    model = YOLO(model_weights)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project_dir,
        name=run_name,
        exist_ok=False,  # prevents overwrite

        # Ablation params
        # alpha_start=alpha_start,
        # alpha_end=alpha_end,
        # alpha_min=0.3,
        # alpha_max=1.0,
        #
        # small_obj_px=48,
        # small_obj_boost=1.0,
        #
        # center_loss_weight_init=center_loss_weight_init,
        # center_loss_weight_min=center_loss_weight_min,
        # center_loss_decay_epochs=50,
        #
        # iou_clip_start=6.0,
        # iou_clip_end=2.0,
        #
        # dfl_clip_start=1000.0,
        # dfl_clip_end=1000.0,
        #
        # tal_topk=8,
        # tal_alpha=0.5,
        # tal_beta=6.0,
    )


# ---------------------------
# MAIN
# ---------------------------
def main():
    DATA_YAML = "/home/constantin/Doctorat/LuggageDataset_v2i_YOLOV12_30percentagesubset/data.yaml"
    MODEL_WEIGHTS = "/home/constantin/Downloads/yolov12s.pt"
    PROJECT_DIR = "runs_top3_final_luggage"

    EPOCHS = 70
    IMG_SIZE = 640
    BATCH = 64
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    WORKERS = 8

    # Optional: timestamp to guarantee uniqueness
    ts = int(time.time())

    # ---------------------------
    # Experiment 1 (your current best)
    # ---------------------------
    run_experiment(
        data_yaml=DATA_YAML,
        model_weights=MODEL_WEIGHTS,
        project_dir=PROJECT_DIR,
        run_name=f"exp_dynamic_alpha_{ts}",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        # alpha_start=0.7,
        # alpha_end=0.4,
        # center_loss_weight_init=0.05,
        # center_loss_weight_min=0.01
    )

    # # ---------------------------
    # # Experiment 2 (baseline)
    # # ---------------------------
    # run_experiment(
    #     data_yaml=DATA_YAML,
    #     model_weights=MODEL_WEIGHTS,
    #     project_dir=PROJECT_DIR,
    #     run_name=f"exp_static_alpha_{ts}",
    #     epochs=EPOCHS,
    #     imgsz=IMG_SIZE,
    #     batch=BATCH,
    #     device=DEVICE,
    #     workers=WORKERS,
    #     alpha_start=1.0,
    #     alpha_end=1.0,
    #     center_loss_weight_init=0.00,
    #     center_loss_weight_min=0.00    )


if __name__ == "__main__":
    main()