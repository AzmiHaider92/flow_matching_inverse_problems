import argparse
import os
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from model_mnist import ImageFlow
import wandb


def main(args):
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    run_name = args.run_name + "_" + timestamp

    if args.use_wandb:
        wandb.init(
            entity="viewformer",
            # Set the wandb project where this run will be logged.
            project="mnist-flow-matching",
            # Track hyperparameters and run metadata.
            config=vars(args),
            name=run_name,
        )
        wandb.save("model_mnist.py", policy="now")  # Save a copy of the model file for reference


    model = ImageFlow(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        n_measurements=args.n_measurements,
        noise_std=args.noise_std,
        flow_lr=args.flow_lr,
        latent_lr=args.latent_lr,
        flow_weight=args.flow_weight,
        render_weight=args.render_weight,
        guided_N=args.guided_N,
        guided_eta=args.guided_eta,
        num_images=args.num_images,
        flow_model_path=args.flow_model_path,
        batch_size=args.batch_size,
        fm_steps=args.fm_steps,
        flow_train_epochs=args.flow_train_epochs,
        warmup_epochs=args.warmup_epochs,
        flow_refine_every=args.flow_refine_every,
        use_persistent_latents=args.PL,
        use_gaussian_smoothing=args.GS,
        model_type=args.model_type,
    )

    save_dir = os.path.join("mnist_flow_checkpoints", run_name)
    os.makedirs(save_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,
            filename="mnist-{epoch:02d}-{train_combined_loss:.4f}",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    loggers = None
    if args.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        loggers = WandbLogger(project="mnist-flow-matching", name=args.run_name, log_model=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,
        accelerator='auto',
        devices=1,
        log_every_n_steps=1,
    )

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"Resuming from {args.resume_from_checkpoint}")
    else:
        print("Starting new training run")

    trainer.fit(model, ckpt_path=args.resume_from_checkpoint if args.resume_from_checkpoint else None)

    print("Saving trained latent images...")
    save_path = os.path.join(save_dir, f"{args.run_name}_latents.pt")
    torch.save({
        'latent_images': model.latent_images.detach().cpu(),
        'gt_images': model.gt_images.cpu(),
        'gt_observations': model.gt_observations.cpu(),
    }, save_path)
    print(f"Saved latents to {save_path}")

    # Final MSE between latents and GT images
    with torch.no_grad():
        latents_01 = model.latent_images.detach().cpu().mul(0.5).add(0.5).clamp(0, 1)
        gt_01 = model.gt_images.cpu().mul(0.5).add(0.5).clamp(0, 1)
        mse = torch.mean((latents_01 - gt_01) ** 2).item()
        psnr = model._psnr(latents_01, gt_01).item()
    print(f"FINAL_LATENT_MSE={mse:.6f}")
    print(f"FINAL_LATENT_PSNR={psnr:.4f}")
    print("Training done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--num_images', type=int, default=10_000)
    parser.add_argument('--n_measurements', type=int, default=100)
    parser.add_argument('--noise_std', type=float, default=1e-3)
    parser.add_argument('--flow_lr', type=float, default=1e-4)
    parser.add_argument('--latent_lr', type=float, default=0.01)
    parser.add_argument('--flow_weight', type=float, default=1.0)
    parser.add_argument('--render_weight', type=float, default=1.0)
    parser.add_argument('--guided_N', type=int, default=20)
    parser.add_argument('--guided_eta', type=float, default=1.0)
    parser.add_argument('--flow_model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fm_steps', type=int, default=20)
    parser.add_argument('--flow_train_epochs', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--flow_refine_every', type=int, default=1,
                        help='Train/refine the flow model every N epochs after warmup.')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--run_name', type=str, default='mnist_flow')
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    # Changes
    parser.add_argument('--PL', type=bool, default=False,
                        help="Persistent latents. If false, latents are replaced after guidance, not using optimizer")
    parser.add_argument('--GS', type=bool, default=False,
                        help="Using Gaussian smoothing: in the optimizer beta-1=0.999 means gaussian smoothing")
    parser.add_argument('--model_type', type=str, default='flow', choices=['flow', 'diffusion'],
                        help="Switch between Flow Matching and Diffusion logic.")

    args = parser.parse_args()
    main(args)
