import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
from model_mnist import ImageFlow


def main(args):
    if args.use_wandb:
        wandb.init(
            project="mnist-flow-matching",
            name=args.run_name,
            config=vars(args),
            save_code=True
        )

    model = ImageFlow(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        n_measurements=args.n_measurements,
        noise_std=args.noise_std,
        flow_lr=args.flow_lr,
        latent_lr=args.latent_lr,
        flow_weight=args.flow_weight,
        render_weight=args.render_weight,
        render_loss_type=args.render_loss_type,
        guided_N=args.guided_N,
        guided_warmup_steps=args.guided_warmup_steps,
        guided_eta=args.guided_eta,
        num_images=args.num_images,
        flow_model_path=args.flow_model_path,
        batch_size=args.batch_size,
        fm_steps=args.fm_steps,
        flow_train_epochs=args.flow_train_epochs,
        warmup_epochs=args.warmup_epochs,
        latent_l1_weight=args.latent_l1_weight,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath="mnist_flow_checkpoints",
            filename="mnist-flow-{epoch:02d}-{train_combined_loss:.4f}",
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
    save_dir = "mnist_trained_latents"
    os.makedirs(save_dir, exist_ok=True)
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
    parser.add_argument('--flow_lr', type=float, default=2e-4)
    parser.add_argument('--latent_lr', type=float, default=0.01)
    parser.add_argument('--flow_weight', type=float, default=1.0)
    parser.add_argument('--render_weight', type=float, default=1.0)
    parser.add_argument('--render_loss_type', type=str, default='mse', choices=['mse', 'mae', 'ncc'])
    parser.add_argument('--guided_N', type=int, default=10)
    parser.add_argument('--guided_warmup_steps', type=int, default=3)
    parser.add_argument('--guided_eta', type=float, default=0.1)
    parser.add_argument('--flow_model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fm_steps', type=int, default=100)
    parser.add_argument('--flow_train_epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--run_name', type=str, default='mnist_flow')
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--latent_l1_weight', type=float, default=0.01)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    args = parser.parse_args()
    main(args)
