import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # mute tokenizer's warning: https://github.com/huggingface/transformers/issues/5486

import random
from datetime import datetime

import numpy as np
import torch.nn as nn
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    
import sys
sys.path.append('.')
from pmc_clip import create_model_and_transforms, trace_model, get_pretrained_url, download_pretrained
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, cosine_annealing_lr
from training.train import train_one_epoch, train_one_epoch_mlm, train_one_epoch_cls
from training.evaluate import evaluate
from training.fusion_method import convert_model_to_cls

# decorator to report error message in torch distributed mode
if torch.__version__ >= '1.10':
    from torch.distributed.elastic.multiprocessing.errors import record
else:
    def record(func):
        def foo():
            func()
        return foo

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

@record
def main():
    args = parse_args()
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.log_dir, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = init_distributed_device(args)

    # assign log path
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.log_dir, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.log_dir, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    assert args.precision in ['amp', 'fp16', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args=args,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
    )
    
    #####add cls head####
    model = convert_model_to_cls(model, 11, args.fusion_method)
    #####add cls head####
    
    args.context_length = model.context_length
    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.log_dir, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            find_unused_parameters=True,
            **ddp_args
        )

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data:
        assert not args.trace, 'Cannot train with traced model'

        # exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        # include = lambda n, p: not exclude(n, p)

        # named_parameters = list(model.named_parameters())
        # gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        # rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        # optimizer = optim.AdamW(
        #     [
        #         {"params": gain_or_bias_params, "weight_decay": 0.},
        #         {"params": rest_params, "weight_decay": args.wd},
        #     ],
        #     lr=args.lr,
        #     betas=(args.beta1, args.beta2),
        #     eps=args.eps,
        # )
        
        # only train fusion and cls head
        model_to_train = model.module if hasattr(model, 'module') else model

        # 创建优化器
        optimizer = optim.AdamW(
            [
                # 所有可训练参数中的权重参数（维度>=2）：应用权重衰减
                {"params": [p for n, p in model_to_train.named_parameters() 
                            if p.requires_grad and p.ndim >= 2], 
                "weight_decay": args.wd},
                
                # 所有可训练参数中的偏置参数（维度<2）：不应用权重衰减
                {"params": [p for n, p in model_to_train.named_parameters() 
                            if p.requires_grad and p.ndim < 2], 
                "weight_decay": 0.},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            if ('epoch' in checkpoint) and (args.resume_model_only == False):
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd, strict=False)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                if len(checkpoint) == 1:
                    model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint['state_dict'])
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at local machine'{}'".format(args.resume))
            logging.info("=> trying to fetch checkpoint from remote server'{}'".format(args.resume))
            pretrained_model_name, tag = args.resume.split(':')
            pretrained_model_url = get_pretrained_url(pretrained_model_name, tag)
            cache_dir = os.path.expanduser("./.cache/clip")
            logging.info("=> fetch {} to {}".format(pretrained_model_url, cache_dir))
            download_target = download_pretrained(pretrained_model_url, cache_dir)

            checkpoint = torch.load(download_target, map_location=device)
            if len(checkpoint) == 1:
                model.load_state_dict(checkpoint)
            else:
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                # model.load_state_dict(checkpoint['state_dict'])
            logging.info(f"=> loaded checkpoint '{download_target}' (epoch {start_epoch})")

    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch)
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler_func = {
            'cosine': cosine_lr,
            'cosine_annealing': cosine_annealing_lr
        }[args.scheduler]
        scheduler = scheduler_func(optimizer, args.lr, args.warmup, total_steps, restarts=3)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.log_dir and args.log_dir.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="pmc-clip",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    trainer, evaluator = {
        'PMC_CLIP': (train_one_epoch_cls, evaluate),
    }.get(
        args.clip_model, (train_one_epoch, evaluate)
    )

    if 'train' not in data:
        evaluator(model, data, start_epoch, args, writer)
        return

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        trainer(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluator(model, data, completed_epoch, args, writer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )

    if args.wandb and is_master(args):
        wandb.finish()


if __name__ == "__main__":
    main()
