import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from rde.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from rde.utils.data import PaddingCollate
from rde.utils.train import *
from rde.datasets.pdbredo_chain import get_pdbredo_chain_dataset
from rde.models.rde import CircularSplineRotamerDensityEstimator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs_rde')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name+'-resume', tag=args.tag)
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    train_dataset = get_pdbredo_chain_dataset(config.data.train)
    val_dataset = get_pdbredo_chain_dataset(config.data.val)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=args.num_workers)
    train_iterator = inf_iterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = CircularSplineRotamerDensityEstimator(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        lsd_result = model.load_state_dict(ckpt['model'], strict=False)
        logger.info('Missing keys (%d): %s' % (len(lsd_result.missing_keys), ', '.join(lsd_result.missing_keys)))
        logger.info('Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), ', '.join(lsd_result.unexpected_keys)))
        
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])

    def train(it):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), args.device)

        # Forward pass
        loss_dict = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        time_forward_end = current_milli_time()

        # Backward
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        scalar_dict = {}
        scalar_dict.update({
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })
        log_losses(loss, loss_dict, scalar_dict, it=it, tag='train', logger=logger, writer=writer)

    def validate(it):
        scalar_accum = ScalarMetricAccumulator()
        chi_pred, chi_native, chi_masked_flag, chi_corrupt_flag, aa_all = [], [], [], [], []
        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)

                # Forward pass
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')

                # Sampling
                xs, _ = model.sample(batch)
                chi_pred.append(xs.cpu())
                chi_native.append(batch['chi_native'].cpu())
                chi_masked_flag.append(
                    (batch['chi_masked_flag'][..., None] * batch['chi_mask']).cpu()
                )
                chi_corrupt_flag.append(
                    (batch['chi_corrupt_flag'][..., None] * batch['chi_mask']).cpu()
                )
                aa_all.append(batch['aa'].cpu())
            
        avg_loss = scalar_accum.get_average('loss')
        scalar_accum.log(it, 'val', logger=logger, writer=writer)

        chi_pred, chi_native = torch.cat(chi_pred, dim=0), torch.cat(chi_native, dim=0)
        chi_masked_flag = torch.cat(chi_masked_flag, dim=0)
        chi_corrupt_flag = torch.cat(chi_corrupt_flag, dim=0)
        aa_all = torch.cat(aa_all, dim=0)
        acc_table_masked = aggregate_sidechain_accuracy(aa_all, chi_pred, chi_native, chi_masked_flag)
        acc_table_corrupt = aggregate_sidechain_accuracy(aa_all, chi_pred, chi_native, chi_corrupt_flag)
        print(acc_table_masked)
        writer.add_figure(
            'val/acc_table_masked', 
            make_sidechain_accuracy_table_image('masked', acc_table_masked), 
            global_step=it
        )
        writer.add_figure(
            'val/acc_table_corrupt',
            make_sidechain_accuracy_table_image('corrupt', acc_table_corrupt),
            global_step=it
        )

        # Trigger scheduler
        if it != it_first:  # Don't step optimizers after resuming from checkpoint
            if config.train.scheduler.type == 'plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        return avg_loss

    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
