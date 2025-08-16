import os
import time
import inspect
import random
from datetime import datetime
from termcolor import colored, cprint
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from options.train_options import TrainOptions
from datasets.dataloader import config_dataloader, get_data_generator
from models.base_model import create_model
from utils.distributed import get_rank, synchronize, get_world_size
from utils.util import seed_everything, category_5_to_num, category_5_to_label
from utils.visualizer import Visualizer


def train_worker(opt, model, train_loader, test_loader, visualizer):
    """Main training worker function."""
    if get_rank() == 0:
        cprint(f'[*] Starting training for experiment: {opt.name}', 'blue')

    train_data_generator = get_data_generator(train_loader)
    test_data_generator = get_data_generator(test_loader)
    
    epoch_length = len(train_loader)
    cprint(f'Epoch length: {epoch_length}', 'cyan')

    total_iters = epoch_length * opt.epochs
    start_iter = opt.start_iter
    epoch = start_iter // epoch_length

    pbar = tqdm(range(start_iter, total_iters), disable=(get_rank() != 0))
    iter_start_time = time.time()

    for current_iter in pbar:
        opt.iter_i = current_iter
        next_iter = current_iter + 1

        if get_rank() == 0:
            visualizer.reset()

        data = next(train_data_generator)
        data['iter_num'] = current_iter
        data['epoch'] = epoch
        model.set_input(data)
        model.optimize_parameters()

        if get_rank() == 0:
            if current_iter % opt.print_freq == 0:
                errors = model.get_current_errors()
                elapsed_time = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(current_iter, errors, elapsed_time)

            if next_iter % opt.save_latest_freq == 0:
                cprint(f'Saving the latest model (iteration {current_iter})', 'blue')
                model.save('steps-latest', next_iter)

            if next_iter % opt.save_steps_freq == 0:
                cprint(f'Saving the model at iteration {next_iter}', 'blue')
                model.save(f'steps-{next_iter}', next_iter)
                cprint(f'[*] End of step {next_iter} \t Time Taken: {time.time() - iter_start_time:.2f} sec',
                       'blue', attrs=['bold'])

        if current_iter % epoch_length == epoch_length - 1:
            epoch += 1
            cprint(f'Finished epoch {epoch - 1}. Starting epoch {epoch}.', 'green')

        if current_iter % opt.display_freq == 0 and (current_iter > 0 or opt.debug == "1"):
            if opt.model == "vae":
                eval_data = next(test_data_generator)
                eval_data['iter_num'] = current_iter
                eval_data['epoch'] = epoch
                model.set_input(eval_data)
                model.inference(save_folder=f'temp/{current_iter}')
            else:
                category = random.choice(list(category_5_to_num.keys())) if opt.category == "im_5" else opt.category
                model.sample(category=category, prefix='results', ema=True, ddim_steps=200,
                             save_index=current_iter, cond=opt.cond, cond_dir=opt.cond_dir, iter_i=current_iter)

        if opt.update_learning_rate:
            model.update_learning_rate_cos(epoch, opt)


def generate_vae_samples(opt, model, test_loader):
    """Generates samples from a VAE model."""
    if get_rank() == 0:
        cprint(f'[*] Starting VAE sample generation for: {opt.name}', 'blue')

    test_data_generator = get_data_generator(test_loader)
    num_samples = len(test_loader)
    pbar = tqdm(range(num_samples), disable=(get_rank() != 0))

    for i in pbar:
        data = next(test_data_generator)
        data['iter_num'] = i
        data['epoch'] = 0
        model.set_input(data)
        seed_everything(opt.seed)
        model.inference()
        pbar.set_description(f"Generated sample {i+1}/{num_samples}")


def generate_diffusion_samples(opt, model):
    """Generates samples from a diffusion model."""
    if get_rank() == 0:
        cprint(f'[*] Starting diffusion model sample generation for: {opt.name}', 'blue')

    total_num = category_5_to_num.get(opt.category)
    if total_num is None:
        raise ValueError(f"Category '{opt.category}' not found in category_5_to_num map.")

    pbar = tqdm(total=total_num, disable=(get_rank() != 0))

    for i in range(total_num):
        result_index = i * get_world_size() + get_rank()
        if result_index >= total_num:
            break

        split_small = None
        if opt.split_dir:
            split_path = os.path.join(opt.split_dir, f'{result_index}.pth')
            if os.path.exists(split_path):
                split_small = torch.load(split_path, map_location=model.device)
        
        model.batch_size = 1
        category = random.choice(list(category_5_to_label.keys())) if opt.category == "im_5" else opt.category
        
        model.sample(category=category, prefix='results', ema=False, save_index=result_index, iter_i=i, cond=opt.cond, cond_dir=opt.cond_dir)
        pbar.update(1)


def main():
    """Main entry point for training and generation."""
    mp.set_sharing_strategy('file_descriptor')
    
    opt = TrainOptions().parse_and_setup()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    opt.local_rank = local_rank
    
    if get_rank() == 0:
        opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')
        cprint(f"Experiment time: {opt.exp_time}", "yellow")

    model = create_model(opt)
    opt.start_iter = model.start_iter
    cprint(f'[*] Model "{opt.model}" initialized.', 'cyan')

    visualizer = Visualizer(opt)
    if get_rank() == 0:
        visualizer.setup_io()
        
        # Backup source files
        expr_dir = os.path.join(opt.logs_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)

        files_to_backup = [
            inspect.getfile(model.__class__),
            "datasets/dualoctree_snet.py",
            "train.py"
        ]

        if opt.model != "vae" and hasattr(model, 'df_module') and hasattr(model.df_module, '__class__'):
            files_to_backup.append(inspect.getfile(model.df_module.__class__))

        if opt.vq_cfg:
            files_to_backup.append(opt.vq_cfg)
        if opt.df_cfg:
            files_to_backup.append(opt.df_cfg)

        for file_path in files_to_backup:
            if os.path.exists(file_path):
                destination_path = os.path.join(expr_dir, os.path.basename(file_path))
                os.system(f'cp {file_path} {destination_path}')

    if opt.mode == 'train':
        train_loader, test_loader = config_dataloader(opt)
        train_worker(opt, model, train_loader, test_loader, visualizer)
    elif opt.mode == 'generate':
        if opt.model == "vae":
            _, test_loader = config_dataloader(opt)
            generate_vae_samples(opt, model, test_loader)
        else:
            generate_diffusion_samples(opt, model)
    else:
        raise ValueError(f"Unknown mode: {opt.mode}")


if __name__ == "__main__":
    main()


