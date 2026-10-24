import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import time
import torch
import numpy as np
from scipy.optimize import minimize
import random

torch.set_printoptions(sci_mode=False)

parser = argparse.ArgumentParser(description=globals().get("__doc__", "Diffusion training"))

# added args
parser.add_argument('--low_mem_mode', type=bool, help='Turing on low VRAM mode', required=False, default=False)
parser.add_argument('--calib', action="store_true", help="Whether to test the calibration model")
parser.add_argument('--mlp_idx', type=int, required=False, help='Train the diffusion model with mlp index')
parser.add_argument('--seed', type=int, help='Seed value', required=False)
parser.add_argument('--preprocess', type=str, help='The preprocess method', choices=['grayscaled', 'standardized'],
                    required=True)
parser.add_argument('--noise_perturbation', type=float, help='The std of the added noise', required=False, default=0.0)
parser.add_argument('--low_resolution', type=int, help='The factor of down sampling', required=False, default=0)
parser.add_argument('--brightness', type=float, help='The level of brightness changes', required=False, default=0.0)
parser.add_argument('--contrast', type=float, help='The level of contrast changes', required=False, default=1.0)
parser.add_argument('--crop', type=float, help='The size of the random crop', required=False, default=0.0)
parser.add_argument('--covered', type=float, nargs=2, help='The size and number of the random cover regions',
                    required=False, default=(0.0, 0.0))
parser.add_argument('--attack_name', type=str, help='The attack name and eps', choices=['None', 'FGSM', 'PGD', 'AUTOPGD'],
                    required=False, default=None)
parser.add_argument('--eps', type=float, help='The attack eps', required=False)

parser.add_argument(
    "--config", type=str, required=True, help="Path to the config file"
)
parser.add_argument('--device', type=int, default=0, help='GPU device id')
parser.add_argument('--thread', type=int, default=4, help='number of threads')
parser.add_argument("--test_sample_seed", type=int, default=-1, help="Random seed during test time sampling")
parser.add_argument(
    "--exp",
    type=str,
    default="exp",
    help="Path for saving running related data."
)
parser.add_argument(
    "--doc",
    type=str,
    required=True,
    help="A string for documentation purpose. "
         "Will be the name of the log folder.",
)
parser.add_argument(
    "--dataroot", type=str, default=None,
    help="This argument will overwrite the dataroot in the config if it is not None."
)
parser.add_argument(
    "--comment", type=str, default="", help="A string for experiment comment"
)
parser.add_argument(
    "--verbose",
    type=str,
    default="info",
    help="Verbose level: info | debug | warning | critical",
)
parser.add_argument("--test", action="store_true", help="Whether to test the model")
parser.add_argument(
    "--confirm-final-test",
    action="store_true",
    help=(
        "Required with --test when aux_cls.fusion_guide.type=texture_only. "
        "Development training never reads the testing split."
    ),
)
parser.add_argument("--tune_T",
                    action="store_true",
                    help="Whether to tune the scaling temperature parameter for calibration with training set.")
parser.add_argument("--sanity_check",
                    action="store_true",
                    help="Whether to quickly check test function implementation by running on only a few subsets.")
parser.add_argument(
    "--sample",
    action="store_true",
    help="Whether to produce samples from the model",
)
parser.add_argument(
    "--train_guidance_only",
    action="store_true",
    help="Whether to only pre-train the guidance classifier f_phi",
)
parser.add_argument(
    "--noise_prior",
    action="store_true",
    help="Whether to apply a noise prior distribution at timestep T",
)
parser.add_argument(
    "--no_cat_f_phi",
    action="store_true",
    help="Whether to not concatenate f_phi as part of eps_theta input",
)
parser.add_argument(
    "--add_ce_loss",
    action="store_true",
    help="Whether to add cross entropy loss",
)
parser.add_argument(
    "--eval_best",
    action="store_true",
    help="Evaluate best model during training, instead of the ckpt stored at the last epoch",
)
parser.add_argument("--fid", action="store_true")
parser.add_argument("--interpolation", action="store_true")
parser.add_argument(
    "--resume_training", action="store_true", help="Whether to resume training"
)
parser.add_argument(
    "-i",
    "--image_folder",
    type=str,
    default="images",
    help="The folder name of samples",
)
parser.add_argument(
    "--n_splits", type=int, default=10, help="total number of runs with different seeds for a specific task"
)
parser.add_argument(
    "--split", type=int, default=0, help="split ID"
)
parser.add_argument(
    "--ni",
    action="store_true",
    help="No interaction. Suitable for Slurm Job launcher",
)
parser.add_argument(
    "--sample_type",
    type=str,
    default="generalized",
    help="sampling approach (generalized or ddpm_noisy)",
)
parser.add_argument(
    "--skip_type",
    type=str,
    default="uniform",
    help="skip according to (uniform or quadratic)",
)
parser.add_argument(
    "--timesteps", type=int, default=None, help="number of steps involved"
)
parser.add_argument(
    "--eta",
    type=float,
    default=0.0,
    help="eta used to control the variances of sigma",
)
parser.add_argument("--sequence", action="store_true")

parser.add_argument(
    "--loss", type=str, default='ddpm', help="loss function"
)

parser.add_argument(
    "--num_sample", type=int, default=1, help="number of samples used in forward and reverse"
)

args = parser.parse_args()

if args.seed is None:
    args.seed = random.randint(0, 10000)


def parse_config():
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    with open(os.path.join(args.config), "r") as f:
        if args.sample:
            config = yaml.unsafe_load(f)
            new_config = config
        else:
            config = yaml.safe_load(f)
            new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    if not args.ni:
        import torch.utils.tensorboard as tb

    if args.dataroot is not None:
        new_config.data.dataroot = args.dataroot

    if not args.sample:
        args.im_path = os.path.join(args.exp, new_config.training.image_folder, args.doc)
        new_config.diffusion.noise_prior = True if args.noise_prior else False
        new_config.model.cat_y_pred = False if args.no_cat_f_phi else True
        if not args.resume_training:
            if args.timesteps is not None:
                new_config.diffusion.timesteps = args.timesteps
            if args.num_sample > 1:
                new_config.diffusion.num_sample = args.num_sample
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    shutil.rmtree(args.im_path)
                    os.makedirs(args.log_path)
                    os.makedirs(args.im_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)
                if not os.path.exists(args.im_path):
                    os.makedirs(args.im_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        if not args.ni:
            new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        else:
            new_config.tb_logger = None
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        raise NotImplementedError

    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    logging.info("Using device: {}".format(device))
    new_config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return new_config, logger


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    config, logger = parse_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    from fusion_guide_runtime import fusion_guide_type_from_config, resolve_fusion_dataset_splits

    # Fail early for texture_only --test without confirm; also reject confirm without --test.
    resolve_fusion_dataset_splits(
        fusion_guide_type_from_config(config),
        is_test=bool(args.test),
        confirm_final_test=bool(args.confirm_final_test),
    )

    if args.loss == 'card_onehot_conditional':
        # sevit / swin / transnext 均在本文件中实现；本仓库无 latent_classification 模块
        from classification_train_separately import Diffusion
    else:
        raise NotImplementedError("Invalid loss option")

    try:
        runner = Diffusion(args, config, device=config.device)
        start_time = time.time()
        procedure = None
        if args.sample:
            runner.sample()
            procedure = "Sampling"
        elif args.test:
            if config.data.dataset in ['ChestXRay', 'ChestXRayAtkFGSM', 'ChestXRayAtkPGD', 'ChestXRayAtkBIM',
                                       'ChestXRayAtkAUTOPGD', 'ChestXRayAtkCW', 'ISICSkinCancer',
                                       'ISICSkinCancerAtkFGSM', 'ISICSkinCancerAtkPGD', 'ISICSkinCancerAtkBIM',
                                       'ISICSkinCancerAtkAUTOPGD', 'ISICSkinCancerAtkCW']:
                runner.test_atk()
            elif config.data.dataset in ['FashionMNIST', 'MNIST', 'CIFAR10', 'CIFAR100', 'IMAGENE100']:
                y_majority_vote_accuracy_all_steps_list = runner.test_image_task()
            else:
                y_majority_vote_accuracy_all_steps_list = runner.test()
            procedure = "Testing"
        elif args.calib:
            if config.data.dataset in ['ChestXRay', 'ChestXRayAtkFGSM', 'ChestXRayAtkPGD', 'ChestXRayAtkBIM',
                                       'ChestXRayAtkAUTOPGD', 'ChestXRayAtkCW', 'ChestXRayValidate', 'ISICSkinCancer',
                                       'ISICSkinCancerAtkFGSM', 'ISICSkinCancerAtkPGD', 'ISICSkinCancerAtkBIM',
                                       'ISICSkinCancerAtkAUTOPGD', 'ISICSkinCancerAtkCW', 'ISICSkinCancerValidate']:
                t_initial = 0.2555
                res = minimize(runner.test_calibrate, t_initial, method='Nelder-Mead',
                               options={'xatol': 1e-4, 'fatol': 1e-5, 'disp': True})
                optimal_t = res.x[0]
                print("Optimal t value: {:.4f}".format(optimal_t))
            procedure = "Testing"
        else:
            runner.train(mlp_idx=args.mlp_idx)
            procedure = "Training"
        end_time = time.time()
        logging.info("\n{} procedure finished. It took {:.4f} minutes.\n\n\n".format(
            procedure, (end_time - start_time) / 60))
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    args.doc = args.doc + "/split_" + str(args.split)
    sys.exit(main())
