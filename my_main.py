#!/usr/bin/env python3
import torch
from torch import optim

import options
from data.load import get_multitask_experiment
import define_models as define
from train import train_cl
from eval import evaluate
from eval import callbacks as cb

## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': False, 'generative': True, 'compare_code': 'none'}
    # Define input options
    parser = options.define_args(filename="main_cl", description='Compare & combine continual learning approaches.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_bir_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    options.check_for_errors(args, **kwargs)
    return args


args = handle_inputs()
device = torch.device("cuda")

# Prepare data for chosen experiment
(train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
    name='splitMNIST', scenario='class', tasks=args.tasks,
    normalize=False,
    augment=False,
    verbose=True, exception=False, only_test=False
)

model = define.define_autoencoder(args=args, config=config, device=device)
model = define.init_params(model, args)
# Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
model.replay_targets = "soft" if args.distill else "hard"
model.KD_temp = args.temp
model.optim_list = [
    {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr},
]
model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
generator = None

# ---------------------#
# ----- CALLBACKS -----#
# ---------------------#

visdom = None
train_gen = False
# Callbacks for reporting on and visualizing loss
generator_loss_cbs = [
    cb._VAE_loss_cb(log=args.loss_log, visdom=visdom, replay=True, model=model, tasks=args.tasks, iters_per_task=args.iters)
]

train_cl(
    model, train_datasets, replay_mode=args.replay if hasattr(args, 'replay') else "none",
    scenario=args.scenario, classes_per_task=classes_per_task, iters=args.iters,
    batch_size=args.batch, batch_size_replay=args.batch_replay if hasattr(args, 'batch_replay') else None,
    generator=generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs,
    feedback=True, sample_cbs=[None], eval_cbs=[None],
    loss_cbs=generator_loss_cbs,
    args=args, reinit=False, only_last=False
)

# Evaluate precision of final model on full test-set
precs = [evaluate.validate(
    model, test_datasets[i], verbose=False, test_size=None, task=i + 1,
    allowed_classes=list(range(classes_per_task * i, classes_per_task * (i + 1))) if args.scenario == "task" else None
) for i in range(args.tasks)]
average_precs = sum(precs) / args.tasks

print("\n Accuracy of final model on test-set:")
for i in range(args.tasks):
    print(" - {} {}: {:.4f}".format("For classes from task" if args.scenario == "class" else "Task",
                                    i + 1, precs[i]))
print('=> Average accuracy over all {} {}: {:.4f}\n'.format(
    args.tasks * classes_per_task if args.scenario == "class" else args.tasks,
    "classes" if args.scenario == "class" else "tasks", average_precs
))

# python my_main.py --experiment=splitMNIST --scenario=class --replay=generative --brain-inspired
