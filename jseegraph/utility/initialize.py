import random
import torch
import os


def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize(args, init_wandb: bool):
    seed_everything(args.seed)

    if init_wandb:
        import wandb
        tags = {x for f in args.frameworks for x in f}
        print(tags)
        #wandb.init(name=f"{args.graph_mode}_{args.name}", config=args, project="IEGraph", tags=list(tags))
        wandb.init(name=args.name, config=args.get_hyperparameters(), project="IEGraph_Final", tags=list(tags))
        args.get_hyperparameters().save("config.json")
        wandb.save("config.json")
        print("Connection to Weights & Biases initialized.", flush=True)
