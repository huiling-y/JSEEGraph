#!/usr/bin/env python3
# coding=utf-8

from utility.schedule.inverse_sqrt_lr import InverseSqrtLr
from utility.schedule.linear_lr import LinearLr


#def multi_scheduler_wrapper(optimizer, args):
#    return MultiScheduler(
#        [
#            InverseSqrtLr(
#                optimizer.param_groups[i], args.encoder_learning_rate * (args.layerwise_lr_decay ** i), args.warmup_steps, args.encoder_delay_steps
#            )
#            for i in range(len(optimizer.param_groups) - 1)
#        ]
#        +
#        [
#            InverseSqrtLr(optimizer.param_groups[-1], args.decoder_learning_rate, args.warmup_steps, args.decoder_delay_steps)
#        ]
#    )

def multi_scheduler_wrapper(optimizer, args, steps_per_epoch):
    n_layers = (len(optimizer.param_groups) - 2) // 2

    return MultiScheduler(
        [
            LinearLr(optimizer.param_groups[i], args.encoder_learning_rate * (args.layerwise_lr_decay ** i), args.epochs * steps_per_epoch, False, args.lr_decay_multiplier)
            for i in range(n_layers)
        ]
        +
        [
            LinearLr(optimizer.param_groups[n_layers + i], args.encoder_learning_rate * (args.layerwise_lr_decay ** i), args.epochs * steps_per_epoch, False, args.lr_decay_multiplier)
            for i in range(n_layers)
        ]
        +
        [
            LinearLr(optimizer.param_groups[-2], args.decoder_learning_rate, args.epochs * steps_per_epoch, False, args.lr_decay_multiplier),
            LinearLr(optimizer.param_groups[-1], args.decoder_learning_rate, args.epochs * steps_per_epoch, False, args.lr_decay_multiplier)
        ]
    )


class MultiScheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def __call__(self, epoch):
        for scheduler in self.schedulers:
            scheduler(epoch)

    def lr(self) -> float:
        return [scheduler.lr() for scheduler in self.schedulers]
