import numpy as np


class kl_annealing:
    def __init__(self, args, current_epoch=0):
        self.annealing_type = args.kl_anneal_type
        assert self.annealing_type in ["Cyclical", "Monotonic", "Without"]
        self.iter = current_epoch + 1

        if self.annealing_type == "Cyclical":
            self.L = self.frange_cycle_linear(
                num_epoch=args.num_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=args.kl_anneal_cycle,
                ratio=args.kl_anneal_ratio,
            )
        elif self.annealing_type == "Monotonic":
            self.L = self.frange_cycle_linear(
                num_epoch=args.num_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=1,
                ratio=args.kl_anneal_ratio,
            )
        else:
            # Use np.ones for consistency, ensure length matches num_epoch + 1
            self.L = np.ones(args.num_epoch + 1)

    def update(self):
        self.iter += 1

    def get_beta(self):
        # Ensure iter does not go out of bounds
        current_iter = min(self.iter, len(self.L) - 1)
        return self.L[current_iter]

    def frange_cycle_linear(self, num_epoch, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        # adapted from https://github.com/haofuml/cyclical_annealing
        L = np.ones(num_epoch + 1)
        period = num_epoch / n_cycle
        step = (stop - start) / (period * ratio)

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop:
                idx = int(i + c * period)
                if idx < len(L):
                    L[idx] = v
                else:
                    break # Stop if index exceeds array bounds
                v += step
                i += 1
        return L 