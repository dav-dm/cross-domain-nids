import numpy as np
import pandas as pd
import shap
import torch
from tqdm import tqdm

from .perturbation_utils import (
    compute_alpha_per_feat,
    get_proto_from_quintuple,
    get_random_background,
    protocol_applicability_mask,
    query_label,
    raw_bounds_by_proto,
)
from .transform_utils import (
    from_norm_to_raw,
    from_raw_to_norm,
    project,
)


class Perturbation:
    def __init__(self, method, approach, loader, train_xy, beta, query_budget, feat_mask, hsj_args):
        self.method = method
        self.approach = approach
        self.loader = loader
        self.query_budget = query_budget
        # Shared budget alias used by each method's loop limit.
        self.max_iter_rand = self.max_iter_shap = self.max_iter_pgd = query_budget
        self.feat_mask = feat_mask

        self.init_num_evals = hsj_args['hsj_init_evals']
        self.max_num_evals = hsj_args['hsj_max_evals']
        self.gamma = hsj_args['hsj_gamma']

        self.alpha_mask = torch.tensor(
            compute_alpha_per_feat(self.feat_mask, beta=beta),
            device=approach.device,
        )

        self.background_x, self.background_y = get_random_background(
            train_xy, n_samples=100, device=approach.device
        )

        if method == 'shap':
            self.explainer = shap.GradientExplainer(approach.net, self.background_x)

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _get_mask(self, x_orig):
        """
        Return a (T, 1) float mask: 1 for real uplink packets, 0 for padding
        and downlink packets.
        """
        padding_norm = torch.tensor(
            [0, 0, 0.5, 0, 0, 0], dtype=x_orig.dtype, device=x_orig.device
        )
        dir_idx = 2
        padding_mask = (x_orig == padding_norm).all(dim=-1)
        uplink_mask = (x_orig[..., dir_idx] == 0.0) & ~padding_mask
        return uplink_mask.float().unsqueeze(-1)

    def _feature_mask_tensor(self, x):
        """Return the feature-level mutability mask as a float tensor."""
        return torch.tensor(self.feat_mask, dtype=x.dtype, device=x.device)

    def _mutable_mask(self, x_orig, proto):
        """
        Return the combined mutability mask of shape (T, F):
          1 = real uplink packet AND mutable feature AND applicable to *proto*
          0 = padding / downlink / immutable feature / non-applicable feature
        """
        packet_mask = self._get_mask(x_orig)
        feature_mask = self._feature_mask_tensor(x_orig)
        proto_mask = protocol_applicability_mask(x_orig, proto)
        return packet_mask * feature_mask * proto_mask

    # ------------------------------------------------------------------
    # Constraint enforcement
    # ------------------------------------------------------------------

    def _repair_real_packet_raw_values(self, x_proj, x_orig, mutable_mask, proto):
        """
        Enforce raw-domain bounds inside the mutable mask and restore the
        original values everywhere else.

        Steps
        ~~~~~
        1. Convert both tensors to raw space.
        2. Clip mutable positions to the protocol-specific valid range.
        3. Restore immutable positions to their original raw values.
        4. Convert back to normalised space and apply the mask one final time
        """
        orig_shape = x_orig.shape
        F = orig_shape[-1]

        x_proj_np = x_proj.detach().cpu().numpy().reshape(-1, F)
        x_orig_np = x_orig.detach().cpu().numpy().reshape(-1, F)
        mutable_mask_np = mutable_mask.detach().cpu().numpy().reshape(-1, F).astype(bool)

        x_orig_raw = from_norm_to_raw(x_orig_np, x_orig_np, self.feat_mask)
        x_proj_raw = from_norm_to_raw(x_proj_np, x_orig_np, self.feat_mask)

        raw_min, raw_max = raw_bounds_by_proto(proto, x_proj_raw.dtype)

        x_proj_raw = np.where(
            mutable_mask_np,
            np.clip(x_proj_raw, raw_min, raw_max),
            x_orig_raw,
        )

        x_repaired = from_raw_to_norm(x_proj_raw, orig_shape, x_orig.dtype, x_orig.device)
        return x_orig + (x_repaired - x_orig) * mutable_mask

    def _project_constrained(self, x_candidate, x_ref, x_orig, mutable_mask, proto):
        """
        Single entry-point that applies *all* domain constraints:

        1. Quantise through raw space (symlog grid).
        2. Restore features outside the mutable mask to their original values.
        3. Clip mutable raw values to protocol-valid bounds (removes stray -1s
           on real packets, etc.).
        """
        x_proj = project(
            x_candidate, x_ref, self.feat_mask,
            x_orig.shape, x_orig.dtype, x_orig.device,
        )
        # Step 2: immutable features revert to original.
        x_proj = x_orig + (x_proj - x_orig) * mutable_mask
        # Step 3: raw-value repair inside the mask.
        return self._repair_real_packet_raw_values(x_proj, x_orig, mutable_mask, proto)

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _make_result_arrays(self, x_orig, delta, proto):
        """
        Produce a consistent set of metrics in both normalised and raw domains.
        """
        F = x_orig.shape[-1]
        x_orig_np = x_orig.squeeze(0).detach().cpu().numpy().reshape(-1, F)
        delta_np = delta.squeeze(0).detach().cpu().numpy().reshape(-1, F)
        x_adv_np = x_orig_np + delta_np

        mutable_mask = self._mutable_mask(x_orig, proto)
        mutable_mask_np = mutable_mask.detach().cpu().numpy().reshape(-1, F).astype(bool)

        # Safety in the normalised domain: restore immutable positions.
        x_adv_np = np.where(mutable_mask_np, x_adv_np, x_orig_np)

        x_orig_raw = from_norm_to_raw(x_orig_np, x_orig_np, self.feat_mask)
        x_adv_raw = from_norm_to_raw(x_adv_np, x_orig_np, self.feat_mask)

        raw_min, raw_max = raw_bounds_by_proto(proto, x_adv_raw.dtype)
        x_adv_raw = np.where(
            mutable_mask_np,
            np.clip(x_adv_raw, raw_min, raw_max),
            x_orig_raw,
        )

        # Recompute normalised adv from the repaired raw version.
        x_adv_t = from_raw_to_norm(x_adv_raw, x_orig.shape, x_orig.dtype, x_orig.device)
        x_adv_np = x_adv_t.squeeze(0).detach().cpu().numpy().reshape(-1, F)
        x_adv_np = np.where(mutable_mask_np, x_adv_np, x_orig_np)

        delta_np = x_adv_np - x_orig_np
        delta_raw = x_adv_raw - x_orig_raw

        return {
            'x_orig_norm':      x_orig_np.astype(np.float32),
            'x_adv_norm':       x_adv_np.astype(np.float32),
            'delta_norm':       delta_np.astype(np.float32),
            'x_orig_raw':       x_orig_raw.astype(np.float32),
            'x_adv_raw':        x_adv_raw.astype(np.float32),
            'delta_raw':        delta_raw.astype(np.float32),
            'delta_l2':         float(np.linalg.norm(delta_np)),
            'delta_per_feat':   np.abs(delta_np).mean(axis=0).astype(np.float32),
            'delta_raw_l2':     float(np.linalg.norm(delta_raw)),
            'delta_raw_per_feat': np.abs(delta_raw).mean(axis=0).astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Attack methods
    # ------------------------------------------------------------------

    def perturb_hsj(self, x_orig, proto):
        """
        HopSkipJump attack (Chen et al., 2020) for decision-based black-box evasion.

          Algorithm overview
          ~~~~~~~~~~~~~~~~~~
          1. Find a feasible benign background sample, restore the immutable
              features from ``x_orig``, and project it into valid space.
          2. Binary-search between the malicious input and the benign sample to
              locate the decision boundary.
          3. At each iteration, estimate the boundary gradient with Monte-Carlo
              queries, take a shrinking step toward the benign side, and
              binary-search back to the boundary.

          The probe radius is ``sqrt(d) * theta * dist`` and the step size is
          ``dist / sqrt(iteration)``. The query budget is enforced through
          ``BudgetExceeded``; every model call uses ``_query_label``.
        """
        orig_shape = x_orig.shape
        dtype, device = x_orig.dtype, x_orig.device

        mutable_mask = self._mutable_mask(x_orig, proto)
        d = float((mutable_mask > 0).sum().item())
        if d <= 0:
            return torch.zeros_like(x_orig), 0, False

        n_queries = 0
        # theta controls when binary search is considered converged.
        theta = self.gamma / (np.sqrt(d) * d)

        class BudgetExceeded(Exception):
            pass

        def _query_label(x):
            nonlocal n_queries
            if n_queries >= self.query_budget:
                raise BudgetExceeded
            label = query_label(x, self.approach)
            n_queries += 1
            return label

        def _project(x_pert, x_ref):
            return self._project_constrained(
                x_candidate=x_pert,
                x_ref=x_ref,
                x_orig=x_orig,
                mutable_mask=mutable_mask,
                proto=proto,
            )

        def _binary_search_boundary(x_a, x_b):
            """
            Linear-interpolation binary search between x_a (malicious) and
            x_b (benign).  Stops when the interpolation interval is < theta.
            Returns the best benign point observed so far.
            """
            lo, hi = 0.0, 1.0
            x_best_benign = x_b

            while (hi - lo) > theta:
                mid = (lo + hi) / 2.0
                x_mid = _project((1 - mid) * x_a + mid * x_b, x_a)
                try:
                    label = _query_label(x_mid)
                except BudgetExceeded:
                    return x_best_benign

                if label == 1:
                    lo = mid          # still malicious → move toward benign
                else:
                    hi = mid          # benign → move toward malicious
                    x_best_benign = x_mid.clone()

            return x_best_benign

        x_boundary = None
        MAX_INIT_TRIES = 10

        try:
            benign_idx = (self.background_y == 0).nonzero(as_tuple=True)[0]
            for _ in range(MAX_INIT_TRIES):
                idx = benign_idx[torch.randint(len(benign_idx), (1,)).item()]
                cand = self.background_x[idx].unsqueeze(0).to(device=device, dtype=dtype)
                cand = x_orig + (cand - x_orig) * mutable_mask  # Build a candidate by restoring immutable features from x_orig
                x_benign = _project(cand, x_orig)  # Project into valid space
                if _query_label(x_benign) == 0:
                    x_boundary = _binary_search_boundary(x_orig, x_benign)  # If benign, binary search to find the boundary
                    break

            if x_boundary is None: 
                return torch.zeros_like(x_orig), n_queries, False  # Failed to find a feasible init point

            for step in range(1, self.query_budget):
                if n_queries >= self.query_budget:
                    break

                dist = (x_boundary - x_orig).norm().item()
                delta_probe = 0.1 if step == 1 else np.sqrt(d) * theta * dist

                remaining_queries = self.query_budget - n_queries
                if remaining_queries <= 0:
                    break

                num_evals = int(
                    min(
                        self.init_num_evals * np.sqrt(step),
                        self.max_num_evals,
                        remaining_queries,
                    )
                )
                if num_evals <= 0:
                    break

                # Monte-Carlo gradient estimation — only perturb mutable features.
                rv = torch.randn(num_evals, *orig_shape, device=device, dtype=dtype) * mutable_mask
                rv_norm = rv.view(num_evals, -1).norm(dim=1).view(
                    num_evals, *([1] * len(orig_shape))
                )
                rv = rv / (rv_norm + 1e-8)

                decisions = torch.zeros(num_evals, dtype=dtype, device=device)
                actual_evals = 0
                for i in range(num_evals):
                    x_probe = _project(
                        (x_boundary + delta_probe * rv[i]).clamp(0, 1), x_boundary
                    )
                    decisions[i] = float(_query_label(x_probe))
                    actual_evals += 1

                if actual_evals == 0:
                    break

                decisions = decisions[:actual_evals]
                rv = rv[:actual_evals]
                # fval: benign → +1, malicious → −1.
                fval = 1.0 - 2.0 * decisions
                fval_mean = fval.mean()

                if fval_mean.abs() == 1.0:
                    gradf = rv.mean(dim=0) * fval_mean
                else:
                    fval = fval - fval_mean   # baseline subtraction
                    gradf = (
                        fval.view(actual_evals, *([1] * len(orig_shape))) * rv
                    ).mean(dim=0)

                gradf = gradf * mutable_mask
                g_norm = gradf.norm()
                if g_norm < 1e-8:
                    continue
                gradf = gradf / g_norm

                # Geometric step-size search: halve until we land on the benign side.
                epsilon = dist / np.sqrt(float(step))

                def _on_benign(eps):
                    x_new = _project((x_boundary + eps * gradf).clamp(0, 1), x_boundary)
                    return _query_label(x_new) == 0, x_new

                on_benign, x_new = _on_benign(epsilon)
                while not on_benign and epsilon > 1e-8:
                    epsilon /= 2.0
                    on_benign, x_new = _on_benign(epsilon)

                if not on_benign:
                    continue

                x_cand = _binary_search_boundary(x_orig, x_new)
                if (x_cand - x_orig).norm() < (x_boundary - x_orig).norm():
                    x_boundary = x_cand
                    
        except BudgetExceeded:
            pass

        if x_boundary is None:  # Failed to find a feasible init point
            return torch.zeros_like(x_orig), n_queries, False
        return (x_boundary - x_orig).detach(), n_queries, True


    def perturb_pgd(self, x_orig, proto):
        """
        Projected Gradient Descent — white-box attack (requires gradients).

        Moves in the direction of the loss gradient for the malicious class,
        then projects back into the valid space via ``_project_constrained``.
        """
        x_norm = x_orig.clone()
        n_queries = 0

        mutable_mask = self._mutable_mask(x_orig, proto)
        alpha = self.alpha_mask.to(x_orig.dtype)
        target = torch.ones(1, dtype=torch.long, device=x_orig.device)

        for _ in range(1, self.max_iter_pgd):
            n_queries += 1
            if query_label(x_norm, self.approach) == 0:
                return (x_norm - x_orig).detach(), n_queries, True

            x_in = x_norm.detach().requires_grad_(True)
            loss, _ = self.approach._predict_step(x_in, target)
            loss.backward()

            grad = x_in.grad * mutable_mask
            grad = grad / (grad.norm() + 1e-8)

            x_candidate = (x_norm + alpha * grad).clamp(0, 1)
            x_norm = self._project_constrained(
                x_candidate=x_candidate,
                x_ref=x_norm,
                x_orig=x_orig,
                mutable_mask=mutable_mask,
                proto=proto,
            )

        n_queries += 1
        evaded = query_label(x_norm, self.approach) == 0
        return (x_norm - x_orig).detach(), n_queries, evaded

    def perturb_shap(self, x_orig, proto):
        """
        SHAP GradientExplainer guided descent — white-box attack.

        At each iteration the SHAP values approximate the feature importance
        for the malicious class; we step *against* them to reduce that class's
        score.
        """
        x_norm = x_orig.clone()
        n_queries = 0

        mutable_mask = self._mutable_mask(x_orig, proto)
        alpha = self.alpha_mask.to(x_orig.dtype)

        for _ in range(1, self.max_iter_shap):
            n_queries += 1
            if query_label(x_norm, self.approach) == 0:
                return (x_norm - x_orig).detach(), n_queries, True

            shap_values = self.explainer.shap_values(x_norm)
            shap_arr = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_t = torch.tensor(shap_arr, dtype=x_norm.dtype, device=x_norm.device)

            x_candidate = (x_norm - alpha * shap_t * mutable_mask).clamp(0, 1)
            x_norm = self._project_constrained(
                x_candidate=x_candidate,
                x_ref=x_norm,
                x_orig=x_orig,
                mutable_mask=mutable_mask,
                proto=proto,
            )

        n_queries += 1
        evaded = query_label(x_norm, self.approach) == 0
        return (x_norm - x_orig).detach(), n_queries, evaded

    def perturb_random(self, x_orig, proto):
        """
        Random-noise baseline — applies uniform noise in the mutable directions.

        Useful as a lower-bound reference against the guided methods.
        """
        x_norm = x_orig.clone()
        n_queries = 0

        mutable_mask = self._mutable_mask(x_orig, proto)
        alpha = self.alpha_mask.to(x_orig.dtype)

        for _ in range(1, self.max_iter_rand):
            n_queries += 1
            if query_label(x_norm, self.approach) == 0:
                return (x_norm - x_orig).detach(), n_queries, True

            noise = (torch.rand_like(x_norm) * 2 - 1) * alpha * mutable_mask
            x_candidate = (x_norm + noise).clamp(0, 1)
            x_norm = self._project_constrained(
                x_candidate=x_candidate,
                x_ref=x_norm,
                x_orig=x_orig,
                mutable_mask=mutable_mask,
                proto=proto,
            )

        n_queries += 1
        evaded = query_label(x_norm, self.approach) == 0
        return (x_norm - x_orig).detach(), n_queries, evaded

    # ------------------------------------------------------------------
    # Dispatch and main loop
    # ------------------------------------------------------------------

    def perturb_single(self, x_orig, proto):
        """Dispatch to the configured perturbation method."""
        if self.method == 'shap':
            return self.perturb_shap(x_orig, proto)
        if self.method == 'random':
            return self.perturb_random(x_orig, proto)
        if self.method == 'hsj':
            return self.perturb_hsj(x_orig, proto)
        if self.method == 'pgd':
            return self.perturb_pgd(x_orig, proto)
        raise ValueError(f'Unknown perturbation method: {self.method!r}')

    def perturbate(self):
        """
        Collect all true-positive predictions from the test loader, perturb
        each one, and return results as a DataFrame.

        At most ``MAX_TP`` samples are processed (randomly subsampled when the
        dataset is larger).
        """
        MAX_TP = 10_000

        self.approach.net.eval()
        all_tp_x = []
        all_tp_q = []

        for batch_x, batch_y, batch_q in tqdm(self.loader, desc=f'[{self.method}] collecting TPs'):
            batch_x = batch_x.to(self.approach.device)
            batch_y = batch_y.to(self.approach.device)

            with torch.no_grad():
                _, logits = self.approach._predict_step(batch_x, batch_y.long())
                preds = torch.argmax(logits, dim=1)

            tp_mask = (batch_y == 1) & (preds == 1)
            if not tp_mask.any():
                continue

            all_tp_x.append(batch_x[tp_mask])
            all_tp_q.extend([q for q, m in zip(batch_q, tp_mask.tolist()) if m])

        all_tp_x = torch.cat(all_tp_x, dim=0)

        if len(all_tp_x) > MAX_TP:
            idx = torch.randperm(len(all_tp_x))[:MAX_TP]
            all_tp_x = all_tp_x[idx]
            all_tp_q = [all_tp_q[i] for i in idx.tolist()]

        rows = []
        for i in tqdm(range(len(all_tp_x)), desc=f'[{self.method}] perturbing'):
            x_orig = all_tp_x[i].unsqueeze(0)
            quintuple = all_tp_q[i]
            proto = get_proto_from_quintuple(quintuple)

            delta, n_queries, evaded = self.perturb_single(x_orig, proto)
            arrays = self._make_result_arrays(x_orig, delta, proto)

            rows.append({
                'method':    self.method,
                'quintuple': quintuple,
                'evaded':    evaded,
                'n_queries': n_queries,

                # Normalised-domain metrics
                'delta_l2':       arrays['delta_l2'],
                'delta_per_feat': arrays['delta_per_feat'],
                'x_orig_norm':    arrays['x_orig_norm'],
                'x_adv_norm':     arrays['x_adv_norm'],
                'delta_norm':     arrays['delta_norm'],

                # Raw-domain metrics
                'x_orig_raw':         arrays['x_orig_raw'],
                'x_adv_raw':          arrays['x_adv_raw'],
                'delta_raw':          arrays['delta_raw'],
                'delta_raw_per_feat': arrays['delta_raw_per_feat'],
                'delta_raw_l2':       arrays['delta_raw_l2'],
            })

        print(f'Total TPs processed: {len(all_tp_x)}')
        return pd.DataFrame(rows)