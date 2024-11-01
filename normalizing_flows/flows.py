from copy import deepcopy
from typing import Union, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from normalizing_flows.bijections.base import Bijection
from normalizing_flows.bijections.continuous.ddnf import DeepDiffeomorphicBijection
from normalizing_flows.regularization import reconstruction_error
from normalizing_flows.utils import flatten_event, get_batch_shape, unflatten_event, create_data_loader


class Flow(nn.Module):
    """
    Normalizing flow class.

    This class represents a bijective transformation of a standard Gaussian distribution (the base distribution).
    A normalizing flow is itself a distribution which we can sample from or use it to compute the density of inputs.
    """

    def __init__(self, bijection: Bijection, record_Ihat_P=False, record_log_px=False, **kwargs):
        """

        :param bijection: transformation component of the normalizing flow.
        """
        super().__init__()
        self.register_module('bijection', bijection)
        self.register_buffer('loc', torch.zeros(self.bijection.n_dim))
        self.register_buffer('covariance_matrix', torch.eye(self.bijection.n_dim))

        # additional parameters for debugging and result visualization
        self.record_Ihat_P = record_Ihat_P
        self.Ihat_P = []
        self.record_log_px = record_log_px
        self.log_px = []

    @property
    def base(self) -> torch.distributions.Distribution:
        """
        :return: base distribution of the normalizing flow.
        """
        return torch.distributions.MultivariateNormal(loc=self.loc, covariance_matrix=self.covariance_matrix)

    def base_log_prob(self, z: torch.Tensor):
        """
        Compute the log probability of input z under the base distribution.

        :param z: input tensor.
        :return: log probability of the input tensor.
        """
        zf = flatten_event(z, self.bijection.event_shape)
        log_prob = self.base.log_prob(zf)
        return log_prob

    def base_sample(self, sample_shape: Union[torch.Size, Tuple[int, ...]]):
        """
        Sample from the base distribution.

        :param sample_shape: desired shape of sampled tensor.
        :return: tensor with shape sample_shape.
        """
        z_flat = self.base.sample(sample_shape)
        z = unflatten_event(z_flat, self.bijection.event_shape)
        return z

    def forward_with_log_prob(self, x: torch.Tensor, context: torch.Tensor = None):
        """
        Transform the input x to the space of the base distribution.

        :param x: input tensor.
        :param context: context tensor upon which the transformation is conditioned.
        :return: transformed tensor and the logarithm of the absolute value of the Jacobian determinant of the
         transformation.
        """
        if context is not None:
            assert context.shape[0] == x.shape[0]
            context = context.to(self.loc)
        z, log_det = self.bijection.forward(x.to(self.loc), context=context)
        log_base = self.base_log_prob(z)
        return z, log_base + log_det

    def log_prob(self, x: torch.Tensor, context: torch.Tensor = None):
        """
        Compute the logarithm of the probability density of input x according to the normalizing flow.

        :param x: input tensor.
        :param context: context tensor.
        :return:
        """
        return self.forward_with_log_prob(x, context)[1]

    def sample(self, n: int, context: torch.Tensor = None, no_grad: bool = False, return_log_prob: bool = False):
        """
        Sample from the normalizing flow.

        If context given, sample n tensors for each context tensor.
        Otherwise, sample n tensors.

        :param n: number of tensors to sample.
        :param context: context tensor with shape c.
        :param no_grad: if True, do not track gradients in the inverse pass.
        :return: samples with shape (n, *event_shape) if no context given or (n, *c, *event_shape) if context given.
        """
        if context is not None:
            z = self.base_sample(sample_shape=torch.Size((n, len(context))))
            context = context[None].repeat(*[n, *([1] * len(context.shape))])  # Make context shape match z shape
            assert z.shape[:2] == context.shape[:2]
        else:
            z = self.base_sample(sample_shape=torch.Size((n,)))
        if no_grad:
            z = z.detach()
            with torch.no_grad():
                x, log_det = self.bijection.inverse(z, context=context)
        else:
            x, log_det = self.bijection.inverse(z, context=context)
        x = x.to(self.loc)

        if return_log_prob:
            log_prob = self.base_log_prob(z) + log_det
            return x, log_prob
        return x
    
    def calculate_Ihat_P(self, x: torch.Tensor, reduction: callable = torch.mean, random_seed: int = None, context: torch.Tensor = None):
        """ calculate Ihat_P for the Flow class so we can compare it with the Principal Manifold objective
        """

        # Evaluate log p(x) with the set prior
        if context is not None:
            assert context.shape[0] == x.shape[0]
            context = context.to(self.loc)
        z, log_det = self.bijection.forward(x.to(self.loc), context=context)
        log_pz = self.base_log_prob(z)
        log_px = log_pz + log_det
        
        # Sample an index in the partition
        z_dim = z.shape[-1]

        if random_seed is None:
            k = torch.randint(0, z_dim, (1,)).item()
        else:
            k = torch.randint(0, z_dim, (1,), generator=torch.Generator().manual_seed(random_seed)).item()
        k_onehot = torch.zeros(z_dim)
        k_onehot[k] = 1.0
        k_mask = k_onehot.repeat(z.shape[0], 1)

        # Compute an unbiased estimate of Ihat_P
        z.requires_grad_(True)
        Gk = torch.autograd.functional.vjp(lambda z: self.bijection.forward(z.to(self.loc), context=context)[0], z, v=k_mask)[1]
        GkGkT = torch.sum(Gk**2, dim=1)
        Ihat_P = -0.5*log_det + z_dim * 0.5 * torch.log(GkGkT)

        return Ihat_P, log_px

    def get_Ihat_P(self, as_tensor=True):
        # transform list to tensor
        if as_tensor:
            return torch.stack(self.Ihat_P)
        return self.Ihat_P
    
    def get_log_px(self, as_tensor=True):
        # transform list to tensor
        if as_tensor:
            return torch.stack(self.log_px)
        return self.log_px


    def fit(self,
            x_train: torch.Tensor,
            n_epochs: int = 500,
            lr: float = 0.05,
            batch_size: int = 1024,
            shuffle: bool = True,
            show_progress: bool = False,
            w_train: torch.Tensor = None,
            context_train: torch.Tensor = None,
            x_val: torch.Tensor = None,
            w_val: torch.Tensor = None,
            context_val: torch.Tensor = None,
            keep_best_weights: bool = True,
            early_stopping: bool = False,
            early_stopping_threshold: int = 50):
        """
        Fit the normalizing flow.

        Fitting the flow means finding the parameters of the bijection that maximize the probability of training data.
        Bijection parameters are iteratively updated for a specified number of epochs.
        If context data is provided, the normalizing flow learns the distribution of data conditional on context data.

        :param x_train: training data with shape (n_training_data, *event_shape).
        :param n_epochs: perform fitting for this many steps.
        :param lr: learning rate. In general, lower learning rates are recommended for high-parametric bijections.
        :param batch_size: in each epoch, split training data into batches of this size and perform a parameter update for each batch.
        :param shuffle: shuffle training data. This helps avoid incorrect fitting if nearby training samples are similar.
        :param show_progress: show a progress bar with the current batch loss.
        :param w_train: training data weights with shape (n_training_data,).
        :param context_train: training data context tensor with shape (n_training_data, *context_shape).
        :param x_val: validation data with shape (n_validation_data, *event_shape).
        :param w_val: validation data weights with shape (n_validation_data,).
        :param context_val: validation data context tensor with shape (n_validation_data, *context_shape).
        :param keep_best_weights: if True and validation data is provided, keep the bijection weights with the highest probability of validation data.
        :param early_stopping: if True and validation data is provided, stop the training procedure early once validation loss stops improving for a specified number of consecutive epochs.
        :param early_stopping_threshold: if early_stopping is True, fitting stops after no improvement in validation loss for this many epochs.
        """
        self.bijection.train()

        # Compute the number of event dimensions
        n_event_dims = int(torch.prod(torch.as_tensor(self.bijection.event_shape)))

        # Set the default batch size
        if batch_size is None:
            batch_size = len(x_train)

        # Process training data
        train_loader = create_data_loader(
            x_train,
            w_train,
            context_train,
            "training",
            batch_size=batch_size,
            shuffle=shuffle,
            event_shape=self.bijection.event_shape
        )

        # Process validation data
        if x_val is not None:
            val_loader = create_data_loader(
                x_val,
                w_val,
                context_val,
                "validation",
                batch_size=batch_size,
                shuffle=shuffle,
                event_shape=self.bijection.event_shape
            )

            best_val_loss = torch.inf
            best_epoch = 0
            best_weights = deepcopy(self.state_dict())

        def compute_batch_loss(batch_, reduction: callable = torch.mean):
            batch_x, batch_weights = batch_[:2]
            batch_context = batch_[2] if len(batch_) == 3 else None

            batch_log_prob = self.log_prob(batch_x.to(self.loc), context=batch_context)
            batch_weights = batch_weights.to(self.loc)
            assert batch_log_prob.shape == batch_weights.shape, f"{batch_log_prob.shape = }, {batch_weights.shape = }"
            batch_loss = -reduction(batch_log_prob * batch_weights) / n_event_dims

            if self.record_Ihat_P or self.record_log_px:
                Ihat_p, log_px = self.calculate_Ihat_P(batch_x, reduction=torch.mean, random_seed=None, context=batch_context)
                return batch_loss, Ihat_p, log_px

            return batch_loss

        iterator = tqdm(range(n_epochs), desc='Fitting NF', disable=not show_progress)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        val_loss = None

        for epoch in iterator:
            # create empty tensors to store Ihat_P and log_px
            Ihat_P_epoch = torch.empty(0)
            log_px_epoch = torch.empty(0)

            for train_batch in train_loader:
                optimizer.zero_grad()
                if self.record_Ihat_P or self.record_log_px:
                    train_loss, Ihat_p, log_px = compute_batch_loss(train_batch, reduction=torch.mean)
                    Ihat_P_epoch = torch.cat((Ihat_P_epoch, Ihat_p))
                    log_px_epoch = torch.cat((log_px_epoch, log_px))
                else:
                    train_loss = compute_batch_loss(train_batch, reduction=torch.mean)

                if hasattr(self.bijection, 'regularization'):
                    train_loss += self.bijection.regularization()
                train_loss.backward()
                optimizer.step()

                if show_progress:
                    if val_loss is None:
                        iterator.set_postfix_str(f'Training loss (batch): {train_loss:.4f}')
                    else:
                        iterator.set_postfix_str(
                            f'Training loss (batch): {train_loss:.4f}, '
                            f'Validation loss: {val_loss:.4f}'
                        )

            if self.record_Ihat_P:
                with torch.no_grad():
                    self.Ihat_P.append(Ihat_P_epoch.mean())
            if self.record_log_px:
                with torch.no_grad():
                    self.log_px.append(log_px_epoch.mean())
                

            # Compute validation loss at the end of each epoch
            # Validation loss will be displayed at the start of the next epoch
            if x_val is not None:
                with torch.no_grad():
                    # Compute validation loss
                    val_loss = 0.0
                    for val_batch in val_loader:
                        n_batch_data = len(val_batch[0])
                        val_loss += compute_batch_loss(val_batch, reduction=torch.sum) / n_batch_data
                    if hasattr(self.bijection, 'regularization'):
                        val_loss += self.bijection.regularization()

                    # Check if validation loss is the lowest so far
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch

                    # Store current weights
                    if keep_best_weights:
                        if best_epoch == epoch:
                            best_weights = deepcopy(self.state_dict())

                    # Optionally stop training early
                    if early_stopping:
                        if epoch - best_epoch > early_stopping_threshold:
                            break

        if x_val is not None and keep_best_weights:
            self.load_state_dict(best_weights)

        self.bijection.eval()

    def variational_fit(self,
                        target_log_prob: callable,
                        n_epochs: int = 500,
                        lr: float = 0.05,
                        n_samples: int = 1000,
                        show_progress: bool = False):
        """
        Train a normalizing flow with stochastic variational inference.
        Stochastic variational inference lets us train a normalizing flow using the unnormalized target log density
        instead of a fixed dataset.

        Refer to Rezende, Mohamed: "Variational Inference with Normalizing Flows" (2015) for more details
        (https://arxiv.org/abs/1505.05770, loss definition in Equation 15, training pseudocode for conditional flows in
         Algorithm 1).

        :param callable target_log_prob: function that computes the unnormalized target log density for a batch of
        points. Receives input batch with shape = (*batch_shape, *event_shape) and outputs batch with
         shape = (*batch_shape).
        :param int n_epochs: number of training epochs.
        :param float lr: learning rate for the AdamW optimizer.
        :param float n_samples: number of samples to estimate the variational loss in each training step.
        :param bool show_progress: if True, show a progress bar during training.
        """
        iterator = tqdm(range(n_epochs), desc='Variational NF fit', disable=not show_progress)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        for _ in iterator:
            optimizer.zero_grad()
            flow_x, flow_log_prob = self.sample(n_samples, return_log_prob=True)
            loss = -torch.mean(target_log_prob(flow_x) + flow_log_prob)
            if hasattr(self.bijection, 'regularization'):
                loss += self.bijection.regularization()
            loss.backward()
            optimizer.step()
            iterator.set_postfix_str(f'Variational loss: {loss:.4f}')


class DDNF(Flow):
    """
    Deep diffeomorphic normalizing flow.

    Salman et al. Deep diffeomorphic normalizing flows (2018).
    """

    def __init__(self, event_shape: torch.Size, **kwargs):
        bijection = DeepDiffeomorphicBijection(event_shape=event_shape, **kwargs)
        super().__init__(bijection)

    def fit(self,
            x_train: torch.Tensor,
            n_epochs: int = 500,
            lr: float = 0.05,
            batch_size: int = 1024,
            shuffle: bool = True,
            show_progress: bool = False,
            w_train: torch.Tensor = None,
            rec_err_coef: float = 1.0):
        """

        :param x_train:
        :param n_epochs:
        :param lr: learning rate. In general, lower learning rates are recommended for high-parametric bijections.
        :param batch_size:
        :param shuffle:
        :param show_progress:
        :param w_train: training data weights
        :param rec_err_coef: reconstruction error regularization coefficient.
        :return:
        """
        if w_train is None:
            batch_shape = get_batch_shape(x_train, self.bijection.event_shape)
            w_train = torch.ones(batch_shape)
        if batch_size is None:
            batch_size = len(x_train)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        dataset = TensorDataset(x_train, w_train)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        n_event_dims = int(torch.prod(torch.as_tensor(self.bijection.event_shape)))

        if show_progress:
            iterator = tqdm(range(n_epochs), desc='Fitting NF')
        else:
            iterator = range(n_epochs)

        for _ in iterator:
            for batch_x, batch_w in data_loader:
                optimizer.zero_grad()

                z, log_prob = self.forward_with_log_prob(batch_x.to(self.loc))  # TODO context!
                w = batch_w.to(self.loc)
                assert log_prob.shape == w.shape
                loss = -torch.mean(log_prob * w) / n_event_dims

                if hasattr(self.bijection, 'regularization'):
                    # Always true for DeepDiffeomorphicBijection, but we keep it for clarity
                    loss += self.bijection.regularization()

                # Inverse consistency regularization
                x_reconstructed = self.bijection.inverse(z)
                loss += reconstruction_error(batch_x, x_reconstructed, self.bijection.event_shape, rec_err_coef)

                # Geodesic regularization

                loss.backward()
                optimizer.step()

                if show_progress:
                    iterator.set_postfix_str(f'Loss: {loss:.4f}')



class PrincipalManifoldFlow(Flow):
    """
    Principal manifold flow.

    This class represents a normalizing flow that learns the principal manifold of the input data.
    """

    def __init__(self, bijection: Bijection, alpha=None, debug=False, record_Ihat_P=False, record_log_px=False, method="default", objective="brute_force", **kwargs):
        """
        :param bijection: transformation component of the normalizing flow.
        :param manifold_dim: dimensionality of the principal manifold.
        """

        # additional parameters for debugging and result visualization
        self.debug = debug
        # self.record_Ihat_P = record_Ihat_P
        # self.Ihat_P = []
        # self.record_log_px = record_log_px
        # self.log_px = []
        if alpha is None:
            self.alpha = 5.0
        else:
            self.alpha = alpha
        self.method = method
        self.objective = objective

        super().__init__(bijection, record_Ihat_P, record_log_px)



    def PF_objective_brute_force_old(self, x: torch.Tensor, P, context: torch.Tensor = None): 
        """ Brute force implementation of the PF objective for the Flow class.

        Inputs:
            x       - Batched input tensor
            P       - List of torch tensors that form a partition over range(x.size)
            alpha   - Regularization hyperparameter
            context - Context tensor

        Outputs:
            objective - PF's objective
        """
        # Evaluate log p(x) with the set prior
        if context is not None:
            assert context.shape[0] == x.shape[0]
            context = context.to(self.loc)
        z, log_det = self.bijection.forward(x.to(self.loc), context=context)
        log_pz = self.base_log_prob(z)
        log_px = log_pz + log_det
        
        # print("z shape: ", z.shape)
        # print("log_det shape: ", log_det.shape)
        # print("log_pz shape: ", log_pz.shape)
        # print("log_px shape: ", log_px.shape)
        # print("x shape: ", x.shape)

        # Create the Jacobian matrix for every item in the batch
        G = torch.autograd.functional.jacobian(lambda x: self.bijection.forward(x.to(self.loc), context=context), x)[0]
        # print("G shape: ", G.shape)

        # Compute Ihat_P for each element in the batch
        Ihat_P = -log_det
        for k in P:
            # print("k: ", k)
            Gk = G[:, k[0]]
            Gk_T = torch.transpose(Gk, -2, -1)
            GkGk_T = torch.matmul(Gk, Gk_T)
            # print("GkGk_T shape: ", GkGk_T.shape)
            # print("Gk shape: ", Gk.shape)
            # print("Gk_T shape: ", Gk_T.shape)
            # print("GkGk_T: ", GkGk_T)
            # print("slogdet shape: ", torch.slogdet(GkGk_T)[1].shape)
            Ihat_P += 0.5 * torch.slogdet(GkGk_T)[1]

        # print("Ihat_P shape: ", Ihat_P.shape)
        # print("Ihat_P: ", Ihat_P)
        # print("log_px shape: ", log_px.shape)
        # print("log_px: ", log_px)
        # print("alpha: ", self.alpha)
        objective = -log_px + self.alpha * Ihat_P

        # print("objective shape: ", objective.shape)
        # print("objective: ", objective)

        if self.record_Ihat_P or self.record_log_px:
            return objective.mean(), Ihat_P, log_px

        return objective.mean()

    

    def PF_objective_unbiased_old(self, x: torch.Tensor, reduction: callable = torch.mean, random_seed: int = None, context: torch.Tensor = None):
        """ Unbiased estimate of the PF objective when the partition size is 1 for the Flow class.

        Inputs:
            x       - Unbatched 1d input (b, d) where b is the batch size and d is the dimensionality of the input
            rng_key - Torch random generator
            alpha   - Regularization hyperparameter

        Outputs:
            objective - PFs objective
        """

        # Evaluate log p(x) with the set prior
        if context is not None:
            assert context.shape[0] == x.shape[0]
            context = context.to(self.loc)

        def apply_func(x):
            z, log_det = self.bijection.forward(x.to(self.loc), context=context)
            log_pz = self.base_log_prob(z)
            return z, (z, log_det, log_pz)
        
        G, (z, log_det, log_pz) = torch.func.jacrev(apply_func, has_aux=True)(x)
        # log_px = log_pz + 0.5*log_det # This was our version
        log_px = log_pz + log_det # taken from author's code

        # Sample an index in the partition
        # print("z shape: ", z.shape)
        z_dim = z.shape[-1]
        batch_size = z.shape[0]

        if self.debug:
            print("z_dim: ", z_dim)

        if random_seed is None:
            k = torch.randint(0, z_dim, (batch_size,))
        else:
            k = torch.randint(0, z_dim, (batch_size,), generator=torch.Generator().manual_seed(random_seed))
        k_onehot = torch.eye(z_dim, dtype=torch.int32)[k]

        if self.debug:
            print("k shape: ", k.shape)
            print("k_onehot shape: ", k_onehot.shape)
            print("k_onehot: ", k_onehot)

        # Compute an unbiased estimate of Ihat_P
        # Gk = torch.autograd.functional.vjp(lambda z: self.bijection.forward(z.to(self.loc), context=context)[0], z, v=k_mask)[1]
        # GkGkT = torch.sum(Gk**2, dim=1)
            
            
        GkGkT = torch.empty(batch_size)
        test1GkGkT = torch.empty(batch_size)
        test2GkGkT = torch.empty(batch_size)
        vjpGkGkT = torch.empty(batch_size)

        if self.method == "default":
            for i in range(len(x)):
                # Option 1: jacobi on whole function, then take [0] for z ([1] would be for log_det), and einsum to get correct shape
                # jacobi = torch.autograd.functional.jacobian(self.bijection.forward, x[i].unsqueeze(0))[0]
                # jacobi = torch.einsum("bibj->ij", jacobi)

                # Option 2 (FASTER): use lambda function to get only jacobi for z
                jacobi = torch.autograd.functional.jacobian(lambda x: self.bijection.forward(x.to(self.loc), context=context)[0], x[i].unsqueeze(0))
                Gk_i = torch.matmul(jacobi, k_onehot[i].float())

                GkGkT[i] = torch.sum(Gk_i**2)

        if self.method == "einsum":
            for i in range(len(x)):
                jacobi = torch.autograd.functional.jacobian(lambda x: self.bijection.forward(x.to(self.loc), context=context)[0], x[i].unsqueeze(0))
                testGk_i = torch.einsum('bibj,j->i', jacobi, k_onehot[i].float())
                test1GkGkT[i] = torch.sum(testGk_i**2)

        if self.method == "vjp":
            for i in range(len(x)):
                if i == 0:
                    print("x_i shape: ", x[i].unsqueeze(0).shape)
                    print("k_onehot[i] shape: ", k_onehot[i].unsqueeze(0).float().shape)
                    print("context: ", context)

                    print("x:_i: ", x[i])
                    print("x:_i: ", x[i].unsqueeze(0))
                    print("k_onehot[i]: ", k_onehot[i])
                    print("k_onehot[i]: ", k_onehot[i].shape)
                    print("k_onehot[i]: ", k_onehot[i].unsqueeze(0).float())
                vjpGkGkT[i] = torch.autograd.functional.vjp(lambda x_i: self.bijection.forward(x_i.to(self.loc), context=context)[0], x[i].unsqueeze(0), v=k_onehot[i].unsqueeze(0).float())[1]

        if self.method == "vectorized":
            Gk = torch.vmap(lambda x_i: torch.autograd.functional.vjp(lambda x: self.bijection.forward(x.to(self.loc), context=context)[0], x_i.unsqueeze(0), v=k_onehot.unsqueeze(-1).float())[1])(x)
            GkGkT = torch.sum(Gk**2, dim=1)


        # Ihat_P = -0.5*log_det + z_dim * 0.5 * torch.log(GkGkT) # This was our version
        Ihat_P = -log_det + 0.5 * torch.log(torch.sum(G ** 2, dim=-1)).sum()  # taken from author's code

        objective = -log_px + self.alpha * Ihat_P


        if self.record_Ihat_P or self.record_log_px:
            return reduction(objective), Ihat_P, log_px

        return reduction(objective)
    
    def PF_objective_brute_force(self, 
                                 x: torch.Tensor,
                                 reduction: callable = torch.mean, 
                                 random_seed: int = None, 
                                 context: torch.Tensor = None):
        if context is not None:
            assert context.shape[0] == x.shape[0]
            context = context.to(self.loc)

        x = x.clone().to(self.loc)
        x.requires_grad_(True)

        def _loss(_x):
            # _x = _x.clone().to(self.loc)
            # _x.requires_grad_(True)
            _z, _log_det = self.bijection.forward(_x.unsqueeze(0))
            _z = _z.squeeze(0)
            _log_det = _log_det.squeeze(0)

            jacobian = torch.stack([
                        torch.autograd.grad(
                            out_dim,
                            _x,
                            retain_graph=True,
                            allow_unused=True,
                            materialize_grads=True
                        )[0] for out_dim in _z
                    ])
            
            _Ihat_P = -_log_det + 0.5 * torch.log(torch.sum(jacobian ** 2, dim=0)).sum()
            _log_px = self.log_prob(_z.unsqueeze(0))

            objective = -_log_px + self.alpha * _Ihat_P
            
            return _Ihat_P, _log_px, objective

        results = [_loss(_x) for _x in x]
        Ihat_P, log_px, objectives = zip(*results)

        Ihat_P = torch.stack(Ihat_P)
        log_px = torch.stack(log_px)
        objective = torch.stack(objectives)

        if self.record_Ihat_P or self.record_log_px:
            return reduction(objective), Ihat_P, log_px
        else:
            return reduction(objective)
        

    def PF_objective_unbiased(self,
                              x: torch.Tensor,
                              reduction: callable = torch.mean,
                              random_seed: int = None,
                              context: torch.Tensor = None):
        if context is not None:
            assert context.shape[0] == x.shape[0]
            context = context.to(self.loc)

        x = x.clone().to(self.loc)
        x.requires_grad_(True)
        z, log_det = self.bijection.forward(x, context=context)
        log_prob = self.log_prob(z)

        def _jacobian(_y, _x):
            # Compute jacobian of f(_x) = _y
            # _x.shape == _y.shape == (d,)

            return torch.stack([
                torch.autograd.grad(
                    out_dim,
                    _x,
                    retain_graph=True,
                    allow_unused=True,
                    materialize_grads=True
                )[0] for out_dim in _y
            ])

        def batch_jacobian(outputs, inputs):
            # outputs.shape == inputs.shape == (n, d)
            return torch.stack([_jacobian(_y, _x) for (_y, _x) in zip(outputs, inputs)])  # shape == (n, d, d)

        jacobian = batch_jacobian(z, x)

        n_data, n_dim = x.shape
        assert jacobian.shape == (n_data, n_dim, n_dim)

        print("jacobian", jacobian)
        print("torch.sum(jacobian ** 2, dim=0)", torch.sum(jacobian ** 2, dim=0))
        print("torch.log(torch.sum(jacobian ** 2, dim=0))", torch.log(torch.sum(jacobian ** 2, dim=0)))
        print("log_det", log_det)
        print("log_prob", log_prob)

        Ihat_P = -log_det + 0.5 * torch.log(torch.sum(jacobian ** 2, dim=0)).sum()
        loss_value = reduction(-log_prob + self.alpha * Ihat_P)

        if self.record_Ihat_P or self.record_log_px:
            return loss_value, Ihat_P, log_prob
        else:
            return loss_value

    
    def get_Ihat_P(self, as_tensor=True):
        # transform list to tensor
        if as_tensor:
            return torch.stack(self.Ihat_P)
        return self.Ihat_P
    
    def get_log_px(self, as_tensor=True):
        # transform list to tensor
        if as_tensor:
            return torch.stack(self.log_px)
        return self.log_px


    def fit(self,
            x_train: torch.Tensor,
            n_epochs: int = 500,
            lr: float = 0.05,
            batch_size: int = 1024,
            shuffle: bool = True,
            show_progress: bool = False,
            w_train: torch.Tensor = None,
            context_train: torch.Tensor = None,
            x_val: torch.Tensor = None,
            w_val: torch.Tensor = None,
            context_val: torch.Tensor = None,
            keep_best_weights: bool = True,
            early_stopping: bool = False,
            early_stopping_threshold: int = 50):
        """
        Fit the normalizing flow.

        Fitting the flow means finding the parameters of the bijection that maximize the probability of training data.
        Bijection parameters are iteratively updated for a specified number of epochs.
        If context data is provided, the normalizing flow learns the distribution of data conditional on context data.

        :param x_train: training data with shape (n_training_data, *event_shape).
        :param n_epochs: perform fitting for this many steps.
        :param lr: learning rate. In general, lower learning rates are recommended for high-parametric bijections.
        :param batch_size: in each epoch, split training data into batches of this size and perform a parameter update for each batch.
        :param shuffle: shuffle training data. This helps avoid incorrect fitting if nearby training samples are similar.
        :param show_progress: show a progress bar with the current batch loss.
        :param w_train: training data weights with shape (n_training_data,).
        :param context_train: training data context tensor with shape (n_training_data, *context_shape).
        :param x_val: validation data with shape (n_validation_data, *event_shape).
        :param w_val: validation data weights with shape (n_validation_data,).
        :param context_val: validation data context tensor with shape (n validation_data, *context_shape).
        :param keep_best_weights: if True and validation data is provided, keep the bijection weights with the highest probability of validation data.
        :param early_stopping: if True and validation data is provided, stop the training procedure early once validation loss stops improving for a specified number of consecutive epochs.
        :param early_stopping_threshold: if early_stopping is True, fitting stops after no improvement in validation loss for this many epochs.
        """
        self.bijection.train()

        # Compute the number of event dimensions
        n_event_dims = int(torch.prod(torch.as_tensor(self.bijection.event_shape)))

        # Set the default batch size
        if batch_size is None:
            batch_size = len(x_train)

        # Process training data 
        train_loader = create_data_loader(  
            x_train,
            w_train,
            context_train,
            "training",
            batch_size=batch_size,
            shuffle=shuffle,
            event_shape=self.bijection.event_shape
        )

        # Process validation data
        if x_val is not None:
            val_loader = create_data_loader(
                x_val,
                w_val,
                context_val,
                "validation",
                batch_size=batch_size,
                shuffle=shuffle,
                event_shape=self.bijection.event_shape
            )

            best_val_loss = torch.inf
            best_epoch = 0
            best_weights = deepcopy(self.state_dict())

        def compute_batch_loss(batch_, reduction: callable = torch.mean):
            batch_x, batch_weights = batch_[:2]
            batch_context = batch_[2] if len(batch_) == 3 else None

            if self.debug:
                print("batch_x shape: ", batch_x.shape)
                print("batch_weights shape: ", batch_weights.shape)
                print("batch_context: ", batch_context)

                print("batch_x: ", batch_x)
                print("batch_weights: ", batch_weights)


            if self.objective == "brute_force":
                # generate partition P (for brute force implementation)
                # z_dim = train_batch[0][0].shape[-1]
                # num_partitions = 5
                # generate_partition = lambda x, num_partitions: [p for p in torch.chunk(torch.arange(x), num_partitions)]
                # P = generate_partition(z_dim, num_partitions)
                # print("P: ", P)
                if self.record_Ihat_P or self.record_log_px: 
                    # batch_objective, Ihat_p, log_px = self.PF_objective_brute_force_old(batch_x, P, context = batch_context)
                    batch_objective, Ihat_p, log_px = self.PF_objective_brute_force(batch_x, reduction=reduction, random_seed=None, context=batch_context)
                    return batch_objective, Ihat_p, log_px
                else:
                    # batch_objective = self.PF_objective_brute_force_old(batch_x, P, context = batch_context)
                    batch_objective = self.PF_objective_brute_force(batch_x, context = batch_context, reduction = reduction)
                    return batch_objective
                
            elif self.objective == "unbiased":
                if self.record_Ihat_P or self.record_log_px:
                    batch_objective, Ihat_p, log_px = self.PF_objective_unbiased(batch_x, reduction=torch.mean, random_seed=None, context=batch_context)
                    return batch_objective, Ihat_p, log_px
                else:
                    batch_objective = self.PF_objective_unbiased(batch_x, context = batch_context, reduction = reduction)

            else:
                raise ValueError("Invalid objective method")

            return batch_objective
        

        iterator = tqdm(range(n_epochs), desc='Fitting Principal Manifold Flow', disable=not show_progress)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        val_loss = None

        for epoch in iterator:
            # create empty tensors to store Ihat_P and log_px
            Ihat_P_epoch = torch.empty(0)
            log_px_epoch = torch.empty(0)
            for train_batch in train_loader:
                optimizer.zero_grad()
                if self.record_Ihat_P or self.record_log_px:
                    train_loss, Ihat_p, log_px = compute_batch_loss(train_batch, reduction=torch.mean)
                    Ihat_P_epoch = torch.cat((Ihat_P_epoch, Ihat_p))
                    log_px_epoch = torch.cat((log_px_epoch, log_px))
                    # update tqdm description
                    iterator.set_description(f'Fitting Principal Manifold Flow ({Ihat_p.mean():.6f}, {log_px.mean():.6f})')
                else:
                    train_loss = compute_batch_loss(train_batch, reduction=torch.mean)
                train_loss.backward()
                optimizer.step()

                if show_progress:
                    if val_loss is None:
                        iterator.set_postfix_str(f'Training loss (batch): {train_loss:.4f}')
                    else:
                        iterator.set_postfix_str(
                            f'Training loss (batch): {train_loss:.4f}, '
                            f'Validation loss: {val_loss:.4f}'
                        )

            if self.record_Ihat_P:
                with torch.no_grad():
                    self.Ihat_P.append(Ihat_P_epoch.mean())
            if self.record_log_px:
                with torch.no_grad():
                    self.log_px.append(log_px_epoch.mean())

            # Compute validation loss at the end of each epoch
            # Validation loss will be displayed at the start of the next epoch
            if x_val is not None:
                with torch.no_grad():
                    # Compute validation loss
                    val_loss = 0.0
                    for val_batch in val_loader:
                        n_batch_data = len(val_batch[0])
                        val_loss += compute_batch_loss(val_batch, reduction=torch.sum) / n_batch_data
                    if hasattr(self.bijection, 'regularization'):
                        val_loss += self.bijection.regularization()

                    # Check if validation loss is the lowest so far
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch

                    # Store current weights
                    if keep_best_weights:
                        if best_epoch == epoch:
                            best_weights = deepcopy(self.state_dict())

                    # Optionally stop training early
                    if early_stopping:
                        if epoch - best_epoch > early_stopping_threshold:
                            break

        if x_val is not None and keep_best_weights:
            self.load_state_dict(best_weights)

        self.bijection.eval()


