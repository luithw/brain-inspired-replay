#!/usr/bin/env python3
import torch
from torch import optim

from data.load import get_multitask_experiment
import define_models as define
from eval import evaluate
from eval import callbacks as cb
from main_cl import handle_inputs
import tqdm
import copy
import utils

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.utils import loss_functions as lf, modules
from models.conv.nets import ConvLayers, DeconvLayers
from models.fc.nets import MLP, MLP_gates
from models.fc.layers import fc_layer, fc_layer_split, fc_layer_fixed_gates
from models.cl.continual_learner import ContinualLearner
from utils import get_data_loader


class AutoEncoder(ContinualLearner):
    """Class for variational auto-encoder (VAE) models."""

    def __init__(self, image_size, image_channels, classes,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, convE=None, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, h_dim=400, fc_drop=0, fc_bn=False, fc_nl="relu", excit_buffer=False,
                 fc_gated=False,
                 # -prior
                 prior="standard", z_dim=20, per_class=False, n_modes=1,
                 # -decoder
                 recon_loss='BCE', network_output="sigmoid", deconv_type="standard", hidden=False,
                 dg_gates=False, dg_type="task", dg_prop=0., tasks=5, scenario="task", device='cuda',
                 # -classifer
                 classifier=True,
                 # -training-specific settings (can be changed after setting up model)
                 lamda_pl=0., lamda_rcl=1., lamda_vl=1., **kwargs):

        # Set configurations for setting up the model
        super().__init__()
        self.label = "VAE"
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.fc_layers = fc_layers
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.fc_units = fc_units
        self.fc_drop = fc_drop
        self.depth = depth if convE is None else convE.depth
        # -replay hidden representations? (-> replay only propagates through fc-layers)
        self.hidden = hidden
        # -type of loss to be used for reconstruction
        self.recon_loss = recon_loss  # options: BCE|MSE
        self.network_output = network_output
        # -settings for class- or task-specific gates in fully-connected hidden layers of decoder
        self.dg_type = dg_type
        self.dg_prop = dg_prop
        self.dg_gates = dg_gates if dg_prop > 0. else False
        self.gate_size = (tasks if dg_type == "task" else classes) if self.dg_gates else 0
        self.scenario = scenario

        # Optimizer (needs to be set before training starts))
        self.optimizer = None
        self.optim_list = []

        # Prior-related parameters
        self.prior = prior
        self.per_class = per_class
        self.n_modes = n_modes * classes if self.per_class else n_modes
        self.modes_per_class = n_modes if self.per_class else None

        # Components deciding how to train / run the model (i.e., these can be changed after setting up the model)
        # -options for prediction loss
        self.lamda_pl = lamda_pl  # weight of classification-loss
        # -how to compute the loss function?
        self.lamda_rcl = lamda_rcl  # weight of reconstruction-loss
        self.lamda_vl = lamda_vl  # weight of variational loss

        # Check whether there is at least 1 fc-layer
        if fc_layers < 1:
            raise ValueError("VAE cannot have 0 fully-connected layers!")

        ######------SPECIFY MODEL------######

        ##>----Encoder (= q[z|x])----<##
        self.convE = ConvLayers(conv_type=conv_type, block_type="basic", num_blocks=num_blocks,
                                image_channels=image_channels, depth=self.depth, start_channels=start_channels,
                                reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
                                output="none" if no_fnl else "normal", global_pooling=global_pooling,
                                gated=conv_gated) if (convE is None) else convE
        self.flatten = modules.Flatten()
        # ------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        if fc_layers < 2:
            self.fc_layer_sizes = [self.conv_out_units]  # --> this results in self.fcE = modules.Identity()
        elif fc_layers == 2:
            self.fc_layer_sizes = [self.conv_out_units, h_dim]
        else:
            self.fc_layer_sizes = [self.conv_out_units] + [int(x) for x in
                                                           np.linspace(fc_units, h_dim, num=fc_layers - 1)]
        real_h_dim = h_dim if fc_layers > 1 else self.conv_out_units
        # ------------------------------------------------------------------------------------------#
        self.fcE = MLP(size_per_layer=self.fc_layer_sizes, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
                       excit_buffer=excit_buffer, gated=fc_gated)
        # to z
        self.toZ = fc_layer_split(real_h_dim, z_dim, nl_mean='none', nl_logvar='none')  # , drop=fc_drop)

        ##>----Classifier----<##
        if classifier:
            self.units_before_classifier = real_h_dim
            self.classifier = fc_layer(self.units_before_classifier, classes, excit_buffer=True, nl='none')

        ##>----Decoder (= p[x|z])----<##
        out_nl = True if fc_layers > 1 else (True if (self.depth > 0 and not no_fnl) else False)
        real_h_dim_down = h_dim if fc_layers > 1 else self.convE.out_units(image_size, ignore_gp=True)
        self.fromZ = fc_layer_fixed_gates(
            z_dim, real_h_dim_down, batch_norm=(out_nl and fc_bn), nl=fc_nl if out_nl else "none",
            gate_size=self.gate_size, gating_prop=dg_prop, device=device
        )
        fc_layer_sizes_down = self.fc_layer_sizes
        fc_layer_sizes_down[0] = self.convE.out_units(image_size, ignore_gp=True)
        # -> if 'gp' is used in forward pass, size of first/final hidden layer differs between forward and backward pass
        self.fcD = MLP_gates(
            size_per_layer=[x for x in reversed(fc_layer_sizes_down)], drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
            gate_size=self.gate_size, gating_prop=dg_prop, device=device,
            output=self.network_output if (self.depth == 0 or self.hidden) else 'normal',
        )
        # to image-shape
        self.to_image = modules.Reshape(image_channels=self.convE.out_channels if self.depth > 0 else image_channels)
        # through deconv-layers
        self.convD = DeconvLayers(
            image_channels=image_channels, final_channels=start_channels, depth=self.depth,
            reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, gated=conv_gated,
            output=self.network_output, deconv_type=deconv_type,
        ) if (not self.hidden) else modules.Identity()

        ##>----Prior----<##
        # -if using the GMM-prior, add its parameters
        if self.prior == "GMM":
            # -create
            self.z_class_means = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            self.z_class_logvars = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            # -initialize
            self.z_class_means.data.normal_()
            self.z_class_logvars.data.normal_()

    ##------ FORWARD FUNCTIONS --------##

    def forward(self, x, gate_input=None, **kwargs):
        '''Forward function to propagate [x] through the encoder, reparametrization and decoder.

        Input: - [x]          <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]
                              (or <4D-tensor> of shape [batch_size]x[out_channels]x[out_size]x[outsize], if self.hidden)
               - [gate_input] <1D-tensor> or <np.ndarray>; for each batch-element in [x] its class-ID (eg, [y]) ---OR---
                              <2D-tensor>; for each batch-element in [x] a probability for each class-ID (eg, [y_hat])'''
        # -encode (forward), reparameterize and decode (backward)
        mu, logvar, hE, hidden_x = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, gate_input=gate_input)
        # -classify
        y_hat = self.classifier(hE)
        # -return
        return (x_recon, y_hat, mu, logvar, z)

    def encode(self, x):
        '''Pass input through feed-forward connections, to get [z_mean], [z_logvar] and [hE].
        Input [x] is either an image or, if [self.hidden], extracted "intermediate" or "internal" image features.'''
        # Forward-pass through conv-layers
        hidden_x = self.convE(x)
        image_features = self.flatten(hidden_x)
        # Forward-pass through fc-layers
        hE = self.fcE(image_features)
        # Get parameters for reparametrization
        (z_mean, z_logvar) = self.toZ(hE)
        return z_mean, z_logvar, hE, hidden_x

    def decode(self, z, gate_input=None):
        '''Decode latent variable activations.

        INPUT:  - [z]            <2D-tensor>; latent variables to be decoded
                - [gate_input]   <1D-tensor> or <np.ndarray>; for each batch-element in [x] its class-/taskID  ---OR---
                                 <2D-tensor>; for each batch-element in [x] a probability for every class-/task-ID

        OUTPUT: - [image_recon]  <4D-tensor>'''

        # -if needed, convert [gate_input] to one-hot vector
        if type(gate_input) == np.ndarray or gate_input.dim() < 2:
            gate_input = lf.to_one_hot(gate_input, classes=self.gate_size, device=self._device())

        # -put inputs through decoder
        hD = self.fromZ(z, gate_input=gate_input)
        image_features = self.fcD(hD, gate_input=gate_input)
        image_recon = self.convD(self.to_image(image_features))
        return image_recon

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def classify(self, x, not_hidden=False, **kwargs):
        '''For input [x] (image or extracted "internal" image features), return all predicted "scores"/"logits".'''
        image_features = self.flatten(x) if (self.hidden and not not_hidden) else self.flatten(self.convE(x))
        hE = self.fcE(image_features)
        return self.classifier(hE)


    ##------ SAMPLE FUNCTIONS --------##

    def sample(self, size, allowed_classes=None, class_probs=None, sample_mode=None, allowed_domains=None,
               only_x=False, **kwargs):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

        INPUT:  - [allowed_classes]     <list> of [class_ids] from which to sample
                - [class_probs]         <list> with for each class the probability it is sampled from it
                - [sample_mode]         <int> to sample from specific mode of [z]-distr'n, overwrites [allowed_classes]
                - [allowed_domains]     <list> of [task_ids] which are allowed to be used for 'task-gates' (if used)
                                          NOTE: currently only relevant if [scenario]=="domain"

        OUTPUT: - [X]         <4D-tensor> generated images / image-features
                - [y_used]    <ndarray> labels of classes intended to be sampled  (using <class_ids>)
                - [task_used] <ndarray> labels of domains/tasks used for task-gates in decoder'''

        # set model to eval()-mode
        self.eval()

        # pick for each sample the prior-mode to be used
        # -sample from modes belonging to [allowed_classes], possibly weighted according to [class_probs]
        allowed_modes = []  # -collect all allowed modes
        unweighted_probs = []  # -collect unweighted sample-probabilities of those modes
        for index, class_id in enumerate(allowed_classes):
            allowed_modes += list(
                range(class_id * self.modes_per_class, (class_id + 1) * self.modes_per_class))
        sampled_modes = np.random.choice(allowed_modes, size, replace=True)
        y_used = np.array([int(mode / self.modes_per_class) for mode in sampled_modes])

        # sample z
        prior_means = self.z_class_means
        prior_logvars = self.z_class_logvars
        # -for each sample to be generated, select the previously sampled mode
        z_means = prior_means[sampled_modes, :]
        z_logvars = prior_logvars[sampled_modes, :]
        with torch.no_grad():
            z = self.reparameterize(z_means, z_logvars)

        # if the gates in the decoder are "task-gates", convert [y_used] to corresponding tasks (if Task-IL or Class-IL)
        #   or simply sample which tasks should be generated (if Domain-IL) from [allowed_domains]
        task_used = None

        # decode z into image X
        with torch.no_grad():
            X = self.decode(z, gate_input=y_used)

        # return samples as [batch_size]x[channels]x[image_size]x[image_size] tensor, plus requested additional info
        return X, y_used, task_used

    ##------ LOSS FUNCTIONS --------##

    def calculate_recon_loss(self, x, x_recon):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]     (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

        OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]'''
        batch_size = x.size(0)
        reconL = F.binary_cross_entropy(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1),
                                        reduction='mean')
        return reconL

    def calculate_log_p_z(self, z, y=None, y_prob=None, allowed_classes=None):
        '''Calculate log-likelihood of sampled [z] under the prior distirbution.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")

        OPTIONS THAT ARE RELEVANT ONLY IF self.per_class IS TRUE:
            - [y]               None or <1D-tensor> with target-classes (as integers)
            - [y_prob]          None or <2D-tensor> with probabilities for each class (in [allowed_classes])
            - [allowed_classes] None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [log_p_z]   <1D-tensor> of length [batch_size]'''

        ## Get [means] and [logvars] of all (possible) modes
        allowed_modes = list(range(self.n_modes))
        # -if we don't use the specific modes of a target, we could select modes based on list of classes
        if (y is None) and (allowed_classes is not None) and self.per_class:
            allowed_modes = []
            for class_id in allowed_classes:
                allowed_modes += list(range(class_id * self.modes_per_class, (class_id + 1) * self.modes_per_class))
        # -calculate/retireve the means and logvars for the selected modes
        prior_means = self.z_class_means[allowed_modes, :]
        prior_logvars = self.z_class_logvars[allowed_modes, :]
        # -rearrange / select for each batch prior-modes to be used
        z_expand = z.unsqueeze(1)  # [batch_size] x 1 x [z_dim]
        means = prior_means.unsqueeze(0)  # 1 x [n_modes] x [z_dim]
        logvars = prior_logvars.unsqueeze(0)  # 1 x [n_modes] x [z_dim]

        ## Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on selected priors)
        n_modes = self.modes_per_class
        a = lf.log_Normal_diag(z_expand, mean=means, log_var=logvars, average=False, dim=2) - math.log(n_modes)
        # --> for each element in batch, calculate log-likelihood for all pseudoinputs: [batch_size] x [n_modes]
        if (y is not None) and self.per_class:
            modes_list = list()
            for i in range(len(y)):
                target = y[i].item()
                modes_list.append(list(range(target * self.modes_per_class, (target + 1) * self.modes_per_class)))
            modes_tensor = torch.LongTensor(modes_list).to(self._device())
            a = a.gather(dim=1, index=modes_tensor)
            # --> reduce [a] to size [batch_size]x[modes_per_class] (ie, per batch only keep modes of [y])
            #     but within the batch, elements can have different [y], so this reduction couldn't be done before

        a_max, _ = torch.max(a, dim=1)  # [batch_size]
        # --> for each element in batch, take highest log-likelihood over all pseudoinputs
        #     this is calculated and used to avoid underflow in the below computation
        a_exp = torch.exp(a - a_max.unsqueeze(1))  # [batch_size] x [n_modes]
        if (y is None) and (y_prob is not None) and self.per_class:
            batch_size = y_prob.size(0)
            y_prob = y_prob.view(-1, 1).repeat(1, self.modes_per_class).view(batch_size, -1)
            # ----> extend probabilities per class to probabilities per mode; y_prob: [batch_size] x [n_modes]
            a_logsum = torch.log(torch.clamp(torch.sum(y_prob * a_exp, dim=1), min=1e-40))
        else:
            a_logsum = torch.log(torch.clamp(torch.sum(a_exp, dim=1), min=1e-40))  # -> sum over modes: [batch_size]
        log_p_z = a_logsum + a_max  # [batch_size]

        return log_p_z

    def calculate_variat_loss(self, z, mu, logvar, y=None, y_prob=None, allowed_classes=None):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")
                - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]

        OPTIONS THAT ARE RELEVANT ONLY IF self.per_class IS TRUE:
            - [y]               None or <1D-tensor> with target-classes (as integers)
            - [y_prob]          None or <2D-tensor> with probabilities for each class (in [allowed_classes])
            - [allowed_classes] None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [variatL]   <1D-tensor> of length [batch_size]'''

        # --> calculate "by estimation"

        ## Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on selected priors)
        log_p_z = self.calculate_log_p_z(z, y=y, y_prob=y_prob, allowed_classes=allowed_classes)
        # ----->  log_p_z: [batch_size]

        ## Calculate "log_q_z_x" (entropy of "reparameterized" [z] given [x])
        log_q_z_x = lf.log_Normal_diag(z, mean=mu, log_var=logvar, average=False, dim=1)
        # ----->  mu: [batch_size] x [z_dim]; logvar: [batch_size] x [z_dim]; z: [batch_size] x [z_dim]
        # ----->  log_q_z_x: [batch_size]

        ## Combine
        variatL = -(log_p_z - log_q_z_x)

        return variatL

    def loss_function(self, x, y, x_recon, y_hat, scores, mu, z, logvar=None, allowed_classes=None):
        '''Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:  - [x]           <4D-tensor> original image
                - [y]           <1D-tensor> with target-classes (as integers, corresponding to [allowed_classes])
                - [x_recon]     (tuple of 2x) <4D-tensor> reconstructed image in same shape as [x]
                - [y_hat]       <2D-tensor> with predicted "logits" for each class (corresponding to [allowed_classes])
                - [scores]         <2D-tensor> with target "logits" for each class (corresponding to [allowed_classes])
                                     (if len(scores)<len(y_hat), 0 probs are added during distillation step at the end)
                - [mu]             <2D-tensor> with either [z] or the estimated mean of [z]
                - [z]              <2D-tensor> with reparameterized [z]
                - [logvar]         None or <2D-tensor> with estimated log(SD^2) of [z]
                - [allowed_classes]None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                - [variatL]      variational (KL-divergence) loss "indicating how close distribion [z] is to prior"
                - [predL]        prediction loss indicating how well targets [y] are predicted
                - [distilL]      knowledge distillation (KD) loss indicating how well the predicted "logits" ([y_hat])
                                     match the target "logits" ([scores])'''

        ###-----Reconstruction loss-----###
        reconL = self.calculate_recon_loss(x=x, x_recon=x_recon).mean(0)  # -> average over pixels

        ###-----Variational loss-----###
        actual_y = torch.tensor([allowed_classes[i.item()] for i in y]).to(self._device()) if (
                (allowed_classes is not None) and (y is not None)
        ) else y
        if (y is None and scores is not None):
            y_prob = F.softmax(scores / self.KD_temp, dim=1)
            if allowed_classes is not None and len(allowed_classes) > y_prob.size(1):
                zeros_to_add = torch.zeros(y_prob.size(0), len(allowed_classes) - y_prob.size(1)).to(self._device())
                y_prob = torch.cat([y_prob, zeros_to_add], dim=1)
        else:
            y_prob = None
        # ---> if [y] is not provided but [scores] is, calculate variational loss using weighted sum of prior-modes
        variatL = self.calculate_variat_loss(z=z, mu=mu, logvar=logvar, y=actual_y, y_prob=y_prob,
                                             allowed_classes=allowed_classes).mean(0)
        variatL /= (self.image_channels * self.image_size ** 2)  # -> divide by # of input-pixels

        ###-----Prediction loss-----###
        if y is not None and y_hat is not None:
            predL = F.cross_entropy(input=y_hat, target=y, reduction='mean')
        else:
            predL = torch.tensor(0., device=self._device())

        ###-----Distilliation loss-----###
        if scores is not None and y_hat is not None:
            # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes would be added to [scores]!
            n_classes_to_consider = y_hat.size(1)  # --> zeros will be added to [scores] to make it this size!
            distilL = lf.loss_fn_kd(scores=y_hat[:, :n_classes_to_consider], target_scores=scores, T=self.KD_temp)  # --> summing over classes & averaging over batch in function
        else:
            distilL = torch.tensor(0., device=self._device())

        # Return a tuple of the calculated losses
        return reconL, variatL, predL, distilL

    ##------ TRAINING FUNCTIONS --------##
    def train_a_batch(self, x, y=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        [x]                 <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]                 None or <tensor> batch of corresponding labels
        [x_]                None or (<list> of) <tensor> batch of replayed inputs
                              NOTE: expected to be at hidden level if [self.hidden], unless [replay_not_hidden]==True
        [y_]                None or (<list> of) <1Dtensor>:[batch] of corresponding "replayed" labels
        [scores_]           None or (<list> of) <2Dtensor>:[batch]x[classes] target "scores"/"logits" for [x_]
        [rnt]               <number> in [0,1], relative importance of new task
        [active_classes]    None or (<list> of) <list> with "active" classes'''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        # Run the model
        x = self.convE(x) if self.hidden else x  # -pre-processing (if 'hidden')
        recon_batch, y_hat, mu, logvar, z = self(x, gate_input=y)
        # -if needed ("class"/"task"-scenario), find allowed classes for current task & remove predictions of others
        y_hat = y_hat[:, active_classes]

        # Calculate all losses
        reconL, variatL, predL, _ = self.loss_function(
            x=x, y=y, x_recon=recon_batch, y_hat=y_hat, scores=None, mu=mu, z=z, logvar=logvar,
            allowed_classes=active_classes
        )  # --> [allowed_classes] will be used only if [y] is not provided

        # Weigh losses as requested
        loss_cur = self.lamda_rcl * reconL + self.lamda_vl * variatL + self.lamda_pl * predL

        # Calculate training-precision
        _, predicted = y_hat.max(1)
        precision = (y == predicted).sum().item() / x.size(0)

        ##--(2)-- REPLAYED DATA --##
        if x_ is not None:

            # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
            # -if needed in the decoder-gates, find class-tensor [y_predicted]
            y_predicted = F.softmax(scores_ / self.KD_temp, dim=1)
            # in case of Class-IL, add zeros at the end:
            n_batch = y_predicted.size(0)
            zeros_to_add = torch.zeros(n_batch, self.classes - y_predicted.size(1))
            zeros_to_add = zeros_to_add.to(self._device())
            y_predicted = torch.cat([y_predicted, zeros_to_add], dim=1)
            # -run full model
            recon_batch, y_hat_all, mu, logvar, z = self(x_, gate_input=y_predicted)


            # perform replay
            # -if needed (e.g., "class" or "task" scenario), remove predictions for classes not in replayed task
            y_hat = y_hat_all[:, active_classes]

            # Calculate all losses
            reconL_r, variatL_r, predL_r, distilL_r = self.loss_function(
                x=x_, y=y_, x_recon=recon_batch, y_hat=y_hat,
                scores=scores_, mu=mu, z=z, logvar=logvar,
                allowed_classes=active_classes,
            )
            # Weigh losses as requested
            loss_replay = self.lamda_rcl * reconL_r + self.lamda_vl * variatL_r
            loss_replay += self.lamda_pl * distilL_r

        # Calculate total loss
        loss_replay = None if (x_ is None) else loss_replay
        loss_total = loss_cur if x_ is None else rnt * loss_cur + (1 - rnt) * loss_replay

        ##--(3)-- ALLOCATION LOSSES --##
        loss_total.backward()
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(), 'precision': precision,
            'recon': reconL.item() if x is not None else 0,
            'variat': variatL.item() if x is not None else 0,
            'pred': predL.item() if x is not None else 0,
            'recon_r': reconL_r if x_ is not None else 0,
            'variat_r': variatL_r if x_ is not None else 0,
            'pred_r': predL_r if x_ is not None else 0,
            'distil_r': distilL_r if x_ is not None else 0,
        }


def train_cl(model, train_datasets, replay_mode="none", classes_per_task=None, iters=2000, batch_size=32, loss_cbs=list()):

    device = model._device()
    cuda = model._is_on_cuda()
    batch_size_replay = batch_size

    # Initiate indicators for replay (no replay for 1st task)
    Generative = False
    previous_model = None

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):

        # Initialize # iters left on data-loader(s)
        iters_left = 1

        # -for "class"-scenario, create one <list> with active classes of all tasks so far
        active_classes = list(range(classes_per_task*task))

        # Define a tqdm progress bar(s)
        iters_main = iters
        progress = tqdm.tqdm(range(1, iters_main+1))
        # Loop over all iterations
        iters_to_use = iters_main
        # -if only the final task should be trained on:
        for batch_index in range(1, iters_to_use+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=True))
                iters_left = len(data_loader)

            #-----------------Collect data------------------#

            #####-----CURRENT BATCH-----#####
            x, y = next(data_loader)                                    #--> sample training data of current task
            x, y = x.to(device), y.to(device)                           #--> transfer them to correct device

            #####-----REPLAYED BATCH-----#####
            x_ = y_ = scores_ = task_used = None   #-> if no replay

            #--------------------------------------------INPUTS----------------------------------------------------#
            ##-->> Generative Replay <<--##
            if Generative:
                #---> Only with generative replay, the resulting [x_] will be at the "hidden"-level
                conditional_gen = True

                # Sample [x_]
                # -which classes are allowed to be generated? (relevant if conditional generator / decoder-gates)
                allowed_classes = list(range(classes_per_task*(task-1)))
                # -which tasks/domains are allowed to be generated? (only relevant if "Domain-IL" with task-gates)
                allowed_domains = list(range(task-1))
                # -generate inputs representative of previous tasks
                x_temp_ = previous_generator.sample(
                    batch_size_replay, allowed_classes=allowed_classes, allowed_domains=allowed_domains,
                    only_x=False,
                )
                x_ = x_temp_[0]
                task_used = x_temp_[2]

            #--------------------------------------------OUTPUTS----------------------------------------------------#
            if Generative:
                # Get target scores & possibly labels (i.e., [scores_] / [y_]) -- use previous model, with no_grad()
                # -if replay does not need to be evaluated for each task (ie, not Task-IL and no task-specific mask)
                with torch.no_grad():
                    all_scores_ = previous_model.classify(x_, not_hidden=False)
                # when scenario=="class", zero probs will be added in [loss_fn_kd]-function
                scores_ = all_scores_[:, :(classes_per_task*(task-1))]
                # -also get the 'hard target'
                _, y_ = torch.max(scores_, dim=1)

            # -only keep predicted y_/scores_ if required (as otherwise unnecessary computations will be done)
            y_ = None

            #-----------------Train model(s)------------------#

            #---> Train MAIN MODEL
            if batch_index <= iters_main:
                # Train the main model with this batch
                loss_dict = model.train_a_batch(x, y=y, x_=x_, y_=y_, scores_=scores_, active_classes=active_classes,
                                                rnt=(1. if task==1 else 1./task))

                # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)

        # Close progres-bar(s)
        progress.close()

        ##----------> UPON FINISHING EACH TASK...

        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model).eval()
        if replay_mode == "generative":
            Generative = True
            previous_generator = previous_model


def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
             no_task_mask=False, task=None):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Get device-type / using cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Set model to eval()-mode
    model.eval()

    # Apply task-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        data, labels = data.to(device), labels.to(device)
        labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
        with torch.no_grad():
            scores = model.classify(data, not_hidden=True)
            scores = scores if (allowed_classes is None) else scores[:, allowed_classes]
            _, predicted = torch.max(scores, 1)
        # -update statistics
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)
    precision = total_correct / total_tested

    # Print result on screen (if requested) and return it
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


if __name__ == "__main__":
    args = handle_inputs()
    device = torch.device("cuda")

    # Prepare data for chosen experiment
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name='splitMNIST', scenario='class', tasks=args.tasks,
        normalize=False,
        augment=False,
        verbose=True, exception=False, only_test=False
    )

    model = AutoEncoder(
        image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
        # -fc-layers
        fc_layers=3, fc_units=400, h_dim=400, fc_drop=0, fc_bn=False, fc_nl="relu", excit_buffer=True,
        # -prior
        prior="GMM", n_modes=1, per_class=True, z_dim=100,
        # -decoder
        recon_loss="BCE", network_output="sigmoid", deconv_type="standard", dg_gates=True, dg_type="class",
        dg_prop=0.8, tasks=5, scenario="class", device=device,
        # -classifier
        classifier=True,
        # -training-specific components
        lamda_rcl=1., lamda_vl=1., lamda_pl=1.,
    ).to(device)

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
        model, train_datasets, replay_mode=args.replay,
        classes_per_task=classes_per_task, iters=args.iters,
        batch_size=args.batch, loss_cbs=generator_loss_cbs
    )

    # Evaluate precision of final model on full test-set
    precs = [validate(
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
