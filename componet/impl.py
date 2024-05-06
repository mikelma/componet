import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import math


def get_position_encoding(seq_len, d, n=10_000):
    """
    Returns the positional encoding matrix for the given parameters.

    seq_len -- Length of the sequence (number of positional encodings).
    d       -- Dimension of the encodings.
    n       -- User defined scalar, set to 10k in "Attention is all you need". Default: 10_000.
    """
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


def logit2prob(logits):
    return F.softmax(logits, dim=1)


class Identity:
    """A callable object that returns it's input. Although this class might seem trivial,
    it fixes the issue of saving CompoNet and FirstModuleWrapper modules using `torch.save`,
    as it cannot serialize lambda functions.

    See: https://stackoverflow.com/questions/70608810/pytorch-cant-pickle-lambda
    """

    def __call__(self, arg, **kwargs):
        return arg


class CompoNet(nn.Module):
    def __init__(
        self,
        previous_units: [nn.Module],
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        internal_policy: nn.Module,
        ret_probs: bool,
        encoder: nn.Module = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        proj_bias: bool = True,
        att_heads_init: object = Identity(),
    ):
        """CompoNet initializer.

        Keyword arguments:

        previous_units    -- List of the previous CompoNet modules.
        input_dim         -- Dimension of the input state vector, or when using an encoder, the output dimension of the encoder.
        hidden_dim        -- Hidden dimension of the model.
        out_dim           -- Dimension of the output vector (should be equal to the number of actions).
        internal_policy   -- The model to be used as internal policy, it's input size must match with `input_dim+hidden_dim` and
                             it's output `out_dim`.
        ret_probs         -- If `True`, the module returns probability vectors, else logits.
        encoder           -- Optionally, the user can define a `nn.Module` to use as the encoder for the current module. The identity
                             function is used if no value is given.
        device            -- Device to operate on. Set to `cuda` if a gpu is found and no user defined value is given, else `cpu`.
        proj_bias         -- Whether to activate bias in the linear transformations of the attention heads. True by default.
        att_heads_init    -- Initialization of the linear transformations of the attention heads. PyTorch default initializer is used
                             by default.
        """
        super(CompoNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.ret_probs = ret_probs
        self.internal_policy = internal_policy
        self.encoder = encoder if encoder is not None else Identity()
        self.att_temp = np.sqrt(
            hidden_dim
        )  # pre-compute the temperature of the attention

        # this attribute is internally used by CompoNet to distinguish between the current
        # module and previous ones
        self.is_prev = False

        # linear transformations of the output attention head
        self.headout_wq = att_heads_init(
            nn.Linear(input_dim, hidden_dim, bias=proj_bias)
        )
        self.headout_wk = att_heads_init(nn.Linear(out_dim, hidden_dim, bias=proj_bias))

        # linear transformations of the input attention head
        self.headin_wq = att_heads_init(
            nn.Linear(input_dim, hidden_dim, bias=proj_bias)
        )
        self.headin_wk = att_heads_init(nn.Linear(out_dim, hidden_dim, bias=proj_bias))
        self.headin_wv = att_heads_init(nn.Linear(out_dim, hidden_dim, bias=proj_bias))

        # pre-compute the positional encodings for the input attention head
        n_prev = len(previous_units)
        pe1 = torch.tensor(
            get_position_encoding(seq_len=n_prev + 1, d=out_dim),
            dtype=torch.float32,
            device=device,
        )  # (n_prev+1, out_dim)
        self.pe1 = pe1[None, :, :]  # (1, n_prev+1, out_dim)

        # if needed, obtain the positional encoding for the output attention head
        if n_prev >= 2:
            self.pe0 = self.pe1[:, :-1, :]  # (1, n_prev, out_dim)
        else:
            self.pe0 = None

        # prepare previous units
        for unit in previous_units:
            # remove previous units if some
            if hasattr(unit, "previous_units"):
                del unit.previous_units
            unit.is_prev = True
            unit.eval()
            # freeze all the parameters
            for param in unit.parameters():
                param.requires_grad = False
        # join all previous units into a single sequential model
        self.previous_units = nn.Sequential(*previous_units)

    def _forward_headout(self, s, phi):
        """Compute the output attention head.
        Returns the result of the attention head and the employed attention weights.

        s    -- The current state representation.
        phi  -- The matrix with the results of the previous modules.
        """

        # compute the query, keys and values
        query = self.headout_wq(s)
        # add pos. enc. and compute K transformation
        keys = self.headout_wk(phi + self.pe0 if self.pe0 is not None else phi)
        values = phi

        # size: (batch, 1, num_policies)
        w = torch.matmul(
            # size: (batch, 1, hidden_dim)
            query[:, None, :],
            # size: (batch, hidden_dim, num_policies)
            keys.permute(0, 2, 1),  # transpose keys matrix
        )

        # get attention weights
        att = F.softmax(w / self.att_temp, dim=-1)

        # size: (batch_size, 1, out_dim)
        att_dot_val = torch.matmul(att, values)
        # att_dot_val = att_dot_val[:, 0, :]  # remove extra dim
        return att_dot_val, att

    def _get_internal_policy(self, s, phi):
        """Compute the input attention head and the internal policy.
        Returns the result of the internal policy and the employed attention weights.

        s    -- The current state representation.
        phi  -- The matrix with the results of the previous modules
                and the result of the output attention head.
        """

        # obtain the elements of the dot-product attentionx
        query = self.headin_wq(s)
        values = self.headin_wv(phi)
        keys = self.headin_wk(phi + self.pe1)

        # size: (batch, 1, num_policies)
        w = torch.matmul(
            # size: (batch, 1, hidden_dim)
            query[:, None, :],
            # size: (batch, hidden_dim, num_policies)
            keys.permute(0, 2, 1),  # transpose keys matrix
        )

        # get attention weights
        att = F.softmax(w / self.att_temp, dim=-1)

        # size: (batch_size, 1, out_dim)
        att_dot_val = torch.matmul(att, values)
        att_dot_val = att_dot_val[:, 0, :]  # remove extra dim

        # concat the current state and the result of the input attention head
        policy_in = torch.hstack([att_dot_val, s])
        # pass forward the internal policy
        policy_out = self.internal_policy(policy_in)

        return policy_out, att

    def forward(
        self,
        s,
        ret_encoder_out=False,
        return_atts=False,
        ret_int_pol=False,
        ret_head_out=False,
        prevs_to_noise=0,
    ):
        """Forward pass of the CompoNet unit.

        This method has two behaviors depending on whether the module is the last module of
        CompoNet or not.

        If it is not the last one, the method takes a matrix with the outputs of the preceding
        modules and the current state as input, and returns the same tuple but with the output
        of the module appended to the input matrix. In this case, all keyword arguments are
        ignored. This mode of operation is only intended to be used internally by CompoNet, and
        you might not need to care about it unless you are hacking with the code.

        If the module is the last one (the one operating in the current task), the method takes
        the current state as the input, and runs the whole CompoNet network to get the final result
        of the model. In this case, the method returns the vector with the output of the model and
        the matrix with the outputs of preceding modules, plus, some other values controlled by the
        method's keyword arguments.

        Keyword arguments:

        s                 -- Input state. Note that if an encoder was given, CompoNet will process
                             this tensor with the encoder before using it.

        ret_encoder_out   -- If true, the output of the encoder is appended to the return values.
                             Default: False.

        return_atts       -- If true, the att. values of the input and output att. heads are returned.
                             Default: False.

        ret_int_pol       -- If true, the output of the internal policy is returned. Default: False.

        ret_head_out      -- If true, the result of the output attention head is returned. Default: False

        prevs_to_noise    -- If >0, the output of the first `prevs_to_noise` number of modules is
                             replaced with random noise. This is intended to be used in ablation
                             experiments.

        Returns:
        pi  -- Output vector of the CompoNet model.
        phi -- The matrix of previous policies with the current one stacked as the last row.

        (optionals, in order)
        enc_out       -- The output of the encoder.
        att_head_in   -- Attention values of the input attention head.
        att_head_out  -- Attention values of the output attention head.
        ret_head_out  -- Output of the output attention head.
        int_pol       -- Output of the internal policy
        """
        # obtain the outputs of preceding modules (the phi matrix)
        if not self.is_prev:  # if it's the last module
            with torch.no_grad():
                # get the output of the previous modules in the Phi matrix
                phi, _s = self.previous_units(s)

                # this code block is only used for ablation purposes, where the output of
                # the first `prevs_to_noise` number of modules is replaced with random noise
                if prevs_to_noise > 0:
                    # sample from a uniform dirichlet if we're dealing with probability vectors
                    if self.ret_probs:
                        m = torch.distributions.Dirichlet(
                            torch.tensor([1 / self.out_dim] * self.out_dim)
                        )
                        r = m.sample(sample_shape=[phi.size(0), prevs_to_noise])
                    else:  # if we're using logits, sample from a 0,1 normal distribution
                        r = torch.randn((phi.size(0), prevs_to_noise, phi.size(-1)))
                    phi[:, :prevs_to_noise, :] = r
        else:
            assert type(s) == tuple, "Input to a previous unit must be a tuple (phi, s)"
            phi, s = s  # phi: (batch, num prev, out dim), s: (batch, input dim)

        hs = self.encoder(s)

        # compute the result of the output att. head
        out_head, att_head_out = self._forward_headout(hs, phi)

        # get the output of the internal policy
        int_pol_phi = torch.cat(
            [phi, out_head], dim=1
        )  # concat in the num_prev dimension
        logits, att_head_in = self._get_internal_policy(hs, int_pol_phi)

        # compute the final output of the module
        out_head = out_head[:, 0, :]  # (batch, 1, out_dim) -> (batch, out_dim)
        out = out_head + logits

        # normalize output if necessary
        if self.ret_probs:
            out = logit2prob(out)

        # add the resulting policy to the phi matrix
        out = out[:, None, :]  # out: (batch, 1, out_dim)
        phi = torch.cat([phi, out], dim=1)  # concat in the num_prev dimension

        if self.is_prev:
            return phi, s

        out = out[:, 0, :]  # (batch, out_dim)

        # build the return value depending on the selected options
        ret_vals = [out, phi]
        if ret_encoder_out:
            ret_vals.append(hs)
        if return_atts:
            ret_vals += [att_head_in, att_head_out]
        if ret_int_pol:
            ret_vals.append(logits)
        if ret_head_out:
            ret_vals.append(out_head)
        return ret_vals


class FirstModuleWrapper(nn.Module):
    """This class servers as a wrapper for the first module of CompoNet.

    The wrapper is around a PyTorch's `nn.Module` that takes as an input states of the
    environment and returns an output of the same shape as the rest of the CompoNet
    modules.

    For example, in atari, this model would take an image as input and it'd output a logit
    vector over the action space.

    **IMPORTANT:** The wrapped module MUST NOT return normalized outputs, the normalization is
    done by the wrapper when setting `ret_probs=True`.
    """

    def __init__(
        self,
        model: nn.Module,
        ret_probs: bool,
        encoder: object = Identity(),
        transform_output: object = Identity(),
    ):
        """FirstModuleWrapper initializer.

        Keyword arguments:
        model             -- The `nn.Module` to wrap.
        ret_probs         -- A boolean specifying whether to normalize the output of the model or not.
        encoder           -- An optional encoder model to process the inputs for the wrapped model.
                             The identity function is used by default.
        transform_output  -- A callable object that can be used to transform the output of the wrapped model.

        """
        super(FirstModuleWrapper, self).__init__()
        self.model = model
        self.encoder = encoder
        self.is_prev = False
        self.ret_probs = ret_probs
        self.transform_output = transform_output

    def forward(self, x, ret_encoder_out=False):
        h = self.encoder(x)
        v = self.transform_output(self.model(h))
        pi = logit2prob(v) if self.ret_probs else v

        phi = pi[None, :, :]  # (1, batch size, out dim)
        phi = phi.permute(1, 0, 2)  # (batch size, 1, out dim)
        if self.is_prev:
            return phi, x
        else:
            if ret_encoder_out:
                return pi, phi, h
            return pi, phi


if __name__ == "__main__":
    # an example of the usage of the CompoNet class

    input_dim = 128
    hidden_dim = 512
    out_dim = 6
    batch_size = 3

    num_units = 4

    def internal_policy_gen():
        return nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    first_module = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )
    dummy = FirstModuleWrapper(first_module, ret_probs=True)
    prevs = [dummy]

    for _ in range(num_units - 1):
        unit = CompoNet(
            previous_units=prevs,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            internal_policy=internal_policy_gen(),
            ret_probs=True,
        )
        prevs.append(unit)

    print(unit)

    x = torch.rand((batch_size, input_dim))
    print(
        f"\nInput dim: {input_dim}, Output dim: {out_dim}, Num units: {num_units}, Batch size: {batch_size}"
    )
    print(f"\nInput tensor:      {x.size()}")
    pi, phi = unit(x)
    print(f"Output policy:     {pi.size()}")
    print(f"Output Phi matrix: {phi.size()}")
