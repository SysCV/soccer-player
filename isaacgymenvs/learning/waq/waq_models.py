import torch
import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

class ModelContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return
    
    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            is_latent = input_dict.get('is_latent', False)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            if is_latent:
                v_est, latent_mean, latent_logSigma, predict = self.a2c_network.get_latent(input_dict)
                result = {
                    "v_est": v_est,
                    "latent_mu": latent_mean,
                    "latent_sigma": latent_logSigma,
                    "predict": predict
                    } 
                return result
            else:
                mu, logstd, value, states, v_est = self.a2c_network(input_dict)
                sigma = torch.exp(logstd)
                distr = torch.distributions.Normal(mu, sigma, validate_args=False)
                if is_train:
                    entropy = distr.entropy().sum(dim=-1)
                    prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                    result = {
                        'prev_neglogp' : torch.squeeze(prev_neglogp),
                        'values' : value,
                        'entropy' : entropy,
                        'rnn_states' : states,
                        'mus' : mu,
                        'sigmas' : sigma
                    }                
                    return result
                else:
                    selected_action = distr.sample()
                    neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                    result = {
                        'neglogpacs' : torch.squeeze(neglogp),
                        'values' : self.unnorm_value(value),
                        'actions' : selected_action,
                        'rnn_states' : states,
                        'mus' : mu,
                        'sigmas' : sigma,
                        'v_est' : v_est
                    }
                    return result
