import os

import experiments_utils as exputils

from .custom_slurm_generator import CustomSlurmGenerator
from .path_manager import PathManager

from .rl_train import set_wandb_params, train_func, sweep_trainer, preprocess_quick_test



if __name__=='__main__':
    
    verbose  = 3
    num_envs = 2
    save_model = False

    quick_test = False

    use_wandb  = True
    use_sweep  = True
    is_offline = False
    
    exp_cfg = dict(
        exp_id       = 'rl_compare_pretrain',
        exp_name     = 'rl_compare_pretrain_jz_full',
        seed         = [0,1,2,3,4,5,6,7,8,9],
        device       = 'default',
        verbose      = verbose,
        version      = '',
        save_model   = save_model,
        hugging_face = False,
        num_envs     = num_envs
    )
    operation_cfg = dict(
        operand_min_size=['plus2'],
        operand_max_size='plus2',
        operators='sum',
        confort_zone_prop=[0.]
    )
    template_cfg = dict(
        template_mode='scratchpad0'
    )
    prompt_cfg = dict(
        min_num_examples = 0,
        max_num_examples = 0
    )
    reward_cfg = dict(
        reward_mode                = 'ternary',
        reward_mode_eval           = 'binary',
        decomposition_step_bonus   = 1,
        wrong_result_penalty       = [-1],
        decomposition_step_penalty = -10,
        math_symbol_bonus          = 0,
        correctness_bonus          = 0,
        time_penalty               = 0,
        padding_penalty            = -10,
        max_ep_length_penalty      = -100
    )
    env_cfg = dict(
        max_episode_length = 'standard',
        result_banner      = None,
        interrupt_wrong_decomposition_step = [False]
    )
    llm_cfg = dict(
        model_name         = 'gpt2_special',
        model_path         = ['lecraquito/gpt2_reduced_vocab_FT_9digits_20k', 'lecraquito/gpt2_reduced_vocab_FT_11digits_20k', 'lecraquito/gpt2_reduced_vocab_FT_13digits_20k'],
        pad_token          = 'default'
    )
    network_cfg = dict(
        num_trainable_layers    = None,
        temperature = [2.],
        delete_forbidden_logit  = False,
        hidden_size_critic1     = 128,
        hidden_size_critic2     = [128]
    )
    exp_cfg_trainer = dict(
        step_per_epoch           = 1,    # num of explore/update steps before test
        step_per_collect         = None, # [16384], # num of exploration steps before policy update
        episode_per_test_confort = 20,
        episode_per_test = 100,
        max_epoch = 3,
        episode_per_collect = 50
    )
    rl_cfg = dict(
        policy_name = 'A2C'
    )
    optimizer_cfg = dict(
        lr = [1e-6]
    )
    trainer_cfg = dict(
        repeat_per_collect = [1],
        batch_size         = 2
    )
    policy_cfg = dict(
        load_model          = None,
        discount_factor     = 1.0,
        deterministic_eval  = False,
        use_scheduler       = False
    )
    kl_config = dict(
          ent_coef = 0.001,
          kl_div_coef = 10.,
          kl_approx_method         = 'k2',   # k1, k2 or k3
          certainty_estimation     = [None], # None, max_prob, negentropy_max_prob, negentropy_sample, negentropy
          certainty_scaling_exponent = [0],  # None, or > 1
          certainty_scaling_intercept = 0,
          temperature_weighted_kl  = [0],
    )
    actor_critic_cfg = dict(
        vf_coef     = [0.1],
        detach_critic_preprocess_output = [False]
    )
    PPO_cfg = dict(
        eps_clip = [0.2]
    )
    replay_buffer_cfg = dict(
        multiply_factor =  [1.]
    )
    config = {**exp_cfg, **operation_cfg, **template_cfg, **prompt_cfg, **reward_cfg, **env_cfg, **exp_cfg_trainer, **llm_cfg, **network_cfg, **rl_cfg, **optimizer_cfg, **trainer_cfg,
              **policy_cfg, **kl_config, **actor_critic_cfg, **PPO_cfg, **replay_buffer_cfg}

    arguments = exputils.retrieve_arguments()
    mode, names_dict, cluster_name = exputils.set_experiment_mode(arguments=arguments)
    wandb_names, metric_goal = set_wandb_params(use_wandb=use_wandb, names_dict=names_dict, mode=mode)
    names_dict = {**names_dict, **wandb_names} if use_wandb or mode=='cluster' else None

    
    filename = os.path.basename(__file__).split('.py')[0]
    if mode=="generate_slurm":
        exputils.generate_slurm(config=config, filename=filename, cluster_name=cluster_name, SlurmGenerator_cls=CustomSlurmGenerator)
    elif mode=="cluster":
        exputils.run_in_cluster_mode(train_func=train_func, filename=filename, CONFIGS_PATH=PathManager.CONFIGS, names_dict=names_dict, SlurmGenerator_cls=CustomSlurmGenerator)
    elif mode=="standard":
        exputils.run_in_standard_mode(config=config, train_func=train_func, filename=filename,
                                      quick_test=quick_test, use_sweep=use_sweep, use_wandb=use_wandb, is_offline=False,
                                      names_dict=names_dict, metric_goal=metric_goal,
                                      sweep_trainer=sweep_trainer, preprocess_quick_test_func=preprocess_quick_test,
                                      SlurmGenerator_cls=CustomSlurmGenerator, wandb_method="grid")
    else:
        raise ValueError(f'Mode {mode} not supported.')
