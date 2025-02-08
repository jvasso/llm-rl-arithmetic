import os
import sys
import time

from types import SimpleNamespace

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tianshou.data import Collector
from tianshou.data import VectorReplayBuffer, PrioritizedVectorReplayBuffer, HERVectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy, PPOPolicy, SACPolicy, A2CPolicy, PGPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import LazyLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

import wandb

import experiments_utils as exputils

from .custom_slurm_generator import CustomSlurmGenerator
from .custom_wandb_logger import CustomWandbLogger

from .rl_modules import CustomPGPolicy, CustomA2CPolicy, CustomPPOPolicy
from .rl_modules.custom_on_policy_trainer import CustomOnpolicyTrainer

from .env import ArithmeticEnv, ActionManager, PromptGenerator, RewardManager, TemplateManager, OperationGenerator
from .llm_manager import LLMmanager

from .policy import CustomActor, CustomCritic, PreprocessNet

from .path_manager import PathManager

from . import utils
from . import wandb_utils


DEFAULT_WANDB_GROUP_NAME = 'rl_train'

EVAL_CONFORT = 'eval_confort'


def create_env(config, llm_manager:LLMmanager, env_type:str='train', verbose:int=0):
    action_manager = ActionManager(llm_manager=llm_manager,
                                   delete_forbidden_logit=config.delete_forbidden_logit,
                                   human_mode=False)
    template_manager = TemplateManager(llm_manager=llm_manager,
                                       template_mode=config.template_mode)
    operation_generator = OperationGenerator(llm_manager=llm_manager,
                                             operand_min_size = config.operand_min_size,
                                             operand_max_size = config.operand_max_size,
                                             env_type=env_type,
                                             operators =config.operators,
                                             confort_zone_prop=config.confort_zone_prop)
    prompt_generator = PromptGenerator(operation_generator = operation_generator,
                                       template_manager    = template_manager,
                                       min_num_examples    = config.min_num_examples,
                                       max_num_examples    = config.max_num_examples)
    reward_mode = config.reward_mode_eval if 'eval' in env_type else config.reward_mode
    reward_manager = RewardManager(reward_mode              = reward_mode,
                                   decomposition_step_bonus   = config.decomposition_step_bonus,
                                   decomposition_step_penalty = config.decomposition_step_penalty,
                                   wrong_result_penalty       = config.wrong_result_penalty,
                                   math_symbol_bonus          = config.math_symbol_bonus,
                                   correctness_bonus          = config.correctness_bonus,
                                   time_penalty               = config.time_penalty,
                                   padding_penalty            = config.padding_penalty,
                                   max_ep_length_penalty      = config.max_ep_length_penalty,
                                   env_type                   = env_type)
    env = ArithmeticEnv(llm_manager, action_manager, prompt_generator, reward_manager, template_manager,
                        max_episode_length = config.max_episode_length,
                        result_banner = config.result_banner,
                        interrupt_wrong_decomposition_step = config.interrupt_wrong_decomposition_step,
                        env_type = env_type,
                        steps_of_interest = [1,2,3],
                        verbose = verbose)

    # other operations such as env.seed(np.random.choice(10))
    def create_env():
        return env
    return create_env


def load_llm_manager_and_models(config, device, logger:CustomWandbLogger, is_sft_model:bool):
    llm_manager = LLMmanager(model_name=config.model_name,
                             model_path=config.model_path,
                             pad_token=config.pad_token)
    temperature_weighted_kl = config.temperature_weighted_kl if is_sft_model else None
    preprocess_net = PreprocessNet(llm_manager=llm_manager,
                                   num_trainable_layers=config.num_trainable_layers,
                                   is_sft_model=is_sft_model,
                                   temperature=config.temperature, # config.temperature
                                   temperature_weighted_kl=temperature_weighted_kl,
                                   delete_forbidden_logit=config.delete_forbidden_logit,
                                   logger=logger,
                                   device=device).to(device)
    actor = CustomActor(preprocess_net=preprocess_net, device=device).to(device)
    dist_fn = torch.distributions.Categorical
    return llm_manager, preprocess_net, actor, dist_fn


################################################################################################################################################################################################

def train_func(config, device:str=None, use_wandb:bool=True, run=None):
    
    device = exputils.preprocess_training(config=config, seed=config.seed, device=config.device)
    
    num_envs = config.num_envs

    logger = CustomWandbLogger(train_interval_ep  = 1,
                               test_interval_ep   = 1,
                               update_interval_ep = 1,
                               use_wandb=use_wandb)
    
    llm_manager, preprocess_net, actor, dist_fn = load_llm_manager_and_models(config=config, device=device, logger=logger, is_sft_model=False)
                             
    ArithmeticEnv.RESET_CLASS_DATA()
    verbose = config.verbose

    train_envs = DummyVectorEnv([create_env(config, llm_manager=llm_manager, env_type='train', verbose=verbose) for _ in range(num_envs)])
    # eval_envs  = DummyVectorEnv([create_env(config, llm_manager=llm_manager, env_type='eval' , verbose=verbose) for _ in range(num_envs)])
    
    eval_envs_confort = DummyVectorEnv([create_env(config, llm_manager=llm_manager, env_type=EVAL_CONFORT, verbose=verbose) for _ in range(num_envs)])
    eval_envs_plus1   = DummyVectorEnv([create_env(config, llm_manager=llm_manager, env_type='eval_plus1', verbose=verbose) for _ in range(num_envs)])
    eval_envs_plus2   = DummyVectorEnv([create_env(config, llm_manager=llm_manager, env_type='eval_plus2', verbose=verbose) for _ in range(num_envs)])
    eval_envs_plus3   = DummyVectorEnv([create_env(config, llm_manager=llm_manager, env_type='eval_plus3', verbose=verbose) for _ in range(num_envs)])
    eval_envs_plus5   = DummyVectorEnv([create_env(config, llm_manager=llm_manager, env_type='eval_plus5', verbose=verbose) for _ in range(num_envs)])
    
    if config.policy_name=='PGPolicy':
        optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)
    else:
        hidden_sizes = utils.extract_hidden_sizes_tuple(config=config)
        critic = CustomCritic(preprocess_net=preprocess_net, hidden_sizes=hidden_sizes, detach_preprocess_output=config.detach_critic_preprocess_output, device=device).to(device)
        actor_critic = ActorCritic(actor=actor, critic=critic).to(device)
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.lr)
    
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', threshold_mode='abs',
                                  factor=config.scheduler_factor, patience=config.scheduler_patience, threshold=config.scheduler_threshold,
                                  verbose=True) if config.use_scheduler else None

    if config.policy_name in {'A2C', 'PPO'}:
        if config.kl_div_coef != 0:
            llm_manager_sft, preprocess_net_sft, actor_sft, dist_fn_sft = load_llm_manager_and_models(config=config, device=device, logger=None, is_sft_model=True)
            for param in actor_sft.parameters():
                param.requires_grad = False
        else:
            llm_manager_sft, preprocess_net_sft, actor_sft, dist_fn_sft = None, None, None, None
    
    action_space = train_envs.action_space[0]
    
    if config.policy_name=='PGPolicy':
        policy = CustomPGPolicy(model=actor, optim=optimizer, action_space=action_space,
                                dist_fn=dist_fn, discount_factor=config.discount_factor, deterministic_eval=config.deterministic_eval, lr_scheduler=scheduler,
                                reward_normalization=False, action_scaling=False)
    elif config.policy_name=='A2C':
        policy = CustomA2CPolicy(actor=actor, critic=critic, action_space=action_space, kl_approx_method=config.kl_approx_method, actor_sft=actor_sft, optim=optimizer,
                                 vf_coef=config.vf_coef, ent_coef=config.ent_coef, kl_div_coef=config.kl_div_coef,
                                 certainty_estimation=config.certainty_estimation,
                                 certainty_scaling_exponent=config.certainty_scaling_exponent, certainty_scaling_intercept=config.certainty_scaling_intercept,
                                 dist_fn=dist_fn, discount_factor=config.discount_factor, deterministic_eval=config.deterministic_eval, lr_scheduler=scheduler,
                                 reward_normalization=False, action_scaling=False, max_batchsize=config.batch_size)
    elif config.policy_name=='PPO':
        policy = CustomPPOPolicy(actor=actor, critic=critic, action_space=action_space, kl_approx_method=config.kl_approx_method, actor_sft=actor_sft, optim=optimizer,
                                 vf_coef=config.vf_coef, ent_coef=config.ent_coef, kl_div_coef=config.kl_div_coef,  certainty_scaling_exponent=config.certainty_scaling_exponent,
                                 certainty_estimation=config.certainty_estimation,
                                 eps_clip=config.eps_clip,
                                 dist_fn=dist_fn, discount_factor=config.discount_factor, deterministic_eval=config.deterministic_eval, lr_scheduler=scheduler,
                                 reward_normalization=False, action_scaling=False, max_batchsize=config.batch_size)
    else:
        raise Exception(f'Policy {config.policy_name} not supported.')
    
    # load model
    if (config.load_model is not None) and (config.load_model != 'none'):
        policy.load_state_dict(torch.load(config.load_model))
    else:
        pass
    
    # collector
    assert config.multiply_factor >= 1
    if config.step_per_collect is None:
        assert config.episode_per_collect is not None
        total_size = 1000 * config.episode_per_collect
    else:
        total_size = int(config.step_per_collect * config.multiply_factor)
    replayBuffer = VectorReplayBuffer(total_size=total_size, buffer_num=len(train_envs))
    
    train_collector = Collector(policy=policy, env=train_envs, buffer=replayBuffer, exploration_noise=False)
    
    eval_confort_collector = Collector(policy=policy, env=eval_envs_confort)
    eval_plus1_collector   = Collector(policy=policy, env=eval_envs_plus1)
    eval_plus2_collector   = Collector(policy=policy, env=eval_envs_plus2)
    if config.operand_max_size=="plus1":
        test_collector_dict = {EVAL_CONFORT:eval_confort_collector, "eval_plus1": eval_plus1_collector}
    elif config.operand_max_size=="plus2":
        test_collector_dict = {EVAL_CONFORT:eval_confort_collector, "eval_plus2": eval_plus2_collector}
    else:
        raise NotImplementedError()
    
    logger.connect_optimizer(optimizer=optimizer)
    
    episode_per_test_dict = {name:config.episode_per_test_confort if name==EVAL_CONFORT else config.episode_per_test for name in test_collector_dict}
    train_results = CustomOnpolicyTrainer(policy=policy,
                                          train_collector=train_collector,
                                          test_collector_dict=test_collector_dict,
                                          max_epoch=config.max_epoch,
                                          step_per_epoch=config.step_per_epoch,
                                          repeat_per_collect=config.repeat_per_collect,
                                          episode_per_test_dict=episode_per_test_dict,
                                          batch_size=config.batch_size,
                                          step_per_collect=config.step_per_collect,
                                          episode_per_collect=config.episode_per_collect,
                                          logger=logger).run(reset_prior_to_run=False)
    
    if hasattr(config, 'save_model') and config.save_model:
        sweep_id = run.sweep_id if run is not None else 'none'
        name     = run.name     if run is not None else 'none'
        import wandb.sdk
        config_dict = config.as_dict() if isinstance(config, wandb.sdk.wandb_config.Config) else vars(config)
        save_rl_training(config_dict=config_dict, results_dict=train_results, policy=policy, llm_manager=llm_manager, sweep_id=sweep_id, name=name)
    
    return train_results, policy

################################################################################################################################################################################################

################################################################################################################################################################################################

def save_rl_training(config_dict:dict, results_dict:dict=None, policy:BasePolicy=None, llm_manager:LLMmanager=None, sweep_id='none', name='none'):
    datetime = utils.current_datetime()
    results_folder_path = os.path.join(PathManager.RL_RESULTS, datetime)
    os.makedirs(results_folder_path)
    
    results_file_path = os.path.join(results_folder_path, "perfs.json")
    config_file_path  = os.path.join(results_folder_path, "config.json")
    infos_path        = os.path.join(results_folder_path, 'infos.json')
    policy_path       = os.path.join(results_folder_path, 'policy.pt')
    llm_path          = os.path.join(results_folder_path, 'llm')

    infos_dict = dict(sweep_id=sweep_id, name=name)
    
    utils.save_dict_as_json(data_dict=config_dict, file_path=config_file_path)
    utils.save_dict_as_json(data_dict=infos_dict, file_path=infos_path)
    if results_dict is not None: utils.save_dict_as_json(data_dict=results_dict, file_path=results_file_path)
    
    if policy is not None:
        torch.save(policy.state_dict(), policy_path)
    
    if llm_manager is not None:
        name = datetime if name=='none' else name
        hugging_face = None if not config_dict['hugging_face'] else name
        llm_manager.save_model(local_path=llm_path, hugging_face=hugging_face)


def sweep_trainer(config_dict=None):
    with wandb.init(config=config_dict) as run:
        train_func(config=wandb.config, use_wandb=True, run=run)


def set_wandb_params(use_wandb:bool, names_dict:dict, mode:str):
    if (not use_wandb) and (mode!='cluster'):
        return None, None
    wandb_names = dict(entity='llm4planning2', project='addition')
    if not 'group' in names_dict.keys(): names_dict['group'] = DEFAULT_WANDB_GROUP_NAME
    metric_goal = None
    return wandb_names, metric_goal


def preprocess_quick_test(config:dict):
    config['step_per_epoch']   = 1
    # config['step_per_collect'] = 16384
    config['step_per_collect'] = None
    config['episode_per_collect'] = 2
    config['num_envs']         = 1
    config['episode_per_test_confort'] = 1
    config['episode_per_test'] = 1 # 2
    config['max_epoch']        = 3
    config['batch_size']       = 2
    config['seed'] = 0
    return config


############################################################################################


if __name__=='__main__':
    
    verbose  = 3
    num_envs = 2
    save_model = False

    quick_test = False

    use_wandb  = True
    use_sweep  = True
    is_offline = False
    
    exp_cfg = dict(
        exp_id       = 'rl_basic_training',
        exp_name     = 'different_exponents',
        seed         = [0,1,2,3,4,5,6,7,8,9],
        device       = 'default',
        verbose      = verbose,
        version      = 'oct2024',
        save_model   = save_model,
        hugging_face = False,
        num_envs     = num_envs
    )
    operation_cfg = dict(
        operand_min_size=['plus1'],
        operand_max_size='plus1',
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
        model_path         = 'lecraquito/gpt2_reduced_vocab_FT_7digits',
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
        max_epoch = 8,
        episode_per_collect = 50
    )
    rl_cfg = dict(
        policy_name = 'A2C'
    )
    optimizer_cfg = dict(
        lr = [1e-6] # 1e-6
    )
    trainer_cfg = dict(
        repeat_per_collect = [1],
        batch_size         = 2 # new
    )
    policy_cfg = dict(
        load_model          = None,
        discount_factor     = 1.0,
        deterministic_eval  = False,
        use_scheduler       = False
    )
    kl_config = dict(
          ent_coef = [0.001],
          kl_div_coef = [5.],
          kl_approx_method         = 'k2', # k1, k2, k3
          certainty_estimation     = ["negentropy_max_prob", None], # None, max_prob, negentropy_max_prob, negentropy_sample, negentropy
          certainty_scaling_exponent = [150],  # None, or > 1
          certainty_scaling_intercept = 0, # 0
          temperature_weighted_kl  = [4],
    )
    actor_critic_cfg = dict(
        vf_coef     = [0.1],
        detach_critic_preprocess_output = [False]
    )
    PPO_cfg = dict(
        eps_clip = [0.2]
    )
    replay_buffer_cfg = dict(
        multiply_factor =  [1.]# [2., 1.5, 1.]
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

