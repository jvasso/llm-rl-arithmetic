import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import BaseLogger
from tianshou.data.collector import BaseCollector

from tianshou.data import (
    CollectStats,
    InfoStats,
    SequenceSummaryStats,
    TimingStats,
)

from ..custom_wandb_logger import CustomWandbLogger



# def custom_test_episode(
#     policy: BasePolicy,
#     collector: Collector,
#     test_fn: Optional[Callable[[int, Optional[int]], None]],
#     epoch: int,
#     n_episode: int,
#     is_final_update:bool,
#     logger: Optional[BaseLogger] = None,
#     global_step: Optional[int] = None,
#     reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
# ) -> Dict[str, Any]:
#     """A simple wrapper of testing policy in collector."""
#     collector.reset_env()
#     collector.reset_buffer()
#     policy.eval()
#     if test_fn:
#         test_fn(epoch, global_step)
#     result = collector.collect(n_episode=n_episode)
#     if reward_metric:
#         rew = reward_metric(result["rews"])
#         result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
    
#     if logger and global_step is not None:
#         # custom add-on
#         assert isinstance(logger, CustomWandbLogger)
#         env_type = collector.env.get_env_attr('env_type')[0]
#         logger.log_test_data(result, global_step, env_type, is_final_update=is_final_update)
#     return result
def custom_test_episode(
    collector: BaseCollector,
    test_fn: Callable[[int, int | None], None] | None,
    epoch: int,
    n_episode: int,
    is_final_update: bool,
    logger: BaseLogger | None = None,
    global_step: int | None = None,
    reward_metric: Callable[[np.ndarray], np.ndarray] | None = None,
) -> CollectStats:
    """A simple wrapper of testing policy in collector."""
    collector.reset(reset_stats=False)
    if test_fn:
        test_fn(epoch, global_step)
    result = collector.collect(n_episode=n_episode)
    if reward_metric:  # TODO: move into collector
        rew = reward_metric(result.returns)
        result.returns = rew
        result.returns_stat = SequenceSummaryStats.from_sequence(rew)
    if logger and global_step is not None:
        assert result.n_collected_episodes > 0
        # custom add-on
        assert isinstance(logger, CustomWandbLogger)
        env_type = collector.env.get_env_attr('env_type')[0]
        # check how difficult it is to access infos in collector
        logger.log_test_data(asdict(result), global_step, env_type, is_final_update=is_final_update)
    return result



# def custom_gather_info(
#     start_time: float,
#     train_collector: Optional[Collector],
#     test_collector: Optional[Collector],
#     best_reward: float,
#     best_reward_std: float,
# ) -> Dict[str, Union[float, str]]:
#     """A simple wrapper of gathering information from collectors.

#     :return: A dictionary with the following keys:

#         * ``train_step`` the total collected step of training collector;
#         * ``train_episode`` the total collected episode of training collector;
#         * ``train_time/collector`` the time for collecting transitions in the \
#             training collector;
#         * ``train_time/model`` the time for training models;
#         * ``train_speed`` the speed of training (env_step per second);
#         * ``test_step`` the total collected step of test collector;
#         * ``test_episode`` the total collected episode of test collector;
#         * ``test_time`` the time for testing;
#         * ``test_speed`` the speed of testing (env_step per second);
#         * ``best_reward`` the best reward over the test results;
#         * ``duration`` the total elapsed time.
#     """
#     duration = max(0, time.time() - start_time)
#     model_time = duration
#     result: Dict[str, Union[float, str]] = {
#         "duration": f"{duration:.2f}s",
#         "train_time/model": f"{model_time:.2f}s",
#     }
#     if test_collector is not None:
#         # custom add-on
#         test_collector = test_collector[0]
        
#         model_time = max(0, duration - test_collector.collect_time)
#         test_speed = test_collector.collect_step / test_collector.collect_time
#         result.update(
#             {
#                 "test_step": test_collector.collect_step,
#                 "test_episode": test_collector.collect_episode,
#                 "test_time": f"{test_collector.collect_time:.2f}s",
#                 "test_speed": f"{test_speed:.2f} step/s",
#                 "best_reward": best_reward,
#                 "best_result": f"{best_reward:.2f} ± {best_reward_std:.2f}",
#                 "duration": f"{duration:.2f}s",
#                 "train_time/model": f"{model_time:.2f}s",
#             }
#         )
#     if train_collector is not None:
#         model_time = max(0, model_time - train_collector.collect_time)
#         if test_collector is not None:
#             train_speed = train_collector.collect_step / (
#                 duration - test_collector.collect_time
#             )
#         else:
#             train_speed = train_collector.collect_step / duration
#         result.update(
#             {
#                 "train_step": train_collector.collect_step,
#                 "train_episode": train_collector.collect_episode,
#                 "train_time/collector": f"{train_collector.collect_time:.2f}s",
#                 "train_time/model": f"{model_time:.2f}s",
#                 "train_speed": f"{train_speed:.2f} step/s",
#             }
#         )
#     return result

def custom_gather_info(
    start_time: float,
    policy_update_time: float,
    gradient_step: int,
    best_reward: float,
    best_reward_std: float,
    train_collector: BaseCollector | None = None,
    test_collector_dict: Dict[str,BaseCollector] | None = None,
) -> InfoStats:
    """A simple wrapper of gathering information from collectors.

    :return: InfoStats object with times computed based on the `start_time` and
        episode/step counts read off the collectors. No computation of
        expensive statistics is done here.
    """
    duration = max(0.0, time.time() - start_time)
    test_time = 0.0
    update_speed = 0.0
    train_time_collect = 0.0
    if test_collector_dict is not None:
        # custom add-on
        test_collector_names = list(test_collector_dict.keys())
        test_collector = test_collector_dict[test_collector_names[0]]
        test_time = test_collector.collect_time
    
    if train_collector is not None:
        train_time_collect = train_collector.collect_time
        update_speed = train_collector.collect_step / (duration - test_time)

    timing_stat = TimingStats(
        total_time=duration,
        train_time=duration - test_time,
        train_time_collect=train_time_collect,
        train_time_update=policy_update_time,
        test_time=test_time,
        update_speed=update_speed,
    )

    return InfoStats(
        gradient_step=gradient_step,
        best_reward=best_reward,
        best_reward_std=best_reward_std,
        train_step=train_collector.collect_step if train_collector is not None else 0,
        train_episode=train_collector.collect_episode if train_collector is not None else 0,
        test_step=test_collector.collect_step if test_collector is not None else 0,
        test_episode=test_collector.collect_episode if test_collector is not None else 0,
        timing=timing_stat,
    )
