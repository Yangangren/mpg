from worker import OnPolicyWorker
from evaluator import Evaluator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer(object):
    def __init__(self, policy_cls, learner_cls, buffer_cls, optimizer_cls, args):
        self.args = args
        self.local_worker = OnPolicyWorker(policy_cls, learner_cls, self.args.env_id, self.args, 0)
        # self.evaluator = Evaluator(policy_cls, self.args.env_id, self.args)
        self.evaluator = None
        self.optimizer = optimizer_cls(self.local_worker, self.evaluator, self.args)

    def load_weights(self, load_dir, iteration):
        self.local_worker.load_weights(load_dir, iteration)
        # self.evaluator.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.local_worker.load_ppc_params(load_dir)
        # self.evaluator.load_ppc_params(load_dir)

    def train(self):
        logger.info('training beginning')
        while self.optimizer.num_sampled_steps < self.args.max_sampled_steps \
                or self.optimizer.num_updated_steps < self.args.max_updated_steps:
            self.optimizer.step()
        self.optimizer.stop()
