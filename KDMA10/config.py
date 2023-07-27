# expert training
EXPERT_LR = 3e-4
EXPERT_BATCH_SIZE = 64
EXPERT_EPOCHS = 500


# reinforcement learning
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
ENTROPY_LOSS_COEF = 0.1
DISCOUNT_FACTOR = 0.85
GRAD_NORM_CLIP = 1.
OPT_EPOCHS = 5
HORIZON = 4096
BATCH_SIZE = 256
MAX_SAMPLES = 4e7
INIT_ACTION_STD = 0.5


# simulation settings
STEP_TIME = 0.12
FRAME_SKIP = 3
TIMEOUT = 200
VISUALIZATION_TIMEOUT = 200


# agent properties
AGENT_RADIUS = 0.1
PREFERRED_SPEED = 0.4
MAX_SPEED = 0.5
NEIGHBORHOOD_RADIUS = 5.
ACC = False
EXPERT = True
COEFF = 0.03
GOALREW = 2