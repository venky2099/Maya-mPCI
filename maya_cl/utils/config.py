# config.py -- Maya-Prana (Paper 9) hyperparameters
# Carries forward P8 (Maya-Shunyata) base. Adds Prana metabolic plasticity budget.
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

SEED = 42
T_STEPS = 4
CONV1_CHANNELS = 64
CONV2_CHANNELS = 64
CONV3_CHANNELS = 128
FC1_SIZE = 2048
NUM_CLASSES = 100
TAU_MEMBRANE = 2.0
V_THRESHOLD = 0.3
V_RESET = 0.0

TAU_SHRADDHA = 10.0
TAU_BHAYA = 3.0
TAU_VAIRAGYA = 20.0
TAU_SPANDA = 5.0
TAU_VIVEKA = 50.0
TAU_BUDDHI = 200.0

HEBBIAN_LR = 0.01
LABILITY_INIT = 1.0
LABILITY_PAIN_BOOST = 5.0
LABILITY_DECAY_RATE = 0.95
PAIN_CONFIDENCE_THRESHOLD = 0.25

VAIRAGYA_DECAY_RATE = 0.002315
VAIRAGYA_PROTECTION_THRESHOLD = 0.3
VAIRAGYA_ACCUMULATE_RATE = 0.0015
VAIRAGYA_PAIN_EROSION_RATE = 0.005

VIVEKA_CONSISTENCY_RISE = 0.01
VIVEKA_CONSISTENCY_DECAY = 0.005
VIVEKA_GAIN_MAX = 3.0
VIVEKA_MIN_TASKS = 2

USE_ORTHOGONAL_HEAD = False
PROTOTYPE_DIM = 2048
NUM_TASKS = 10
CLASSES_PER_TASK = 10
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20
REPLAY_BUFFER_SIZE = 50
REPLAY_RATIO = 0.3
REPLAY_VAIRAGYA_PARTIAL_LIFT = 0.8
REPLAY_PAIN_EXEMPT = True
CIL_BOUNDARY_DECAY = 0.50
CIL_MAX_VFOUT_PROTECTION = 0.70

# Chitta -- carried from P6/P7/P8, unchanged
CHITTA_SAMSKARA_RISE = 0.002315
CHITTA_SAMSKARA_DECAY = 0.0007
CHITTA_MOHA_THRESHOLD = 0.95
CHITTA_MOHA_RELEASE_RATE = 0.60
CHITTA_MIN_TASKS = 1
CHITTA_GATE_STRENGTH = 0.30

# Manas -- carried from P7/P8, unchanged
A_MANAS = 0.10
MANAS_GANE_PEAK_THRESHOLD = 0.5
MANAS_MIN_TASKS = 0

# Karma -- carried from P8, unchanged
KARMA_ACCUMULATE_RATE = 1.0
KARMA_THRESHOLD = 0.05
KARMA_THRESHOLD_LOW = 0.03
KARMA_THRESHOLD_HIGH = 0.75
KARMA_DECAY_RATE = 0.002315
KARMA_MIN_TASKS = 1
SHUNYATA_PRUNE_AT_BOUNDARY = True
SHUNYATA_MASK_RECOVERY = False

# Prana -- P9 new contribution
# Metabolic plasticity budget. Biological ground: astrocyte-neuron lactate shuttle (ANLS).
# Prana depletes under learning load, recovers during rest, gates effective learning rate.
# High Prana = full plasticity available. Low Prana = metabolic starvation, learning suppressed.
PRANA_INIT = 1.0                  # full budget at start
PRANA_COST_RATE = 0.002315        # ORCID magic number -- depletion per unit gradient magnitude
PRANA_RECOVERY_RATE = 0.05        # recovery toward 1.0 during low-activity batches
PRANA_MIN = 0.05                  # floor -- never fully starved (biological: baseline ANLS supply)
PRANA_RECOVERY_THRESHOLD = 0.3    # activity level below which recovery fires
PRANA_AGGRESSIVE_COST = 0.008     # ablation E -- aggressive depletion
PRANA_DISABLED_VALUE = 1.0        # ablation C -- fixed Prana (constant 1.0, no dynamics)

DATA_DIR = "data/"
RESULTS_DIR = "results/"
# mPCI -- Machine Perturbational Complexity Index experiment constants
# Paper: "From Representation to Experience"
# Biological ground: Perturbational Complexity Index (Casali et al., 2013)
# adapted for SNN spike trains via Lempel-Ziv Complexity.
MPCI_PERTURBATION_SCALE = 0.05    # Gaussian noise std on fc1 weights per trial
MPCI_N_PERTURBATIONS    = 10      # number of independent perturbation trials
MPCI_SEED_BASE          = 42      # base seed -- trial i uses MPCI_SEED_BASE + i
MPCI_BATCH_SIZE         = 32      # batch size for mPCI forward passes
MPCI_QUIESCENCE_TASK    = 4       # task index after which Bhaya=0.000 confirmed
MPCI_THRESHOLD_SIGMA    = 2.0     # pooled SD multiplier for genuine state detection
