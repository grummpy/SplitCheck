"""Fixed settings for the lab. Change these only if you mean to change the experiment."""

DEFAULT_SEED = 42

N_PATIENTS = 360
VISITS_PER_PATIENT = 4
TEST_SIZE = 0.25

# Clinical features are a weak function of a patient-level risk draw.
SIGNAL = 3.5
NOISE = 6.5
VISIT_NOISE = {"age": 1.5, "bmi": 0.8, "systolic_bp": 4.0}

N_FINGERPRINTS = 4
FINGERPRINT_NOISE = 0.01

# Shrinkage for the billing-code target encoding: (sum(y) + m * mean(y)) / (count + m).
TARGET_ENCODING_SMOOTHING = 1.0

# Chebyshev distance on numeric features. Exact copies sit at 0; planted near-copies sit at 5e-4.
EXACT_ATOL = 1e-8
NEAR_ATOL = 1e-3
NEAR_OFFSET = 5e-4

PLANTED_EXACT_COPIES = 11
PLANTED_NEAR_COPIES = 9
PLANTED_SHARED_GROUPS = ("SHARE0", "SHARE1", "SHARE2", "SHARE3", "SHARE4")

CLINICAL_FEATURES = ("age", "bmi", "systolic_bp")
FINGERPRINT_FEATURES = ("fp0", "fp1", "fp2", "fp3")
AUDIT_FEATURES = CLINICAL_FEATURES + FINGERPRINT_FEATURES + ("billing_code",)
GROUP_COLUMN = "patient_id"
TARGET_COLUMN = "y"

N_TREES = 200
LOGREG_MAX_ITER = 2000
