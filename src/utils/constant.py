DATASET_DIR = '../dataset/'     # processed train/valid/test data directory
DATA_DIR = '../data/'           # raw data directory

# Preprocess/DataLoader
TRAIN_SUFFIX = '.train.tsv'             # train file suffix
VALIDATION_SUFFIX = '.validation.tsv'   # validation file suffix
TEST_SUFFIX = '.test.tsv'               # test file suffix
ALL_SUFFIX = '.all.tsv'                 # all data file

# Recommendation
UID = 'uid'                     # head node column name
IID = 'iid'                     # tail node column name
SEQ = 'sequence'                # sequence column name
LABEL = 'label'                 # label column name
SAMPLE_ID = 'sample_id'         # sample id for each record
ARC = 'sampled_arc'
NEG_ITEM = 'neg_items'          # negative samples for validation/testing
NEG_SYMBOL = '~'                # negation symbol
