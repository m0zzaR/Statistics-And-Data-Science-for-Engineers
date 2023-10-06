import pickle

with open('data.pickle','rb') as f:
    DI, DImany, DII = pickle.load(f)

def draw_50_samples_from_I():
    return DI

def draw_50_samples_from_I_300_times():
    return DImany

def draw_50_samples_from_II():
    return DII
