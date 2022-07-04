import autoencoder_glove_train as glove
import autoencoder_synthetic_train as synthetic

'''
End to end pipeline for running the autoencoder on both the synthetic and glove data
'''
if __name__ == "__main__":
    glove.run_glove()
    synthetic.run_synthetic()