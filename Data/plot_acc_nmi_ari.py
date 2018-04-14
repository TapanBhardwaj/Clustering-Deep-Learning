import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('final_weights/idec/mnist/idec_log.csv', index_col=False)
df.plot(x='iter', y=['acc', 'nmi', 'ari'])
plt.savefig('IDEC_MNIST.png')
plt.show()
print(df.head())
print(df.columns)
