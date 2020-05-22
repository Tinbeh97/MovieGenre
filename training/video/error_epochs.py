import numpy as np
import matplotlib.pyplot as plt

loss_path = #where you saved loss per epoch 
os.chdir(loss_path)

#plotting training loss per epoch
l1 = np.load('LSTM_merge_loss.npy')
l2 = np.load('model_mm_9_genre_4_loss.npy')
x1 = np.arange(len(l1))
x2 = np.arange(len(l2))
fig, ax = plt.subplots()
ax.plot(x1, l1,'-b', label='LSTM')
ax.plot(x2, l2,'C1', label='1D convolutional')
ax.plot(epoch, error_CCT,'-g', label='CTT-MMC-C')
plt.title('loss of various methods')
plt.xlabel('epoch')
plt.ylabel('training loss')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large', fancybox=True)

#ploting validation loss per epoch
l1 = np.load('LSTM_merge_val_loss.npy')
l2 = np.load('model_mm_9_genre_4_val_loss.npy')
x1 = np.arange(len(l1))
x2 = np.arange(len(l2))
plt.figure()
plt.plot(x1, l1,'-b', label='LSTM')
plt.plot(x2, l2,'C1', label='1D convolutional')
plt.title('validation loss of various methods', fontsize=12)
plt.xlabel('epoch', fontsize=12)
plt.ylabel('validation loss', fontsize=12)
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large', fancybox=True)
plt.savefig('/Users/tina/Downloads/epoch.pdf', format='pdf', dpi=1200)

