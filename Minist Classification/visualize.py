from matplotlib import pyplot as plt
losses=[]
epochs=[]

with open("vit_training_records.txt", 'r') as f:
    for i in range(20):
      line=f.readline().split(',')
      loss=line[1].split(':')
      epoch=line[0].split()
      losses.append((float(loss[1])))
      epochs.append(float(epoch[1]))

plt.plot(epochs, losses, 'b', label='Training Loss')  # 'b'代表蓝色线条
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

