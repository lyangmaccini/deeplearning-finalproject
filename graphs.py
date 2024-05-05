import csv
import matplotlib.pyplot as plt
bs_losses = []
bs_mean_squared_errors = []
bs_r2scores = []
epochs = [*range(1, 201)]
with open('black-scholes.log', newline='') as csvfile:
    black_scholes_data = csv.reader(csvfile)
    a = 1
    for line in black_scholes_data:
        _,loss,lr,mean_squared_error,r2score_fn,val_loss,val_mean_squared_error,val_r2score_fn = line
        if a == 1:
            a = 0
        else:
            bs_losses.append(float(loss))
            bs_mean_squared_errors.append(float(mean_squared_error))
            bs_r2scores.append(float(r2score_fn))
bs_losses.pop(0)
bs_mean_squared_errors.pop(0)
bs_r2scores.pop(0)
print(bs_mean_squared_errors)
# plt.plot(bs_r2scores)
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy (R2 Score)")
# plt.title("Black-Scholes Model: Accuracy (R2 Score) across 200 Epochs")
# plt.show()

plt.plot(bs_mean_squared_errors)
plt.xlabel("Epochs")
plt.ylabel("Loss (Mean Squared Error)")
plt.title("Black-Scholes Model: Loss (Mean Squared Error) across 200 Epochs")
plt.show()
