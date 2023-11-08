import matplotlib.pyplot as plt


def show_plot(samples):
    for sample in samples:
        plt.plot(sample)
    plt.show(block=True)
