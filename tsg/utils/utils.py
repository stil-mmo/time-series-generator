import matplotlib.pyplot as plt


def show_plot(samples):
    for sample in samples:
        plt.plot(sample)
    plt.show(block=True)
    plt.close()


def info_to_string(info: dict) -> str:
    line = ""
    for key, value in info.items():
        line += f"{key}: {value}\n"
    return line


def draw_process_plot(values, info, path=None):
    plt.plot(values)
    plt.title(info["name"])
    plt.xlabel("timestamp")
    plt.ylabel("value")
    plt.annotate(
        info_to_string(info), xy=(0.05, 0.0), xycoords="axes fraction", fontsize=5
    )
    if path is None:
        plt.show(block=True)
    else:
        plt.savefig(path)
    plt.close()
