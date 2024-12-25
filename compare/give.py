import pickle
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

font = FontProperties(family='sans-serif', size=12)


def load_pickle(file_path, label):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if label == 'AAESS':
            if 'results' in data and isinstance(data['results'], list) and len(data['results']) > 0:
                result = data['results'][0]  # or another rule for best result
                if 'fprs' in result and 'tprs' in result:
                    return result['fprs'], result['tprs']
                else:
                    print(f"Warning: Missing 'fprs' or 'tprs' in results for {label} in {file_path}")
                    return None, None
            else:
                print(f"Warning: 'results' key is missing or empty for {label} in {file_path}")
                return None, None
        else:
            if 'fprs' in data and 'tprs' in data:
                return data['fprs'], data['tprs']
            else:
                print(f"Warning: Missing 'fprs' or 'tprs' for {label} in {file_path}")
                return None, None

    except Exception as e:
        print(f"Failed to load data from {file_path} for {label}: {str(e)}")
        return None, None


def plot_curves(file_paths, labels, save_path=None):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    label_to_color = {label: color for label, color in zip(labels, colors)}

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for i, file_path in enumerate(file_paths):
        fprs, tprs = load_pickle(file_path, labels[i])
        if fprs is not None and tprs is not None:
            line_style = '--' if labels[i] in ['LDA', 'LR', 'GBDT', 'RF', 'NN', 'XGB'] else '-'
            ax.plot(fprs, tprs, linestyle=line_style, label=labels[i], color=label_to_color[labels[i]])
        else:
            print(f"Warning: No data for {file_path}")

    axins = inset_axes(ax, width="20%", height="20%", loc='lower center', borderpad=2.5)
    for i, file_path in enumerate(file_paths):
        fprs, tprs = load_pickle(file_path, labels[i])
        if fprs is not None and tprs is not None and labels[i] in ['AAESS', 'XGB']:
            axins.plot(fprs, tprs, linestyle='-', label=labels[i], color=label_to_color[labels[i]])
            axins.set_xlim(0.95, 0.98)
            axins.set_ylim(0.95, 1.0)

    ax.set_xlabel('False Positive Rate (%)')
    ax.set_ylabel('True Positive Rate (%)')
    ax.set_title('ROC Curves Comparison Across Different Algorithms')
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved at {save_path}")

    plt.show()


file_paths = [
    r'D:\study\second\outcome\give\lr_res.pickle',
    r'D:\study\second\outcome\give\lda_res.pickle',
    r'D:\study\second\outcome\give\dt_res.pickle',
    r'D:\study\second\outcome\give\knn_res.pickle',
    r'D:\study\second\outcome\give\adaboost_res.pickle',
    r'D:\study\second\outcome\give\rf_res.pickle',
    r'D:\study\second\outcome\give\gbdt_res.pickle',
    r'D:\study\second\outcome\give\xgb_res.pickle',
    r'D:\study\second\outcome\give\lgb_res.pickle',
    r'D:\study\second\outcome\give\AAESS_res.pickle',
]
labels = ['LR', 'LDA', 'DT', 'KNN', 'Adaboost', 'RF', 'gbdt', 'XGB', 'LGB', 'AAESS']

plot_curves(file_paths, labels,
            save_path=r"D:\study\second\picture\compare\give.png")