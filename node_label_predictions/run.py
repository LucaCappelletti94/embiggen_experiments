from experiment import run_node_label_prediction

if __name__ == "__main__":
    performance = run_node_label_prediction()
    performance.to_csv("node_label_performance.csv")
