import yaml
import json
import networkx as nx


from agent import Agent
from dgu.utils import draw_graph


def main(
    config_filename: str, data_filename: str, ckpt_filename: str, graph_filename: str
) -> None:
    with open(config_filename) as f:
        config = yaml.safe_load(f)
    agent = Agent(config)
    agent.load_pretrained_model(ckpt_filename, load_partial_graph=False)
    agent.eval()

    prev_action = "restart"
    graph = nx.DiGraph()
    triplets = [[]]
    with open(data_filename) as f:
        for t, line in enumerate(f):
            step = json.loads(line)
            pred_strings = agent.command_generation_greedy_generation(
                [f"{step['observation']} <sep> {prev_action}"], triplets
            )
            predict_cmds = pred_strings[0].split("<sep>")
            if predict_cmds[-1].endswith("<eos>"):
                predict_cmds[-1] = predict_cmds[-1][:-5].strip()
            else:
                predict_cmds = predict_cmds[:-1]
            for item in predict_cmds:
                if item == "":
                    continue
                parts = item.split()
                cmd = parts[0].strip()
                src = " ".join(parts[1:-2])
                dst = parts[-2]
                edge = parts[-1]
                if cmd == "add":
                    graph.add_edge(src, dst, label=edge)
                elif cmd == "delete":
                    graph.remove_edge(src, dst)
                else:
                    raise ValueError(f"Unknown command: {cmd}")
            draw_graph(graph, graph_filename)
            input(">> ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename")
    parser.add_argument("data_filename")
    parser.add_argument("ckpt_filename")
    parser.add_argument("graph_filename")
    args = parser.parse_args()
    main(
        args.config_filename,
        args.data_filename,
        args.ckpt_filename,
        args.graph_filename,
    )
