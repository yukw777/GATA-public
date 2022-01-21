import yaml
import json
import networkx as nx


from agent import Agent
from dgu.graph import process_triplet_cmd
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
            cmds = []
            for item in predict_cmds:
                if item == "":
                    continue
                parts = item.split()
                cmds.append(
                    " , ".join([parts[0], " ".join(parts[1:-2]), parts[-2], parts[-1]])
                )

            for cmd in cmds:
                print(cmd)
                process_triplet_cmd(graph, t, cmd)

            for n, data in graph.nodes.data():
                data["label"] = n.label
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
