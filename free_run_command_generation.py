import torch
import tqdm
import yaml

from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Any

from dgu.data import TWCmdGenGraphEventFreeRunDataset
from dgu.metrics.f1 import F1
from dgu.metrics.exact_match import ExactMatch

from agent import Agent


def main(
    config_filename: str,
    data_filename: str,
    ckpt_filename: str,
    f1_scores_filename: str,
    em_scores_filename: str,
    batch_size: int,
) -> None:
    with open(config_filename) as f:
        config = yaml.safe_load(f)
    dataset = TWCmdGenGraphEventFreeRunDataset(data_filename, batch_size)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1)
    agent = Agent(config)
    agent.load_pretrained_model(ckpt_filename, load_partial_graph=False)
    agent.eval()

    graph_f1 = F1()
    graph_em = ExactMatch()

    game_id_to_step_data_graph: Dict[int, Tuple[Dict[str, Any], List[List[str]]]] = {}
    with tqdm.tqdm(total=len(dataset)) as pbar:
        for batch in dataloader:
            # finished games are the ones that were in game_id_to_graph, but are not
            # part of the new batch
            for finished_game_id in game_id_to_step_data_graph.keys() - {
                game_id for game_id, _ in batch
            }:
                step_data, graph = game_id_to_step_data_graph.pop(finished_game_id)
                groundtruth_graph = agent.update_knowledge_graph_triplets(
                    [
                        [
                            triple_str.split(" , ")
                            for triple_str in step_data["previous_graph_seen"]
                        ]
                    ],
                    [
                        " <sep> ".join(
                            " ".join(cmd.split(" , "))
                            for cmd in step_data["target_commands"]
                        )
                        + " <eos>"
                    ],
                )
                rdfs = [[" , ".join(triple) for triple in graph]]
                groundtruth_rdfs = [
                    [" , ".join(triple) for triple in groundtruth_graph[0]]
                ]
                graph_f1.update(rdfs, groundtruth_rdfs)
                graph_em.update(rdfs, groundtruth_rdfs)
                pbar.update()
                pbar.set_postfix(
                    {
                        "graph_f1": graph_f1.compute().item(),
                        "graph_em": graph_em.compute().item(),
                    }
                )

            # new games are the ones that were not in game_id_to_graph, but are now
            # part of the new batch.
            # due to Python's dictionary ordering (insertion order), new games are
            # added always to the end.
            for game_id, step_data in batch:
                if game_id in game_id_to_step_data_graph:
                    _, graph = game_id_to_step_data_graph[game_id]
                    game_id_to_step_data_graph[game_id] = (step_data, graph)
                else:
                    game_id_to_step_data_graph[game_id] = (step_data, [])

            # sanity check
            assert [game_id for game_id, _ in batch] == [
                game_id for game_id in game_id_to_step_data_graph
            ]

            # construct a batch
            batched_obs: List[str] = []
            graph_list: List[List[List[str]]] = []
            for game_id, (step_data, graph) in game_id_to_step_data_graph.items():
                batched_obs.append(
                    step_data["observation"] + " <sep> " + step_data["previous_action"]
                )
                graph_list.append(graph)

            # greedy decode
            with torch.no_grad():
                batch_pred_strings = agent.command_generation_greedy_generation(
                    batched_obs, graph_list
                )

            # update graphs in game_id_to_step_data_graph
            for (game_id, (step_data, graph)), pred_strings in zip(
                game_id_to_step_data_graph.items(), batch_pred_strings
            ):
                game_id_to_step_data_graph[game_id] = (
                    step_data,
                    agent.update_knowledge_graph_triplets([graph], [pred_strings])[0],
                )
    print(f"Free Run Graph F1: {graph_f1.compute()}")
    print(f"Free Run Graph EM: {graph_em.compute()}")
    if f1_scores_filename:
        torch.save(graph_f1.scores.cpu(), f1_scores_filename)
    if em_scores_filename:
        torch.save(graph_em.scores.cpu(), em_scores_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename")
    parser.add_argument("data_filename")
    parser.add_argument("ckpt_filename")
    parser.add_argument("--f1-scores-filename", default="")
    parser.add_argument("--em-scores-filename", default="")
    parser.add_argument("--batch-size", default=512, type=int)
    args = parser.parse_args()
    main(
        args.config_filename,
        args.data_filename,
        args.ckpt_filename,
        args.f1_scores_filename,
        args.em_scores_filename,
        args.batch_size,
    )
