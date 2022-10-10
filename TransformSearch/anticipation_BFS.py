from train_agent import *
from utils import is_transform_actions_influence_anticipated_policy


class TransformNode:
    def __init__(self, actions, idxes, values, val, depth, parents, preconditions_graph):
        self.val = val
        self.depth = depth
        self.actions = actions
        self.idxes = idxes
        self.values = values
        self.parents = parents
        self.adjacent_nodes = []
        self.transform_name = ""
        if not preconditions_graph:
            self.transform_name = ORIGINAL_ENV
        else:
            for a, i, v in zip(self.actions, self.idxes, self.values):
                self.transform_name += f"{a}_{i}_{v}_"
            self.transform_name = self.transform_name[:-1]
        self.visited = False
        if preconditions_graph:
            preconditions_graph.graph[self.transform_name] = self


# This class represents a directed graph using adjacency list representation
class PreconditionsGraph:
    def __init__(self, env_name, preconditions, anticipated_policy):
        self.root = TransformNode(actions=[], idxes=[], values=[], val=0, depth=0, parents=[],
                                  preconditions_graph=None)
        self.anticipated_policy = anticipated_policy
        self.queue = [self.root]
        self.root.visited = True
        self.graph = defaultdict(TransformNode)
        self.graph[ORIGINAL_ENV] = self.root
        self.max_depth = len(anticipated_policy) + 1
        self.build_preconditions_graph(preconditions, anticipated_policy)
        self.original_env = get_env(env_name)

    def add_edge(self, u, v):
        self.graph[u].adjacent_nodes.append(v)

    def bfs(self):
        self.original_env = get_env(SINGLE_TAXI_EXAMPLE)  # TODO - Temporary!! to delete!!
        satisfaction = False
        explanation_env = None
        while self.queue and not satisfaction:
            transform_node = self.queue.pop(0)
            print(transform_node.transform_name, end=" ")
            for neighbour_mdp in transform_node.adjacent_nodes:
                if not neighbour_mdp.visited:
                    self.queue.append(neighbour_mdp)
                    neighbour_mdp.visited = True
                    satisfaction = train_agent_and_save_results(self.original_env, neighbour_mdp)
                    if satisfaction:
                        explanation_env = neighbour_mdp
                        break
        if satisfaction:
            print(f"Congratulations!! We have an explanation env for you! Its name: {explanation_env.transform_name}")
        else:
            print("Sorry, we don't know why the agent doing weird stuff :-(")

    def get_graph_node_by_name(self, node_name):
        return self.graph[node_name]

    def update_graph(self, all_preconditions_list, depth, precondition_node):
        possible_parents = all_preconditions_list[depth - 1]
        pre_actions_possible_parents = [get_precondition_actions_from_string(pre_string[0][0])[0] for pre_string in
                                        possible_parents]
        for i, action in enumerate(pre_actions_possible_parents):
            if action in precondition_node.actions:
                parent_node = self.get_graph_node_by_name(possible_parents[i][0])
                precondition_node.parents.append(parent_node)
                parent_node.adjacent_nodes.append(precondition_node)

    def build_preconditions_graph(self, preconditions, anticipated_policy):
        basic_preconditions_list = []
        depth = 1
        for action, action_info in preconditions.items():
            for idx, values in action_info.items():
                for val in values:
                    pre_string = f"{action}_{idx}_{val}"
                    precondition_actions = get_precondition_actions_from_string(pre_string)
                    influence = is_transform_actions_influence_anticipated_policy(list(anticipated_policy.values()),
                                                                                  precondition_actions)
                    if influence:
                        basic_preconditions_list.append(f"{action}_{idx}_{val}")
                        node = TransformNode(actions=[action], idxes=[idx], values=[val], val=0, depth=depth,
                                             parents=[self.root], preconditions_graph=self)
                        self.root.adjacent_nodes.append(node)

        all_preconditions_list = {1: tuple(tuple([pre]) for pre in basic_preconditions_list)}
        for i in range(2, self.max_depth):
            depth += 1
            print(f"depth: {depth}")
            all_preconditions_list[i] = tuple(itertools.combinations(basic_preconditions_list, i))
            for pre in all_preconditions_list[i]:
                actions, idxes, vals = list(), tuple(), list()
                idxes, vals = list(), list()
                for p in pre:
                    action = get_precondition_actions_from_string(p)
                    idx = get_precondition_idx_from_string(p)
                    val = get_precondition_val_from_string(p)
                    actions += action
                    idxes.append(idx)
                    vals.append(val)
                node = TransformNode(actions=actions, idxes=idxes, values=vals, val=0, depth=depth, parents=[],
                                     preconditions_graph=self)
                self.update_graph(all_preconditions_list, depth, node)


def get_precondition_from_transform_node(mdp_node):
    precondition = {}
    for act, idx, val in zip(mdp_node.actions, mdp_node.idxes, mdp_node.values):
        if act in precondition.keys():
            if idx in precondition[act].keys():
                precondition[act][idx].append(val)
            else:
                precondition[act][idx] = [val]
        else:
            precondition[act] = {idx: val}
    return precondition


def train_agent_and_save_results(original_env, mdp_node):
    # transform_name, new_env = load_transform_by_name(neighbour_mdp.transform_name + ".pkl")
    precondition = get_precondition_from_transform_node(mdp_node)
    new_env = generate_transformed_env(precondition, env_file_name=TRANSFORMS_PATH + mdp_node.transform_name, save=True,
                                       try_load_existing=True)
    satisfaction = generate_agent(original_env, KERAS_DQN, ITER_NUM, new_env, mdp_node.transform_name)
    return satisfaction

# if __name__ == '__main__':
#     from Transforms import env_precinditions
#
#     cur_env_preconditions = load_pkl_file("../" + TAXI_TRANSFORM_DATA_PATH + "small_taxi_env_preconditions.pkl")
#     # precondition_graph = PreconditionsGraph(cur_env_preconditions.not_allowed_features, ANTICIPATED_POLICY)
#     precondition_graph = load_pkl_file("precondition_graph_small_taxi_env.pkl")
#     save_pkl_file("precondition_graph_small_taxi_env.pkl", precondition_graph)
#     precondition_graph.bfs()
