from mesa import Model, Agent
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import random
import networkx as nx
from digraph import cluster_nodes, k_shortest_paths


class PathModel(Model):
    def __init__(self, graph, num_agents, origin_id, destination_id, route_defining_locations_ids,
                 agents_familiarity_rates):
        super().__init__()

        self.graph = graph
        self.grid = NetworkGrid(self.graph)
        self.num_agents = num_agents
        self.origin_id = origin_id
        self.destination_id = destination_id
        self.route_defining_locations_ids = route_defining_locations_ids
        self.agents_familiarity_rates = agents_familiarity_rates

        self.grid.set_distance_to_travel_time(1)
        for (u, v, data) in self.graph.edges(data=True):
            self.grid.set_edge_weight((u, v), data['distance'])

        (self.low_bet_cent_node_list, self.middle_bet_cent_node_list, self.high_bet_cent_node_list,
         self.low_lh_node_list, self.middle_lh_node_list, self.high_lh_node_list) = \
            cluster_nodes(self.graph)

        self.create_agents()

        self.datacollector = DataCollector(
            agent_reporters={
                "traversed_path": lambda a: a.traversed_path,
                "traversed_path_distance": lambda a: a.traversed_path_distance,
                "traversed_path_complexity": lambda a: a.traversed_path_complexity,
                "state_familiarity": lambda a: a.state_familiarity,
                "state_traversed_path": lambda a: a.state_traversed_path
            })

    def create_agents(self):
        for i in range(self.num_agents):
            agent = PathAgent(i, self, self.origin_id, self.agents_familiarity_rates[i])
            self.schedule.add(agent)
            self.grid.place_agent(agent, self.origin_id)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def run_model(self, num_steps):
        for i in range(num_steps):
            self.step()


class PathAgent(Agent):
    def __init__(self, unique_id, model, origin_id, agent_familiarity_rate):
        super().__init__(unique_id, model)
        self.knowledge = []  # list of familiar location's id (location = decision points)
        self.initialize_knowledge(agent_familiarity_rate, origin_id)
        self.traversed_path = []
        self.traversed_path_distance = 0
        self.traversed_path_complexity = 0
        self.location = origin_id
        self.state_familiarity = 0
        self.state_traversed_path = []
        self.traversed_path.append(self.location)

    def set_global_knowledge(self, agent_familiarity_rate):
        global_knowledge = set()
        nodes_num = self.model.G.number_of_nodes()
        low_bet_cent_node_list_length = len(self.model.low_bet_cent_node_list)
        middle_bet_cent_node_list_length = len(self.model.middle_bet_cent_node_list)
        high_bet_cent_node_list_length = len(self.model.high_bet_cent_node_list)

        low_percent = agent_familiarity_rate * (low_bet_cent_node_list_length / nodes_num)
        middle_percent = agent_familiarity_rate * (middle_bet_cent_node_list_length / nodes_num)
        high_percent = agent_familiarity_rate * (high_bet_cent_node_list_length / nodes_num)

        low_num = int(low_percent * low_bet_cent_node_list_length)
        middle_num = int(middle_percent * middle_bet_cent_node_list_length)
        high_num = int(high_percent * high_bet_cent_node_list_length)

        low_bet_know = random.sample(self.model.low_bet_cent_node_list,
                                     k=min(low_num, len(self.model.low_bet_cent_node_list)))
        middle_bet_know = random.sample(self.model.middle_bet_cent_node_list,
                                        k=min(middle_num, len(self.model.middle_bet_cent_node_list)))
        high_bet_know = random.sample(self.model.high_bet_cent_node_list,
                                      k=min(high_num, len(self.model.high_bet_cent_node_list)))
        global_bet_knowledge = low_bet_know + middle_bet_know + high_bet_know

        low_lh_node_list_length = len(self.model.low_lh_node_list)
        middle_lh_node_list_length = len(self.model.middle_lh_node_list)
        high_lh_node_list_length = len(self.model.high_lh_node_list)

        low_percent = agent_familiarity_rate * (low_lh_node_list_length / nodes_num)
        middle_percent = agent_familiarity_rate * (middle_lh_node_list_length / nodes_num)
        high_percent = agent_familiarity_rate * (high_lh_node_list_length / nodes_num)

        low_num = int(low_percent * low_lh_node_list_length)
        middle_num = int(middle_percent * middle_lh_node_list_length)
        high_num = int(high_percent * high_lh_node_list_length)

        low_lh_know = random.sample(self.model.low_lh_node_list,
                                    k=min(low_num, len(self.model.low_lh_node_list)))
        middle_lh_know = random.sample(self.model.middle_lh_node_list,
                                       k=min(middle_num, len(self.model.middle_lh_node_list)))
        high_lh_know = random.sample(self.model.high_lh_node_list,
                                     k=min(high_num, len(self.model.high_lh_node_list)))
        global_lh_knowledge = low_lh_know + middle_lh_know + high_lh_know

        global_knowledge = global_bet_knowledge + list(set(global_lh_knowledge) - set(global_bet_knowledge))
        return global_knowledge

    def set_local_knowledge(self):
        random_home_node = random.sample(list(self.model.G.nodes()), 1)
        reachablenode_length_dic = nx.single_source_shortest_path_length(
            self.model.G, source=random_home_node[0], cutoff=15)
        return list(reachablenode_length_dic.keys())

    def initialize_knowledge(self, agent_familiarity_rate, origin_id):
        global_knowledge = self.set_global_knowledge(self.model)
        local_knowledge = self.set_local_knowledge(self)
        knowledge_result = list(global_knowledge.union(local_knowledge))
        self.knowledge = knowledge_result

    def move(self):
        idx = self.model.schedule.time
        next_node = self.model.route_defining_locations[idx]

        if self.model.G.has_edge(self.location, next_node):
            path = [self.location, next_node]
        else:
            if next_node in self.knowledge:
                # Strategy: shortest path
                self.state_familiarity = 1
                paths = k_shortest_paths(self.model.G, self.location, next_node, 'distance', 1)
                path = paths[0]
            else:
                # Strategy: random(th) shortest path with a limit
                self.state_familiarity = 0
                shortest_path_array = k_shortest_paths(self.model.G, self.location, next_node, 'distance', 1)
                shortest_path = shortest_path_array[0]
                shortest_path_length = len(shortest_path)
                rand = random.randint(0, 10)
                paths = k_shortest_paths(self.model.G, self.location, next_node, 'distance', rand)
                limit = 3 * shortest_path_length
                path = paths[0]
                for p in paths:
                    if len(p) > limit:
                        break
                    path = p

            for n in range(len(path)):
                if n != 0:
                    self.traversed_path.append(path[n])
            for n in range(len(path) - 1):
                edge = self.model.G.edges[path[n], path[n + 1]]
                self.traversed_path_distance += edge['distance']
                self.traversed_path_complexity += edge['complexity']

            self.location = next_node
            self.model.grid.move_agent(self, next_node)
            self.state_traversed_path = path

    def step(self):
        self.move()
