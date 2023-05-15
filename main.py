import networkx as nx
import random
from models import PathModel
from digraph import create_osm_digraph, assign_edge_betweenness_centrality_to_digraph,\
    assign_node_betweenness_centrality_to_digraph, generate_random_origin_destination_pairs

# ##osm digraph creation##
csv_file_name = 'extracted_hamburg_landmarks.csv'
osm = "extracted_hamburg.osm"
dg = create_osm_digraph(osm, csv_file_name)
dg = assign_edge_betweenness_centrality_to_digraph(dg, 'distance')
dg = assign_node_betweenness_centrality_to_digraph(dg, 'distance')
od_list = generate_random_origin_destination_pairs(dg, 1)
origin, destination = od_list[0]

print('origin and destination')
print(origin)
print(destination)

# ##defining some random route_defining locations from the origin to the destination##
shortest_path = nx.dijkstra_path(dg, origin, destination, weight='distance')
num_of_selections = int(0.55*(len(shortest_path)-1))
shortest_path_without_od = shortest_path[1:-1]
route_defining_locations = random.choices(shortest_path_without_od, k=num_of_selections)
route_defining_locations.append(shortest_path[-1])

# ##run agent_based model##  unique_id, model, origin_id, agent_familiarity_rate
path_model = PathModel(dg, 5, origin, destination, route_defining_locations, [0.5, 0.4, 0.3, 0.2, 0.1])
time_step = len(route_defining_locations)
path_model.run_model(time_step)  # each time step: one route-defining locations

agent_variables = path_model.datacollector.get_agent_vars_dataframe()
result = agent_variables.loc[time_step-1, 'traversed_path_length']
result_np = result.values[0]

# print some of the results
print(agent_variables)
print(agent_variables.loc[time_step-1])
print(result)
print(result_np)
print(result_np[2]/result_np[1])
