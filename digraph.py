import xml.sax
import pyproj
import numpy as np
from itertools import islice
import itertools
import networkx as nx
from statistics import mean


class Node:
    def __init__(self, id, lon, lat, is_dp):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.tags = {}
        self.is_dp = is_dp


class Way:
    def __init__(self, id, osm):
        self.osm = osm
        self.id = id
        self.nds = []
        self.tags = {}


class OSM:
    def __init__(self, filename_or_stream):
        self.nodes = {}
        self.ways = {}

        class OSMHandler(xml.sax.ContentHandler):
            def __init__(self):
                self.curr_elem = None

            def startElement(self, name, attrs):
                if name == 'node':
                    self.curr_elem = Node(attrs['id'], float(attrs['lon']), float(attrs['lat']), False)
                elif name == 'way':
                    self.curr_elem = Way(attrs['id'], self)
                elif name == 'tag':
                    self.curr_elem.tags[attrs['k']] = attrs['v']
                elif name == 'nd':
                    self.curr_elem.nds.append(attrs['ref'])

            def endElement(self, name):
                if name == 'node':
                    self.nodes[self.curr_elem.id] = self.curr_elem
                elif name == 'way':
                    self.ways[self.curr_elem.id] = self.curr_elem

            def characters(self, chars):
                pass

        parser = xml.sax.make_parser()
        parser.setContentHandler(OSMHandler())
        parser.parse(filename_or_stream)

        dp_refs = set()
        for way in self.ways.values():
            if 'highway' in way.tags and way.tags['highway'] in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential']:
                for nd_ref in way.nds:
                    if nd_ref in self.nodes:
                        node = self.nodes[nd_ref]
                        if not node.is_dp:
                            node.is_dp = True
                        else:
                            dp_refs.add(nd_ref)
        print(f'Number of Decision Points: {len(dp_refs)}')

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_way(self, way_id):
        return self.ways[way_id]


def create_osm_digraph(filename: str, csv_file_name: str) -> nx.DiGraph:
    print('Starting digraph creation...')

    osm = OSM(filename)
    G = nx.DiGraph()

    for w in osm.ways.values():
        if 'highway' in w.tags:
            highway = w.tags['highway']
            if highway in {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential'}:
                w_dp = [n_id for n_id in w.nds if osm.nodes[n_id].isDP]
                if 'oneway' in w.tags and w.tags['oneway'] == 'yes':
                    nx.add_path(G, w_dp, id=w.id)
                else:
                    nx.add_path(G, w_dp, id=w.id)
                    nx.add_path(G, w_dp[::-1], id=w.id)

    print('Calculating node parameters...')
    for DP_id in G.nodes():
        DP = osm.nodes[DP_id]
        neighbor_one = list(G.neighbors(DP_id))
        neighbor_two = list(G.predecessors(DP_id))
        neighbor_list = list(set(neighbor_one) | set(neighbor_two))  # Union of two lists
        G.nodes[DP_id].update({
            'lat': DP.lat,
            'lon': DP.lon,
            'id': DP.id,
            'numOfBranches': len(neighbor_list),
        })
        deviation_list = []
        bearing_differences_list = []
        neighbor_distance_list = []
        for a, b in itertools.combinations(neighbor_list, 2):  # Every 2 combinations for avgDeviation
            node_a = osm.nodes[a]
            node_b = osm.nodes[b]
            pointA = [node_a.lat, node_a.lon]
            pointDP = [G.nodes[DP_id]['lat'], G.nodes[DP_id]['lon']]
            pointB = [node_b.lat, node_b.lon]
            fwd_azimuth_DPA, _, distance_DPA = get_azimuth(pointDP, pointA)
            neighbor_distance_list.append(distance_DPA)
            edge_data = {'bearing': fwd_azimuth_DPA, 'distance': distance_DPA}
            G.add_edge(DP_id, a, **edge_data) if G.has_edge(DP_id, a) else G.add_edge(a, DP_id, **edge_data)
            fwd_azimuth_DPB, _, distance_DPB = get_azimuth(pointDP, pointB)
            neighbor_distance_list.append(distance_DPB)
            edge_data = {'bearing': fwd_azimuth_DPB, 'distance': distance_DPB}
            G.add_edge(DP_id, b, **edge_data) if G.has_edge(DP_id, b) else G.add_edge(b, DP_id, **edge_data)
            bearing = abs(fwd_azimuth_DPA - fwd_azimuth_DPB)
            bearing_differences_list.append(bearing)
            deviation_list.append(calculate_deviation(bearing))
        if len(neighbor_list) == 1:
            a = neighbor_list[0]
            node_a = osm.nodes[a]
            pointA = [node_a.lat, node_a.lon]
            pointDP = [G.nodes[DP_id]['lat'], G.nodes[DP_id]['lon']]
            fwd_azimuth_DPA, back_azimuth_DPA, distance_DPA = get_azimuth(pointDP, pointA)
            neighbor_distance_list.append(distance_DPA)

            if G.has_edge(DP_id, a):
                G.edges[DP_id, a]['bearing'] = fwd_azimuth_DPA
                G.edges[DP_id, a]['distance'] = distance_DPA
            else:
                G.edges[a, DP_id]['bearing'] = fwd_azimuth_DPA
                G.edges[a, DP_id]['distance'] = distance_DPA

        if deviation_list:
            avg_deviation = mean(deviation_list)
        else:
            avg_deviation = 0
        G.nodes[DP_id]['avgDeviation'] = avg_deviation

        bearing_list = []
        for neighbor in neighbor_list:
            if G.has_edge(DP_id, neighbor):
                bearing_list.append(G.edges[DP_id, neighbor]['bearing'])
            else:
                bearing_list.append(G.edges[neighbor, DP_id]['bearing'])

        inst_complexity = calculate_instruction_complexity(bearing_differences_list, len(neighbor_list))
        inst_equivalent = calculate_instruction_equivalent(bearing_list)
        G.nodes[DP_id]['instComplexity'] = inst_complexity
        G.nodes[DP_id]['instEquivalent'] = inst_equivalent

    print('Calculating edge parameters...')

    way_type_list = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential']

    for node1, node2, data in G.edges(data=True):
        if 'distance' not in data:
            data['distance'] = G.edges[node2, node1]['distance']
            way_id = G.edges[node2, node1]['id']
            way = osm.ways[way_id]
            way_type = way.tags.get('highway')
            if way_type and way_type in way_type_list:
                data['way_type'] = way_type
                data['way_type_num'] = way_type_list.index(way_type) + 1

        if 'bearing' in data:
            way_id = data['id']
            way = osm.ways.get(way_id)
            way_type = way.tags.get('highway')
            if way_type and way_type in way_type_list:
                data['way_type'] = way_type
                data['way_type_num'] = way_type_list.index(way_type) + 1

    return G


def get_azimuth(pointA, pointB):
    lat1, long1 = pointA
    lat2, long2 = pointB
    geodesic = pyproj.Geod(ellps='WGS84')
    fwd_azimuth, _, _ = geodesic.inv(long1, lat1, long2, lat2)
    return fwd_azimuth


def assign_edge_betweenness_centrality_to_digraph(digraph, weight_string):
    edge_bet_centrality = nx.edge_betweenness_centrality(digraph, normalized=True, weight=weight_string)
    for (node1, node2), v in edge_bet_centrality.items():
        digraph.edges[node1, node2]['edge_bet_centrality'] = v
    return digraph


def assign_node_betweenness_centrality_to_digraph(digraph, weight):
    node_bet_centrality = nx.betweenness_centrality(digraph, normalized=True, weight=weight, endpoints=False)
    nx.set_node_attributes(digraph, node_bet_centrality, name='node_bet_centrality')
    return digraph


def generate_random_origin_destination_pairs(digraph, num_pairs):
    dp_list = list(digraph.nodes())
    od_list = []
    while len(od_list) < num_pairs:
        source, destination = np.random.choice(dp_list, 2, replace=False)
        if source != destination and not digraph.has_edge(source, destination) and nx.has_path(digraph, source, destination):
            od_list.append((source, destination))
    return od_list


def find_shortest_path(digraph, source, destination):
    return nx.dijkstra_path(digraph, source, destination, weight='distance')


def find_k_shortest_paths(digraph, source, target, weight, k):
    return list(islice(nx.shortest_simple_paths(digraph, source, target, weight=weight), k))


def add_complexity(digraph):
    distance_list = [data['distance'] for _, _, data in digraph.edges(data=True)]
    max_distance = max(distance_list)
    max_branch = max(digraph.nodes[node]['numOfBranches'] for node in digraph)
    max_deviation = max(digraph.nodes[node]['avgDeviation'] for node in digraph)
    max_instruction_equivalent = max(digraph.nodes[node]['instEquivalent'] for node in digraph)
    max_instruction_complexity = max(digraph.nodes[node]['instComplexity'] for node in digraph)
    max_landmark = max(digraph.nodes[node]['landmark_hierarchy'] for node in digraph)

    for node1, node2, data in digraph.edges(data=True):
        weight_sum = (1 - (data['distance'] / max_distance)) + \
                     (digraph.nodes[node1]['numOfBranches'] / max_branch) + \
                     (digraph.nodes[node1]['avgDeviation'] / max_deviation) + \
                     (digraph.nodes[node1]['instEquivalent'] / max_instruction_equivalent) + \
                     (digraph.nodes[node1]['instComplexity'] / max_instruction_complexity) + \
                     (1 - (digraph.nodes[node1]['landmark_hierarchy'] / max_landmark))

        digraph.edges[node1, node2]['complexity'] = weight_sum / 6  # Model parameters number is hardcoded

    return digraph


def calculate_deviation(bearing):
    a = abs((bearing % 90) - 45)
    return min(a, 90 - a)


def calculate_instruction_complexity(bearing_differences_list, num_of_branches):
    instruction_complexity_list = []
    if bearing_differences_list:
        for b in bearing_differences_list:
            if abs(b) <= 10:
                complexity = 1
            elif num_of_branches == 3:
                complexity = 6
            else:
                complexity = 5 + num_of_branches
            instruction_complexity_list.append(complexity)

        ins_complexity = mean(instruction_complexity_list)
        return ins_complexity
    else:
        return 0


def calculate_instruction_equivalent(bearing_list):
    if bearing_list:
        directions = [0, 90, 180, -90, -180]
        counts = [sum(abs(b - d) <= 45 for b in bearing_list) for d in directions]
        return max(counts)
    else:
        return 1


def add_complexity_to_digraph(digraph):
    distance_list = [data['distance'] for (node1, node2, data) in digraph.edges(data=True)]
    branch_list = [digraph.nodes[node1]['numOfBranches'] for (node1, node2, data) in digraph.edges(data=True)]
    deviation_list = [calculate_deviation(digraph.nodes[node1]['bearing']) for (node1, node2, data) in digraph.edges(data=True)]
    instruction_complexity_list = [calculate_instruction_complexity(digraph.nodes[node1]['bearing_differences'], digraph.nodes[node1]['numOfBranches']) for (node1, node2, data) in digraph.edges(data=True)]
    instruction_equivalent_list = [calculate_instruction_equivalent(digraph.nodes[node1]['bearing_list']) for (node1, node2, data) in digraph.edges(data=True)]
    landmark_list = [digraph.nodes[node1]['landmark_hierarchy'] for (node1, node2, data) in digraph.edges(data=True)]
    max_distance = max(distance_list)
    max_branch = max(branch_list)
    max_deviation = max(deviation_list)
    max_instruction_equivalent = max(instruction_equivalent_list)
    max_instruction_complexity = max(instruction_complexity_list)
    max_landmark = max(landmark_list)

    for (node1, node2, data) in digraph.edges(data=True):
        weight_sum = (1 - (data['distance'] / max_distance)) + \
                     (digraph.nodes[node1]['numOfBranches'] / max_branch) + \
                     (calculate_deviation(digraph.nodes[node1]['bearing']) / max_deviation) + \
                     (calculate_instruction_equivalent(digraph.nodes[node1]['bearing_list']) / max_instruction_equivalent) + \
                     (calculate_instruction_complexity(digraph.nodes[node1]['bearing_differences'], digraph.nodes[node1]['numOfBranches']) / max_instruction_complexity) + \
                     (1 - (digraph.nodes[node1]['landmark_hierarchy'] / max_landmark))

        digraph.edges[node1, node2]['complexity'] = weight_sum / 6

    return digraph


def cluster_nodes(digraph):
    bet_cent_list = []
    lh_list = []

    low_bet_cent_nodes = []
    mid_bet_cent_nodes = []
    high_bet_cent_nodes = []
    low_lh_nodes = []
    mid_lh_nodes = []
    high_lh_nodes = []

    for node in digraph.nodes():
        bet_value = digraph.nodes[node]['node_bet_centrality']
        bet_cent_list.append(bet_value)
        lh_value = digraph.nodes[node]['landmark_hierarchy']
        lh_list.append(lh_value)

    max_bet_cent = max(bet_cent_list)
    min_bet_thr = 0.3 * max_bet_cent
    mid_bet_thr = 0.7 * max_bet_cent
    max_lh = max(lh_list)
    min_lh_thr = 0.3 * max_lh
    mid_lh_thr = 0.7 * max_lh

    for node in digraph.nodes():
        bet_value = digraph.nodes[node]['node_bet_centrality']
        if 0 <= bet_value <= min_bet_thr:
            low_bet_cent_nodes.append(node)
        elif min_bet_thr < bet_value <= mid_bet_thr:
            mid_bet_cent_nodes.append(node)
        else:
            high_bet_cent_nodes.append(node)

        lh_value = digraph.nodes[node]['landmark_hierarchy']
        if 0 <= lh_value <= min_lh_thr:
            low_lh_nodes.append(node)
        elif min_lh_thr < lh_value <= mid_lh_thr:
            mid_lh_nodes.append(node)
        else:
            high_lh_nodes.append(node)

    return low_bet_cent_nodes, mid_bet_cent_nodes, high_bet_cent_nodes, low_lh_nodes, mid_lh_nodes, high_lh_nodes


def k_shortest_paths(digraph, source, target, weight, k):
    return list(islice(nx.shortest_simple_paths(digraph, source, target, weight), k))
