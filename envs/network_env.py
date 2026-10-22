import os
import pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm


class TransitNetwork:
    """
    Manages the static public transit network topology, OD demand matrix,
    and shortest-path caching mechanisms.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.G = nx.Graph()
        self.node_df = None
        self.od_df = None
        self.sp_len = {}
        self.sp = {}
        self.stop_nodes_set = set()
        self.total_demand = 0

        self._load_network()

    def _load_network(self):
        """Loads nodes, edges, OD matrix, and computes/loads shortest paths."""
        print(f"[NetworkEnv] Loading data from '{self.base_path}'...")

        # 1. Load basic CSV files
        try:
            edge_df = pd.read_csv(os.path.join(self.base_path, 'edge.csv'))
            self.node_df = pd.read_csv(os.path.join(self.base_path, 'node.csv'))
            raw_od_df = pd.read_csv(os.path.join(self.base_path, 'ODmatrix.csv'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required network files: {e}")

        # 2. Identify physical bus stops
        if 'type' not in self.node_df.columns:
            raise ValueError("'type' field is missing in node.csv.")
        self.stop_nodes_set = set(self.node_df[self.node_df['type'] == 'S']['nodeID'])

        # 3. Filter valid OD demand and calculate total demand
        self.od_df = raw_od_df[raw_od_df['PD'] > 0].copy()
        self.total_demand = self.od_df['PD'].sum()

        # 4. Build Graph
        for _, row in edge_df.iterrows():
            self.G.add_edge(row['FromNode'], row['ToNode'], weight=row['TT'])

        # 5. Process Shortest Paths (with caching for speed)
        self._handle_shortest_paths()

    def _handle_shortest_paths(self):
        """Calculates or loads cached Dijkstra shortest paths to avoid redundant computation."""
        sp_len_path = os.path.join(self.base_path, 'cached_sp_len.pkl')
        sp_path = os.path.join(self.base_path, 'cached_sp.pkl')

        if os.path.exists(sp_len_path) and os.path.exists(sp_path):
            print("[NetworkEnv] Loading cached shortest path files...")
            with open(sp_len_path, 'rb') as f:
                self.sp_len = pickle.load(f)
            with open(sp_path, 'rb') as f:
                self.sp = pickle.load(f)
        else:
            print("[NetworkEnv] Cache not found. Calculating all-pairs shortest paths...")
            od_nodes = set(self.od_df['OriginID']).union(set(self.od_df['DestinationID']))
            relevant_nodes = list(od_nodes.union(self.stop_nodes_set))

            for source in tqdm(relevant_nodes, desc="Calculating Shortest Paths"):
                if source in self.G:
                    lengths, paths = nx.single_source_dijkstra(self.G, source, weight='weight')
                    self.sp_len[source] = lengths
                    self.sp[source] = paths

            with open(sp_len_path, 'wb') as f:
                pickle.dump(self.sp_len, f)
            with open(sp_path, 'wb') as f:
                pickle.dump(self.sp, f)
            print("[NetworkEnv] Shortest path caching complete.")