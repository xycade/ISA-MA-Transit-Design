import math
import copy
import collections


class PassengerSimulator:
    """
    Evaluates fitness, passenger routing, transfer penalties, and
    calculates demand-responsive frequencies for the transit network.
    """

    def __init__(self, network_env, config: dict):
        self.net = network_env
        self.config = config

        # Extract constants from config for fast access
        self.max_transfers = config.get('MAX_TRANSFERS', 2)
        self.transfer_penalty = config.get('TRANSFER_PENALTY_TIME', 5)
        self.alpha_unmet = config.get('ALPHA_UNMET_DEMAND', 10000)
        self.bus_capacity = config.get('BUS_CAPACITY', 50)
        self.min_freq = config.get('MIN_FREQUENCY', 2)
        self.max_freq = config.get('MAX_FREQUENCY', 99)

    def evaluate_fitness(self, individual: dict, fitness_cache: dict) -> float:
        routes = individual.get('routes', [])
        if not routes:
            return self.net.total_demand * self.alpha_unmet

        canonical_routes = sorted([",".join(map(str, r)) for r in routes if r])
        cache_key = ";".join(canonical_routes)

        if cache_key in fitness_cache:
            return fitness_cache[cache_key]

        ind_copy = copy.deepcopy(individual)
        ind_copy['frequencies'] = self.calculate_demand_responsive_frequencies(ind_copy)

        precomputed_data = self._prepare_network_data(ind_copy)
        metrics = self.calculate_performance_metrics(precomputed_data)

        fitness_cache[cache_key] = metrics["fitness"]
        return metrics["fitness"]

    def _prepare_network_data(self, individual: dict) -> dict:
        routes = individual.get('routes', [])
        frequencies = individual.get('frequencies', [])
        if not routes or not frequencies or len(routes) != len(frequencies):
            return None

        num_routes = len(routes)
        route_sets = [set(r) for r in routes]
        route_node_positions = [{node: pos for pos, node in enumerate(r)} for r in routes]

        route_prefix_times = []
        for r in routes:
            prefix_times = [0] * len(r)
            for j in range(len(r) - 1):
                travel_time = self.net.sp_len.get(r[j], {}).get(r[j + 1], float('inf'))
                prefix_times[j + 1] = prefix_times[j] + travel_time
            route_prefix_times.append(prefix_times)

        node_to_routes_map = collections.defaultdict(list)
        for i, r_set in enumerate(route_sets):
            for node in r_set:
                if node in self.net.stop_nodes_set:
                    node_to_routes_map[node].append(i)

        transfer_hubs = {}
        one_hop_routes = {i: set() for i in range(num_routes)}
        for i in range(num_routes):
            for j in range(i + 1, num_routes):
                shared_stops = route_sets[i].intersection(route_sets[j]).intersection(self.net.stop_nodes_set)
                if shared_stops:
                    transfer_hubs[(i, j)] = transfer_hubs[(j, i)] = shared_stops
                    one_hop_routes[i].add(j)
                    one_hop_routes[j].add(i)

        wait_times = [30 / f if f > 0 else float('inf') for f in frequencies]

        return {
            "num_routes": num_routes,
            "route_sets": route_sets,
            "route_node_positions": route_node_positions,
            "route_prefix_times": route_prefix_times,
            "node_to_routes_map": node_to_routes_map,
            "transfer_hubs": transfer_hubs,
            "one_hop_routes": one_hop_routes,
            "wait_times": wait_times
        }

    def _get_route_travel_time(self, route: list) -> float:
        """Helper to calculate the total topological travel time of a route"""
        if not route: return 0.0 # [Fixed] Empty routes no longer consume time or vehicles.
        travel_time = 0
        for i in range(len(route) - 1):
            travel_time += self.net.sp_len.get(route[i], {}).get(route[i + 1], float('inf'))
        return travel_time

    def _calculate_best_two_transfer_path(self, origin, dest, r1, r_mid, r3, transfer_hubs, route_prefix_times,
                                          route_node_positions):
        min_ivt1, best_t1_node = float('inf'), None
        t1_nodes = transfer_hubs.get((r1, r_mid), [])
        if not t1_nodes: return float('inf')
        origin_pos_r1 = route_node_positions[r1].get(origin)
        if origin_pos_r1 is None: return float('inf')
        for t1 in t1_nodes:
            t1_pos_r1 = route_node_positions[r1].get(t1)
            if t1_pos_r1 is not None:
                ivt = abs(route_prefix_times[r1][t1_pos_r1] - route_prefix_times[r1][origin_pos_r1])
                if ivt < min_ivt1: min_ivt1, best_t1_node = ivt, t1

        if best_t1_node is None: return float('inf')

        min_ivt3, best_t2_node = float('inf'), None
        t2_nodes = transfer_hubs.get((r_mid, r3), [])
        if not t2_nodes: return float('inf')
        dest_pos_r3 = route_node_positions[r3].get(dest)
        if dest_pos_r3 is None: return float('inf')
        for t2 in t2_nodes:
            t2_pos_r3 = route_node_positions[r3].get(t2)
            if t2_pos_r3 is not None:
                ivt = abs(route_prefix_times[r3][dest_pos_r3] - route_prefix_times[r3][t2_pos_r3])
                if ivt < min_ivt3: min_ivt3, best_t2_node = ivt, t2

        if best_t2_node is None: return float('inf')

        t1_pos_rmid = route_node_positions[r_mid].get(best_t1_node)
        t2_pos_rmid = route_node_positions[r_mid].get(best_t2_node)
        if t1_pos_rmid is None or t2_pos_rmid is None: return float('inf')

        min_ivt2 = abs(route_prefix_times[r_mid][t2_pos_rmid] - route_prefix_times[r_mid][t1_pos_rmid])
        return min_ivt1 + min_ivt2 + min_ivt3

    def calculate_performance_metrics(self, precomputed_data: dict) -> dict:
        if not precomputed_data:
            return {
                "fitness": self.net.total_demand * self.alpha_unmet,
                "unmet_passengers": self.net.total_demand,
                "total_passenger_time_cost": 0, "passengers_0_transfer": 0,
                "passengers_1_transfer": 0, "passengers_2_transfer": 0,
                "total_in_vehicle_time": 0, "total_wait_time": 0, "total_transfer_time": 0
            }

        route_node_positions = precomputed_data["route_node_positions"]
        route_prefix_times = precomputed_data["route_prefix_times"]
        node_to_routes_map = precomputed_data["node_to_routes_map"]
        one_hop_routes = precomputed_data["one_hop_routes"]
        transfer_hubs = precomputed_data["transfer_hubs"]
        wait_times = precomputed_data["wait_times"]

        total_passenger_time_cost, unmet_demand = 0, 0
        total_wait_t, total_transfer_t, total_in_vehicle_t = 0, 0, 0
        p0, p1, p2 = 0, 0, 0

        for od_row in self.net.od_df.itertuples(index=False):
            origin, dest, demand = od_row.OriginID, od_row.DestinationID, od_row.PD
            if origin == dest: continue

            best_path = {'cost': float('inf'), 'transfers': -1, 'ivt': 0, 'wait': 0, 'transfer_p': 0}
            origin_routes = node_to_routes_map.get(origin, [])
            dest_routes_set = set(node_to_routes_map.get(dest, []))

            # 0-Transfer
            for i in set(origin_routes).intersection(dest_routes_set):
                pos1 = route_node_positions[i].get(origin)
                pos2 = route_node_positions[i].get(dest)
                if pos1 is not None and pos2 is not None:
                    ivt = abs(route_prefix_times[i][pos2] - route_prefix_times[i][pos1])
                    cost = ivt + wait_times[i]
                    if cost < best_path['cost']:
                        best_path.update(
                            {'cost': cost, 'transfers': 0, 'ivt': ivt, 'wait': wait_times[i], 'transfer_p': 0})

            # 1-Transfer
            if self.max_transfers >= 1:
                for r1 in origin_routes:
                    for r2 in one_hop_routes[r1].intersection(dest_routes_set):
                        min_ivt = float('inf')
                        for t_node in transfer_hubs.get((r1, r2), []):
                            pos1_o, pos1_t = route_node_positions[r1].get(origin), route_node_positions[r1].get(t_node)
                            pos2_t, pos2_d = route_node_positions[r2].get(t_node), route_node_positions[r2].get(dest)
                            if all(p is not None for p in [pos1_o, pos1_t, pos2_t, pos2_d]):
                                ivt_temp = abs(route_prefix_times[r1][pos1_t] - route_prefix_times[r1][pos1_o]) + \
                                           abs(route_prefix_times[r2][pos2_d] - route_prefix_times[r2][pos2_t])
                                min_ivt = min(min_ivt, ivt_temp)
                        if min_ivt != float('inf'):
                            cost = min_ivt + wait_times[r1] + wait_times[r2] + self.transfer_penalty
                            if cost < best_path['cost']:
                                best_path.update({'cost': cost, 'transfers': 1, 'ivt': min_ivt,
                                                  'wait': wait_times[r1] + wait_times[r2],
                                                  'transfer_p': self.transfer_penalty})

            # 2-Transfers
            if self.max_transfers >= 2 and best_path['transfers'] != 0:
                for r1 in origin_routes:
                    for r_mid in one_hop_routes[r1]:
                        for r3 in one_hop_routes[r_mid].intersection(dest_routes_set):
                            if r3 == r1 or r3 == r_mid: continue
                            total_ivt = self._calculate_best_two_transfer_path(origin, dest, r1, r_mid, r3,
                                                                               transfer_hubs, route_prefix_times,
                                                                               route_node_positions)
                            if total_ivt != float('inf'):
                                wait = wait_times[r1] + wait_times[r_mid] + wait_times[r3]
                                cost = total_ivt + wait + 2 * self.transfer_penalty
                                if cost < best_path['cost']:
                                    best_path.update({'cost': cost, 'transfers': 2, 'ivt': total_ivt, 'wait': wait,
                                                      'transfer_p': 2 * self.transfer_penalty})

            # Tally metrics
            if best_path['transfers'] != -1:
                total_passenger_time_cost += best_path['cost'] * demand
                total_wait_t += best_path['wait'] * demand
                total_transfer_t += best_path['transfer_p'] * demand
                total_in_vehicle_t += best_path['ivt'] * demand
                if best_path['transfers'] == 0:
                    p0 += demand
                elif best_path['transfers'] == 1:
                    p1 += demand
                else:
                    p2 += demand
            else:
                unmet_demand += demand

        fitness = total_passenger_time_cost + unmet_demand * self.alpha_unmet

        return {
            "fitness": fitness,
            "total_passenger_time_cost": total_passenger_time_cost,
            "total_wait_time": total_wait_t,
            "total_transfer_time": total_transfer_t,
            "total_in_vehicle_time": total_in_vehicle_t,
            "unmet_passengers": unmet_demand,
            "passengers_0_transfer": p0,
            "passengers_1_transfer": p1,
            "passengers_2_transfer": p2
        }

    def _calculate_passenger_loads_on_routes(self, individual: dict, precomputed_data: dict) -> tuple:
        num_routes = precomputed_data['num_routes']
        route_node_positions = precomputed_data["route_node_positions"]
        route_prefix_times = precomputed_data["route_prefix_times"]
        node_to_routes_map = precomputed_data["node_to_routes_map"]
        one_hop_routes = precomputed_data["one_hop_routes"]
        transfer_hubs = precomputed_data["transfer_hubs"]
        wait_times = precomputed_data["wait_times"]

        route_segment_loads = {i: collections.defaultdict(int) for i in range(num_routes)}
        unmet_od_pairs, served_od_info = [], []

        for od_row in self.net.od_df.itertuples(index=False):
            origin, dest, demand = od_row.OriginID, od_row.DestinationID, od_row.PD
            if origin == dest: continue

            best_path = {'cost': float('inf'), 'details': None}
            origin_routes = node_to_routes_map.get(origin, [])
            dest_routes_set = set(node_to_routes_map.get(dest, []))

            for i in set(origin_routes).intersection(dest_routes_set):
                pos1, pos2 = route_node_positions[i].get(origin), route_node_positions[i].get(dest)
                if pos1 is not None and pos2 is not None:
                    ivt = abs(route_prefix_times[i][pos2] - route_prefix_times[i][pos1])
                    cost = ivt + wait_times[i]
                    if cost < best_path['cost']:
                        best_path = {'cost': cost, 'details': ('0_transfer', i, origin, dest)}

            if self.max_transfers >= 1:
                for r1 in origin_routes:
                    for r2 in one_hop_routes[r1].intersection(dest_routes_set):
                        best_transfer_node, min_ivt = None, float('inf')
                        for t_node in transfer_hubs.get((r1, r2), []):
                            pos1_o, pos1_t = route_node_positions[r1].get(origin), route_node_positions[r1].get(t_node)
                            pos2_t, pos2_d = route_node_positions[r2].get(t_node), route_node_positions[r2].get(dest)
                            if all(p is not None for p in [pos1_o, pos1_t, pos2_t, pos2_d]):
                                ivt = abs(route_prefix_times[r1][pos1_t] - route_prefix_times[r1][pos1_o]) + \
                                      abs(route_prefix_times[r2][pos2_d] - route_prefix_times[r2][pos2_t])
                                if ivt < min_ivt: min_ivt, best_transfer_node = ivt, t_node

                        if best_transfer_node:
                            cost = min_ivt + wait_times[r1] + wait_times[r2] + self.transfer_penalty
                            if cost < best_path['cost']:
                                best_path = {'cost': cost,
                                             'details': ('1_transfer', r1, r2, origin, dest, best_transfer_node)}

            if best_path['details']:
                path_type, details = best_path['details'][0], best_path['details'][1:]
                served_od_info.append({'O': origin, 'D': dest, 'demand': demand, 'cost': best_path['cost']})
                if path_type == '0_transfer':
                    r_idx, o, d = details
                    route_segment_loads[r_idx][(o, d)] += demand
                elif path_type == '1_transfer':
                    r1, r2, o, d, t_node = details
                    route_segment_loads[r1][(o, t_node)] += demand
                    route_segment_loads[r2][(t_node, d)] += demand
            else:
                unmet_od_pairs.append({'O': origin, 'D': dest, 'demand': demand})

        return route_segment_loads, unmet_od_pairs, served_od_info

    def calculate_demand_responsive_frequencies(self, individual: dict) -> list:
        num_routes = len(individual.get('routes', []))
        if num_routes == 0:
            return []

        # Step 1: Trial Assignment
        avg_freq = (self.min_freq + self.max_freq) / 2
        temp_individual = copy.deepcopy(individual)
        temp_individual['frequencies'] = [avg_freq] * num_routes

        precomputed_data = self._prepare_network_data(temp_individual)
        if not precomputed_data:
            return [self.min_freq] * num_routes

        route_segment_loads, _, _ = self._calculate_passenger_loads_on_routes(temp_individual, precomputed_data)

        # Step 2: Extract Max Load for each route
        max_loads = [0] * num_routes
        for r_idx in range(num_routes):
            route = individual['routes'][r_idx]
            if not route: continue

            pos_map = {node: pos for pos, node in enumerate(route)}
            load_deltas = collections.defaultdict(int)

            for (o, d), demand in route_segment_loads[r_idx].items():
                pos_o, pos_d = pos_map.get(o), pos_map.get(d)
                if pos_o is not None and pos_d is not None:
                    start, end = min(pos_o, pos_d), max(pos_o, pos_d)
                    if start < end:
                        load_deltas[start] += demand
                        load_deltas[end] -= demand

            current_load = 0
            for i in range(len(route) - 1):
                current_load += load_deltas.get(i, 0)
                max_loads[r_idx] = max(max_loads[r_idx], current_load)

        # Step 3: Capacity-based calculation
        if self.bus_capacity > 0:
            frequencies = [math.ceil(load / self.bus_capacity) for load in max_loads]
        else:
            frequencies = [self.max_freq] * num_routes

        frequencies = [max(self.min_freq, min(f, self.max_freq)) for f in frequencies]

        # Step 4: [NEW] Fleet Size Constraint Repair
        max_fleet = self.config.get('MAX_TOTAL_FLEET', 999)
        while True:
            round_trip_times = [2 * self._get_route_travel_time(r) for r in individual['routes']]

            # [修复] 单独计算所需车队，忽略无效路线，绝不强行 break 整个惩罚循环
            fleet_per_route = []
            for rt, fq in zip(round_trip_times, frequencies):
                if math.isinf(rt) or rt == 0:
                    fleet_per_route.append(0)
                else:
                    fleet_per_route.append(math.ceil((rt * fq) / 60))

            if sum(fleet_per_route) <= max_fleet:
                break  # Fleet constraint satisfied!

            efficiencies = {}
            for i, r in enumerate(individual['routes']):
                if frequencies[i] > self.min_freq and fleet_per_route[i] > 0:
                    actual_stops = len([n for n in r if n in self.net.stop_nodes_set])
                    efficiencies[i] = actual_stops / fleet_per_route[i]

            if not efficiencies:
                break  # Cannot reduce any further (all routes hit min_freq)

            route_idx_to_reduce = min(efficiencies, key=efficiencies.get)
            frequencies[route_idx_to_reduce] -= 1

        return frequencies