#!/usr/bin/env python
# coding: utf-8
import os
import gc
import sys
import time 
import random
import logging 
import warnings
import numpy as np
import pandas as pd 
import rasterio as rio
import scipy.io as sio
import geopandas as gpd
from ctypes import c_double
from shapely.wkt import loads
from shapely.geometry import Point
import scipy.spatial.distance
from scipy.stats import truncnorm
### user
import haversine as haversine
from queue_class import Network, Node, Link, Agent

random.seed(0)
np.random.seed(0)

# warnings.filterwarnings("error")

def link_model(t, network):

    for link_id, link in network.links.items(): 
        link.run_link_model(t, agent_id_dict=network.agents)
#         print(link.link_type)
        if link.link_type == 'v': ### do not track the link time of virtual links
            link.travel_time_list = []
        else:
            link.travel_time_list = []
        # link.update_travel_time(self, t_now, link_time_lookback_freq=None, g=None, update_graph=False)
    return network

def node_model(t, network, move, check_traffic_flow_links_dict):
    node_ids_to_run = set([link.end_nid for link in network.links.values() if len(link.queue_vehicles)>0])
    metrics = 0
    for node_id in node_ids_to_run:
        node = network.nodes[node_id] 
        n_t_move, transfer_links, agent_update_dict, link_update_dict = node.run_node_model(t, node_id_dict=network.nodes, link_id_dict=network.links, agent_id_dict=network.agents, node2link_dict=network.node2link_dict)
        move += n_t_move
        ### how many people moves across a specified link pair in this step
        for transfer_link in transfer_links:
            if transfer_link in check_traffic_flow_links_dict.keys():
                check_traffic_flow_links_dict[transfer_link] += 1
        for agent_id, agent_new_info in agent_update_dict.items():
            if agent_new_info[0] == 'shelter_p': metrics += 1
            ### move stopped vehicles to a separate dictionary
            if agent_new_info[0] in ['shelter_arrive', 'arrive']:
                network.agents_stopped[agent_id] = agent_new_info
                del network.agents[agent_id]
            ### update vehicles still not stopped
            else:
                agent = network.agents[agent_id]
                [agent.status, agent.current_link_start_nid, agent.current_link_end_nid,  agent.current_link_enter_time] = agent_new_info
                agent.find_next_link(t, node2link_dict = network.node2link_dict)

        for link_id, link_new_info in link_update_dict.items():
            if link_new_info[0] == 'inflow':
                [_, network.links[link_id].queue_vehicles, 
                network.links[link_id].remaining_outflow_capacity, 
                network.links[link_id].travel_time_list] = link_new_info
            elif link_new_info[0] == 'outflow':
                [_, network.links[link_id].run_vehicles, 
                network.links[link_id].remaining_inflow_capacity,
                network.links[link_id].remaining_storage_capacity] = link_new_info
            else:
                print('invalid link update information')
        
    return network, move, check_traffic_flow_links_dict, metrics

def one_step(t, network, check_traffic_flow_links, scen_nm, simulation_outputs):
    
    move = 0
    ### agent model
    t_agent_0 = time.time()
    for agent_id, agent in network.agents.items():
        # initial route 
        if t==0: routing_status = agent.get_path(g=network.g)
        agent.load_vehicle(t, node2link_dict=network.node2link_dict, link_id_dict=network.links)
        # reroute upon closure
        if (agent.next_link is not None) and (network.links[agent.next_link].status=='closed'):
            routing_status = agent.get_path(g=network.g)
            agent.find_next_link(t, node2link_dict=network.node2link_dict)
    t_agent_1 = time.time()

    ### link model
    ### Each iteration in the link model is not time-consuming. So just keep using one process.
    t_link_0 = time.time()
    network = link_model(t, network)
    t_link_1 = time.time()
    
    ### node model
    t_node_0 = time.time()
    check_traffic_flow_links_dict = {link_pair: 0 for link_pair in check_traffic_flow_links}
    network, move, check_traffic_flow_links_dict, metrics = node_model(t, network, move, check_traffic_flow_links_dict)
    t_node_1 = time.time()

    ### metrics
    if t%1 == 0:
        arrival_cnts = len([agent_id for agent_id, agent_info in network.agents_stopped.items() if agent_info[0]=='arrive'])
        shelter_cnts = len([agent_id for agent_id, agent_info in network.agents_stopped.items() if agent_info[0]=='shelter_arrive'])
        if len(network.agents)==0:
            logging.info("all agents arrive at destinations")
            return 0
        # vehicle locations
        veh_loc = [network.links[network.node2link_dict[(agent.current_link_start_nid, agent.current_link_end_nid)]].midpoint for agent in network.agents.values()]
        ### arrival
        with open(simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm),'a') as t_stats_outfile:
            t_stats_outfile.write(",".join([str(x) for x in [t, arrival_cnts, shelter_cnts, move, metrics]]) + "\n")
        ### transfer
        with open(simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(scen_nm), 'a') as transfer_stats_outfile:
            transfer_stats_outfile.write("{},".format(t) + ",".join([str(check_traffic_flow_links_dict[(il, ol)]) for (il, ol) in check_traffic_flow_links])+"\n")

    ### stepped outputs
    if t%1==0:
        link_output = pd.DataFrame(
            [(link.link_id, len(link.queue_vehicles), len(link.run_vehicles), round(link.travel_time, 2)) for link in network.links.values() if not link.link_type=='v'], columns=['link_id', 'q', 'r', 't'])
        link_output.to_csv(simulation_outputs + '/link_stats/link_stats_{}_t{}.csv'.format(scen_nm, t), index=False)
        node_agent_cnts = pd.DataFrame(
            [(agent.current_link_end_nid, agent.status, 1) for agent in network.agents.values()], columns=['node_id', 'status', 'cnt']).groupby(['node_id', 'status']).agg({'cnt': np.sum}).reset_index()
        node_agent_cnts.to_csv(simulation_outputs + '/node_stats/node_agent_cnts_{}_t{}.csv'.format(scen_nm, t), index=False)

    if t%200==0: 
        logging.info(" ".join([str(i) for i in [t, arrival_cnts, shelter_cnts, move, '|', round(t_agent_1-t_agent_0, 2), round(t_node_1-t_node_0, 2), round(t_link_1-t_link_0, 2)]]) + " " + str(len(veh_loc)))
    return network

def demand(dept_time=[0,0,0,1000], demand_files=None, fire_dir = 'north', speed = 25, edge_file = None):
    logger = logging.getLogger("bk_evac")
    edges = pd.read_csv(edge_file)
    edges['maxmph'] = speed
    direction = {'south': 8398747289,
    'north': 2006817311,
    'east': 2882876550,
    'west': 110377352 }
    direction_opp ={'north':'south', 'south': 'north', 'east': 'west', 'west': 'east'}

    all_od_list = []
    [dept_time_mean, dept_time_std, dept_time_min, dept_time_max] = dept_time
    # demand_files = ['/projects/berkeley_trb/demand_inputs/od_test.csv']
#     for demand_file in demand_files:
    od = pd.read_csv(demand_files)
#         ### transform OSM based id to graph node id
#         od['origin_nid'] = od['o_osmid'].apply(lambda x: nodes_osmid_dict[x])
#         od['destin_nid'] = od['d_osmid'].apply(lambda x: nodes_osmid_dict[x])
    ### assign agent id
#         if 'agent_id' not in od.columns: od['agent_id'] = np.arange(od.shape[0])
    ### assign departure time. dept_time_std == 0 --> everyone leaves at the same time
    if (dept_time_std==0): od['dept_time'] = dept_time_mean
    else:
        truncnorm_a, truncnorm_b = (dept_time_min-dept_time_mean)/dept_time_std, (dept_time_max-dept_time_mean)/dept_time_std
        od['dept_time'] = truncnorm.rvs(truncnorm_a, truncnorm_b, loc=dept_time_mean, scale=dept_time_std, size=od.shape[0])
        od['dept_time'] = od['dept_time'].astype(int)
        # od.to_csv(scratch_dir + '/od.csv', index=False)
        # sys.exit(0)
    od['destin_osmid'] = int(direction[direction_opp[fire_dir]])

#     if phase_tdiff is not None:
#         od['dept_time'] += (od['evac_zone']-1)*phase_tdiff*60 ### t_diff in minutes
    edges.to_csv(edge_file, index=False)
    od.to_csv(demand_files, index=False)

def preparation(dept_time_col=None,dept_time_id = 'imm', fire_dir = 'south', speed = 10):
    ### logging and global variables

    reroute_freq = 10 ### sec
    link_time_lookback_freq = 20 ### sec
    ### dir

    ### logging and global variables

    dept_time_dict = {'imm': [0, 0, 0, 1000], 'fst': [20 * 60, 10 * 60, 10 * 60, 30 * 60],
                      'mid': [40 * 60, 20 * 60, 20 * 60, 60 * 60], 'slw': [60 * 60, 30 * 60, 30 * 60, 90 * 60]}
    dept_time = dept_time_dict[dept_time_id]


    work_dir = os.environ['HOME']+'/Documents/UCB/Spring 2021/Capstone/Capstone/projects/woodacre_civic'
    network_file_edges = work_dir + '/network_inputs/osm_edges_woodacre.csv'
    network_file_nodes = work_dir + '/network_inputs/osm_nodes_woodacre.csv'
    demand_files = [work_dir + "/demand_inputs/od_csv/od.csv"]
    simulation_outputs = work_dir + '/../woodacre_civic/simulation_outputs' ### scratch_folder

    demand(dept_time= dept_time, demand_files=demand_files[0], fire_dir=fire_dir, speed=speed, edge_file = network_file_edges)
    scen_nm = "test"
    logging.basicConfig(filename=simulation_outputs+'/log/{}.log'.format(scen_nm), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info(scen_nm)
    print('log file created')

    ### network
    network = Network()
    network.dataframe_to_network(network_file_edges = network_file_edges, network_file_nodes = network_file_nodes)
    network.add_connectivity()

    ### demand
    network.add_demand(demand_files = demand_files)
    logging.info('total numbers of agents taken {}'.format(len(network.agents.keys())))
    
    ### time step output
    with open(simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm), 'w') as t_stats_outfile:
        t_stats_outfile.write(",".join(['t', 'arr', 'shelter', 'move','shelter_p'])+"\n")
    ### track the traffic flow from the following link pairs
    check_traffic_flow_links = [(29,33)]
#     with open(simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(
#         scen_nm), 'w') as transfer_stats_outfile:
#         transfer_stats_outfile.write("t,"+",".join(['{}-{}'.format(il, ol) for (il, ol) in check_traffic_flow_links])+"\n")

    return network, check_traffic_flow_links, scen_nm, simulation_outputs

# if __name__ == "__main__":
#     main(dept_time_col='dept_time_scen_1')