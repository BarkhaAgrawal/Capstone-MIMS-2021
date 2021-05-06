from flask import Flask, render_template
from flask import request
import matplotlib.pyplot as plt
import matplotlib.image as image
import shapely.wkt
import shapely.ops

app = Flask(__name__)
app.debug = True
app.config['TEMPLATES_AUTO_RELOAD'] = True


from basic_woodacre import *
# sys.path.append('/Users/barkha/Documents/UCB/Spring\ 2021/Capstone/Capstone/projects/woodacre_civic')

# import basic_woodacre
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulation-comparison')
def comparison():
    return render_template('comparison.html')


absolute_path = '/Users/barkha/Documents/UCB/Spring 2021/Capstone/Capstone/projects/woodacre_civic'
network_file_edges = '/network_inputs/osm_edges_woodacre.csv'
simulation_outputs = '/simulation_outputs/link_stats'
visualization_outputs = '/static/images/'
network_file_nodes = '/network_inputs/osm_nodes_woodacre.csv'

# Python code to get the Cumulative sum of a list
def Cumulative(lists):
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    print(cu_list[1:])
    return cu_list[1:]

def getimage(nodeid = 111):
    fig, ax = plt.subplots(figsize=(10, 10))
    absolute_path = '/Users/barkha/Documents/UCB/Spring 2021/Capstone/Capstone/projects/woodacre_civic'
    t_list = {}
    for t in range(0, 400, 2):
        n_stats = pd.read_csv(absolute_path + '/simulation_outputs/node_stats/node_agent_cnts_test_t{}.csv'.format(t))
        # print(n_stats['node_id'])
        #     print(n_stats[n_stats.node_id == 111]['cnt'])
        if len(list(n_stats[n_stats.node_id == nodeid]['cnt'])) > 0:
            t_list[t] = list(n_stats[n_stats.node_id == nodeid]['cnt'])[0]
        else:
            t_list[t] = 0
    # print(t_list)
    t = list(t_list.values())
    t1 = t.copy()
    t_list_total = Cumulative(t1)
    x = list(range(0, 400, 2))
    newList = [m / 60 for m in x]
    print(t1, t_list_total, newList)
    plt.plot(list(newList), np.array(t1), color='red', linestyle='--', label='arrived')
    plt.plot(list(newList), np.array(t_list_total), color='blue', linestyle='-', label='cummulative')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.xlim([0, 10])
    plt.ylim([0, 2000])
    plt.xlabel('minutes', fontsize=14)
    plt.ylabel('Vehicles arrived at Shelter',fontsize=14)
    plt.legend()
    #     plt.savefig(absolute_path + visualization_outputs + 'demand.png', transparent=True)
    # plt.show()
    plt.savefig(absolute_path + visualization_outputs + 'demand.png', transparent=True)
    # plt.show()
    plt.close()

def make_img(t=0, memory=None):
    ### Get edge veh counts
    links_nodes = pd.read_csv(absolute_path + network_file_nodes)
    links_df = pd.read_csv(absolute_path + network_file_edges)
    link_queue_duration = pd.read_csv(absolute_path + simulation_outputs + '/link_stats_test_t{}.csv'.format(t))
    link_queue_duration = pd.merge(link_queue_duration, links_df[['eid', 'geometry']], left_on='link_id',
                                   right_on='eid', how='left')
    road_gdf = gpd.GeoDataFrame(link_queue_duration,
                                crs={'init': 'epsg:4326'},
                                geometry=link_queue_duration['geometry'].map(shapely.wkt.loads))
    links_nodes['geometry'] = links_nodes['geometry'].apply(loads)
    gdf = gpd.GeoDataFrame(links_nodes, crs='epsg:4326')
    fig= plt.figure(figsize=(12, 10))
    ax1 = plt.gca()
    im = image.imread(absolute_path + visualization_outputs+'direction1.png')
    gdf.plot(ax=ax1)
    road_gdf['norm_r'] = (road_gdf['r'] - road_gdf['r'].min()) / (road_gdf['r'].max() - road_gdf['r'].min())
    road_plot = road_gdf.plot(ax=ax1, column='norm_r',legend=True, cmap=plt.get_cmap('RdYlGn_r',10),vmin=0, vmax=1,  legend_kwds={'shrink': 0.35,'label': '% of Road Capacity Filled'})
    # ax1.imshow(im, aspect='auto', extent=(0.4, 0.6, .5, .7), zorder=-1)
    # fig.figimage(im, 0, fig.bbox.ymax - height)

    fig.text(0.5, 0.85, '{} sec into evacuation'.format(t), fontsize=20, ha='center', va='center')
    plt.xlabel("Longitude")
    plt.ylabel('Latitude')
    plt.title("Road Network of Woodacre")
    newax = fig.add_axes([0.62, 0.25, 0.1, 0.1], anchor='SE', zorder=-1)
    newax.imshow(im)
    newax.axis('off')

    plt.savefig(absolute_path + visualization_outputs + 'veh_loc_t{}.png'.format(t), transparent=True)
    plt.close()
    # #cmap=plt.get_cmap('Blues',10),
    #          vmin=0, vmax=1,
    #          legend_kwds={'label': 'Coverage', 'ticks': np.arange(0,1.1, 0.2)}
    return memory


def make_gif(memory=None):
    import imageio
    images = []
    for t in range(0,400, 2):
        memory = make_img(t=t, memory=memory)
        images.append(imageio.imread(absolute_path + visualization_outputs + 'veh_loc_t{}.png'.format(t)))
    imageio.mimsave(absolute_path + visualization_outputs + '/veh_loc_animation.gif', images, fps=2)

@app.route('/simulate', methods=['POST'])
def simulate():
    print(request.form.getlist('incorporateSVI'))
    print(request.form.getlist('inputDirection'))
    # select = request.form.get('comp_select')
    if request.form.getlist("inputNoticeTime")[0] == '1':
        dept_time_id = 'imm'
    elif request.form.getlist("inputNoticeTime")[0] == '2':
        dept_time_id = 'fst'
    else:
        dept_time_id = 'mid'

    if request.form.getlist("inputDirection")[0] == '1':
        dir = 'north'
        node = 111
    elif request.form.getlist("inputDirection")[0] == '2':
        dir = 'south'
        node = 76
    elif request.form.getlist("inputDirection")[0] == '3':
        dir = 'east'
        node = 8
    else:
        dir = 'west'
        node = 95

    if len(request.form.getlist('incorporateSVI')) != 0:
        speed = 5
    else:
        speed = 20
    print(dir, speed, dept_time_id)
    network, check_traffic_flow_links, scen_nm, simulation_outputs = preparation(dept_time_col='dept_time_scen_1',dept_time_id = dept_time_id, fire_dir = dir, speed = speed )

    # for t in range(0, 400+1):
    #     network = one_step(t, network, check_traffic_flow_links, scen_nm, simulation_outputs)
    # make_gif(memory=None)
    getimage(node)
    return render_template('index.html', filename=visualization_outputs + '/veh_loc_animation.gif',demand=visualization_outputs + '/demand.png')