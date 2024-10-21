# import community
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pylab as pylab



params = {'font.family': 'sans-serif', 'font.serif': 'Times'}
pylab.rcParams.update(params)

def viz(membership_path, embedding_path):
    """
    Visualizing graph given embeddings learned
    :param membership_path: labels of each node, each line is one label of one node
    :param embedding_path: embedding file of each node, each line is embedding of one nodes
    :return:
    """
    G = nx.karate_club_graph()
    membership = {}
    with open(membership_path, 'r') as member:
        for idx, line in enumerate(member):
            membership[idx] = int(line.strip())

    # https://networkx.github.io/documentation/development/_modules/networkx/drawing/layout.html
    pos = nx.fruchterman_reingold_layout(G)
    #pos = nx.spring_layout(G)

    # http://matplotlib.org/mpl_examples/color/named_colors.pdf
    colors = ['orange', 'orangered', 'darkturquoise', 'goldenrod', 'dodgerblue']


    ## Draw original network
    # fig, ax = plt.subplots()
    # fig.patch.set_visible(False)
    # ax.axis('off')

    # count = 0
    # for com in set(membership.values()):
    #     count += 1
    #     list_nodes = [nodes for nodes in membership.keys() if membership[nodes] == com]
    #     nx.draw_networkx_nodes(G, pos, list_nodes, node_size=200, linewidths=0, node_color=colors[count])
    # nx.draw_networkx_labels(G, pos, font_size=9, font_color='k')#, font=font)
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.show()

    fig, ax = plt.subplots()
    ax.get_yaxis().set_tick_params(which='major', direction='out')
    ax.get_xaxis().set_tick_params(which='major', direction='out')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    embedding = {}
    with open(embedding_path, 'r') as member:
        # member.readline()
        for line in member:
            res = line.strip().split()
            embedding[int(res[0])] = [float(res[1]), float(res[2])]
    count = 0
    for com in set(membership.values()):
        count += 1
        list_nodes = [nodes for nodes in membership.keys() if membership[nodes] == com]
        nx.draw_networkx_nodes(G, embedding, list_nodes, node_size=200, linewidths=0., node_color=colors[count])
    nx.draw_networkx_labels(G, embedding, font_size=9, font_color='k')#, font=font)
    plt.show()


    # pp = PdfPages('./fig-embed.pdf')
    # plt.savefig(pp, format='pdf', bbox_inches='tight', pad_inches=0)
    # pp.close()

if __name__ == "__main__":
    membership_path = "../data/membership.txt"
    embedding_path = "./output/example/embed_afterLSTM_50.txt"
    viz(membership_path, embedding_path)
