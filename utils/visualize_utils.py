
"""_utils functions for visualization purposes_
"""
import numpy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import animation
import matplotlib.colors as colors


def make_gif_for_multiple_plots(itrs):
    # Create new figure for GIF
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Adjust figure so GIF does not have extra whitespace
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0)
    ax.axis('off')
    ax2.axis('off')
    ims = []
    
    for itr in itrs:
        im = ax.imshow(plt.imread(
            f'plots/doe_at_itr_{itr}.png'), animated=True)
        im2 = ax2.imshow(plt.imread(f'plots/hologram_at_itr_{itr}.png'), animated=True)
        ims.append([im, im2])
    ani = animation.ArtistAnimation(fig, ims, interval=500)
    ani.save('plots/optimization_evolution.gif')
    print('save evolution of DOE patter and hologram images')


def plot_bar3d(data, title='3D bar plot', save_name=None):
    
    """ ref: 
    - https://stackoverflow.com/questions/50203580/how-to-use-matplotlib-to-draw-3d-barplot-with-specific-color-according-to-the-ba
    - https: // matplotlib.org/stable/gallery/mplot3d/3d_bars.html
    """
    # Creating figure
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')

    # Creating plot
    nx, ny = data.shape[-1], data.shape[-2]
    X, Y = numpy.meshgrid(range(nx), range(ny))  # `plot_surface` expects `x` and `y` data to be 2D
    x, y = X.ravel(), Y.ravel()
    data = data.ravel()
    width = depth = 1
    bottom = np.zeros_like(data)
    cmap = plt.get_cmap('plasma', np.max(data) - np.min(data) + 1)

    norm = colors.Normalize(vmin=min(data), vmax=max(data))
    color = cmap(norm(data))
    
    ax.bar3d(x, y, bottom, width, depth, data, color, shade=True)
    
    sc = cm.ScalarMappable(cmap=cmap, norm=norm)
    sc.set_array([])
    plt.colorbar(sc, ticks=np.arange(np.min(data), np.max(data) + 1))
    # ax.set_title(title)
    
    ax.locator_params(nbins=5)  # limit the number of major ticks
    ax.legend(loc='best')  # show legend in a best location
    fig.tight_layout(pad=0.1)  # make layout as tight as possible
    # show plot
    ax.set_facecolor('white')

    if save_name is not None:
        plt.savefig(save_name + '.png', bbox_inches='tight')

    # # displaying the title
    plt.title(title,
            fontsize = 20)

    plt.show()


def plot_loss(iter, loss, filename="loss", label=None, newfig=True, color="b"):
    plt.figure(1)
    plt.clf()
    plt.title(filename)
    plt.xlabel("itrs")
    # plt.ylabel("loss")

    if newfig:
        _ = plt.plot(iter, loss)
        # if filename is not None:
        #     plt.savefig('imgs/'+filename + ".png",
        #                 dpi=200, bbox_inches="tight")
    plt.draw()
    plt.show()


def discrete_matshow(data, title="image", show_flag=True, save_name=None):
    """ 
    Two refs shown below: 
    https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    
    https://matplotlib.org/1.2.1/examples/pylab_examples/poormans_contour.html
    """
    if torch.is_tensor(data):
        if data.device.type != 'cpu':
            print('detect the cuda')
            data = data.detach().cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    axes = fig.add_subplot(111)

    # get discrete colormap   {'tab20', 'RdBu', 'PiYG', 'plasma'}
    cmap = plt.get_cmap('plasma', np.max(data) - np.min(data) + 1)
    # set limits .5 outside true range
    mat = axes.matshow(data, cmap=cmap, vmin=np.min(data) - 0.5,
                      vmax=np.max(data) + 0.5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))
    plt.title(title)

    if save_name is not None:
        plt.rcParams['axes.facecolor'] = 'white'
        plt.savefig(save_name + '.png', bbox_inches='tight')
        
    if show_flag:
        plt.show()
        
# def animate_loss(input, title="image"):
#     fig, ax = plt.subplots()
#     fontdict = {'size': 16}
#     ax.tick_params(axis='both', which='major', direction='in', labelsize=11)
#     ax.set_ylabel("PSNR", fontdict=fontdict)
#     ax.set_xlabel("Iterations", fontdict=fontdict)

#     ax.set_xticks([0, 50, 100, 150, 200])
#     ax.set_xticklabels(['5,000', '10,000', '15,000'])
#     ax.set_xlim(0, 15000)
#     ax.set_yticks([-4, 30, 40, 50, 60])
#     ax.set_ylim(-4, -2)
#     ax.grid()


def show(input, title="image", cut=False, cmap='gray',
         clim=None,
         save_name=None, log_scale=False, show_flag=True):
    if log_scale:
        if torch.is_tensor(input):
            input = torch.log(input)
        else:
            input = np.log(input)

    if torch.is_tensor(input):
        if input.device.type != 'cpu':
            print('detect the cuda')
            input = input.detach().cpu()
            
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    ax.title.set_text(title)
    if cut:
        img = ax.imshow(input, cmap=cmap, vmin=0, vmax=1)
    else:
        img = ax.imshow(input, cmap=cmap)
    plt.colorbar(img, ax=ax)
        
    if save_name is not None:
        plt.rcParams['axes.facecolor'] = 'white'
        plt.savefig(save_name + '.png', bbox_inches='tight')
    
    if show_flag:
        plt.show()
            
