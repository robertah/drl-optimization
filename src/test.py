import numpy as np
from utils import get_results
from sklearn.decomposition import PCA
import plotly
import plotly.plotly as py
from IPython.display import IFrame
from IPython.core.display import display
from plotly.graph_objs import Scatter3d, Data, Marker, Figure 

#weights = np.load('/Users/fdangelo/Desktop/DeepLearning project/deep-learning/results/weights/CartPole-v1-20190102130024.npy')
#scores = np.load('/Users/fdangelo/Desktop/DeepLearning project/deep-learning/results/scores/CartPole-v1-20190102130024.npy')

weights, scores = get_results()
plotly.tools.set_credentials_file(username='fdfdangelo', api_key='Vxga89BO64eKhvKzhkoR')
n_generations, n_agents = weights.shape[0], weights.shape[1]
w_re = weights.reshape(n_generations * n_agents, -1)
w = np.empty([n_generations * n_agents, w_re[0,0].shape[1]*w_re[0,0].shape[0]+w_re[0,2].shape[0]*w_re[0,2].shape[1]])
sc_re = scores.reshape(n_generations * n_agents, -1)
for i in range(n_generations * n_agents):
    w[i, :] = np.concatenate((w_re[i, 0].flatten(), w_re[i, 2].flatten()))
pca = PCA(n_components=2)
weights_2d = pca.fit_transform(w)


color = []
color.extend([[x]*n_agents for x in range(n_generations)])
flat_color = [item for sublist in color for item in sublist]
# First three dimensions from reduced X VS the Y
trace0 = Scatter3d(
    x=weights_2d[:, 0],
    y=weights_2d[:, 1],
    z=sc_re,
    marker=dict(
        size=7,
        cmax=49,
        cmin=0,
        color=flat_color,
        colorbar=dict(
            title='Colorbar'
        ),
        colorscale='Viridis'
    ),
    mode='markers'
)
data = [trace0]

fig = Figure(data = data)
#py.iplot(fig, filename='pca-cloud')
url = py.plot(fig, filename = 'ciao', validate = False)
display(IFrame(url, '100%', '600px'))