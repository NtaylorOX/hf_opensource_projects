"""
Demo is Derived from  https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py
"""

import numpy as np
import matplotlib.pyplot as plt


from scipy import linalg
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def create_dataset(n_samples=500, n_features=25, rank=5, sigma=1.0, random_state=42, n_components=5):
    '''
    Function to create a dataset with homoscedastic noise and heteroscedastic noise
    '''
    
    # Create a random dataset and add homoscedastic noise and heteroscedastic noise

    rng = np.random.RandomState(random_state)
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    # here n_features must be >= rank as we do a dot product with U[:, :rank].T
    X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)

    # Adding homoscedastic noise
    X_homo = X + sigma * rng.randn(n_samples, n_features)

    # Adding heteroscedastic noise
    sigmas = sigma * rng.rand(n_features) + sigma / 2.0
    X_hetero = X + rng.randn(n_samples, n_features) * sigmas
    n_components_range = np.arange(0, n_features, n_components)
    return X_homo, X_hetero, n_components_range, rank


def compute_scores(X, n_components_range):
    ''' 
    Function to run PCA and FA with different number of componenets and run cross validation
    
    Returns mean PCA and FA scores
    '''
    
    pca = PCA(svd_solver="full")
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components_range:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {"shrinkage": shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X))

#TODO - allow selection of one or both methods
# def plot_pca_fa_analysis(n_features, n_components):
    
#     '''
#     Function to plot results of PCA and FA cross validation analysis
#     '''
    
#     X_homo, X_hetero, n_components_range, rank = create_dataset(n_features=n_features, n_components = n_components)
    
    
#     for X, title in [(X_homo, "Homoscedastic Noise"), (X_hetero, "Heteroscedastic Noise")]:        
        
#         # compute the pca and fa scores
#         pca_scores, fa_scores = compute_scores(X, n_components_range=n_components_range)
#         n_components_pca = n_components_range[np.argmax(pca_scores)]
#         n_components_fa = n_components_range[np.argmax(fa_scores)]

#         pca = PCA(svd_solver="full", n_components="mle")
#         pca.fit(X)
#         n_components_pca_mle = pca.n_components_

#         print("best n_components by PCA CV = %d" % n_components_pca)
#         print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
#         print("best n_components by PCA MLE = %d" % n_components_pca_mle)

#         fig = plt.figure()
#         fig, (ax1, ax2) = plt.subplots(1,2)
#         plt.plot(n_components_range, pca_scores, "b", label="PCA scores")
#         plt.plot(n_components_range, fa_scores, "r", label="FA scores")
#         plt.axvline(rank, color="g", label="TRUTH: %d" % rank, linestyle="-")
#         plt.axvline(
#             n_components_pca,
#             color="b",
#             label="PCA CV: %d" % n_components_pca,
#             linestyle="--",
#         )
#         plt.axvline(
#             n_components_fa,
#             color="r",
#             label="FactorAnalysis CV: %d" % n_components_fa,
#             linestyle="--",
#         )
#         plt.axvline(
#             n_components_pca_mle,
#             color="k",
#             label="PCA MLE: %d" % n_components_pca_mle,
#             linestyle="--",
#         )

#         # compare with other covariance estimators
#         plt.axhline(
#             shrunk_cov_score(X),
#             color="violet",
#             label="Shrunk Covariance MLE",
#             linestyle="-.",
#         )
#         plt.axhline(
#             lw_score(X),
#             color="orange",
#             label="LedoitWolf MLE" % n_components_pca_mle,
#             linestyle="-.",
#         )
        
#         plt.xlabel("nb of components")
#         plt.ylabel("CV scores")
#         plt.legend(loc="lower right")
#         plt.title(title)

#     return fig

def plot_pca_fa_analysis_side(n_samples, n_features, n_components):
    
    X_homo, X_hetero, n_components_range, rank = create_dataset(n_samples = n_samples, n_features=n_features, n_components = n_components)
    
    # set up figure - here we will be doing a side by side plot    
    fig, axes = plt.subplots(2,1, sharey= False, sharex=True, figsize = (10,8))
    
    for X, title, idx in [(X_homo, "Homoscedastic Noise", 0), (X_hetero, "Heteroscedastic Noise", 1)]:       
        
        # compute the pca and fa scores
        pca_scores, fa_scores = compute_scores(X, n_components_range=n_components_range)
        n_components_pca = n_components_range[np.argmax(pca_scores)]
        n_components_fa = n_components_range[np.argmax(fa_scores)]

        pca = PCA(svd_solver="full", n_components="mle")
        pca.fit(X)
        n_components_pca_mle = pca.n_components_

        print("best n_components by PCA CV = %d" % n_components_pca)
        print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
        print("best n_components by PCA MLE = %d" % n_components_pca_mle)
        
        
        axes[idx].plot(n_components_range, pca_scores, "b", label="PCA scores")
        axes[idx].plot(n_components_range, fa_scores, "r", label="FA scores")
        axes[idx].axvline(rank, color="g", label="TRUTH: %d" % rank, linestyle="-")
        axes[idx].axvline(
            n_components_pca,
            color="b",
            label="PCA CV: %d" % n_components_pca,
            linestyle="--",
        )
        axes[idx].axvline(
            n_components_fa,
            color="r",
            label="FactorAnalysis CV: %d" % n_components_fa,
            linestyle="--",
        )
        axes[idx].axvline(
            n_components_pca_mle,
            color="k",
            label="PCA MLE: %d" % n_components_pca_mle,
            linestyle="--",
        )

        # compare with other covariance estimators
        axes[idx].axhline(
            shrunk_cov_score(X),
            color="violet",
            label="Shrunk Covariance MLE",
            linestyle="-.",
        )
        axes[idx].axhline(
            lw_score(X),
            color="orange",
            label="LedoitWolf MLE" % n_components_pca_mle,
            linestyle="-.",
        )
        
    
        # axes[idx].legend(bbox_to_anchor=(1.01, 1.05))
        # plt.xlabel("nb of components")
        # plt.ylabel("CV scores")
        axes[idx].set_xlabel("nb of components")
        axes[idx].set_ylabel("CV scores")
        axes[idx].legend(loc="lower right")
        axes[idx].set_title(title)

    return fig



title = " Illustration of factor analysis vs PCA for dimensionality reduction of a noisy dataset "
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(" This example shows how one can Prinicipal Components Analysis (PCA) and Factor Analysis (FA) for model selection <br>"
    " The number of samples (n_samples) will determine the number of data points to produce.  <br>"
    " The number of components (n_components) will determine the number of components each method will fit to, and will affect the likelihood of the held-out set.  <br>"
    " The number of features (n_components) determine the number of features the toy dataset X variable will have.  <br>"
    " Play with the n_components parameter to see.<br>")

    gr.Markdown(" **[Demo is based on sklearn docs](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py)** <br>")

    gr.Markdown(" **Dataset** : A toy dataset with corrupted with homoscedastic noise (noise variance is the same for each feature) or heteroscedastic noise (noise variance is the different for each feature) . <br>")
    gr.Markdown(" Different number of features and number of components affect how well the low rank space is recovered. <br>"
                "  Larger Depth trying to overfit and learn even the finner details of the data.<br>"
               )

    with gr.Row():
        n_samples = gr.Slider(value=100, min=100, maximum=1000, step=100, label="n_samples")
        n_components = gr.Slider(value=2, min=1, maximum=20, step=1, label="n_components")
        n_features = gr.Slider(value=5, min=5, maximum=25, step=1, label="n_features")
        
    
      # options for n_components    
    btn = gr.Button(value="Submit")
    btn.click(plot_pca_fa_analysis_side, inputs= [n_samples, n_features, n_components], outputs= gr.Plot(label='Multi-output regression with decision trees') ) # 
    

demo.launch()