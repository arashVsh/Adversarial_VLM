"""Render the actual frozen prompt graph for the default backbone."""
def make_graph_figure(path, device):
    import matplotlib.pyplot as plt
    import networkx as nx
    import open_clip
    from Constants import Models
    from Utils import create_model_and_transforms_cached
    from Detection.Pretraining.Anchors import AnchorBank
    from Detection.Pretraining.PromptGraph import PromptGraph
    model, _, _ = create_model_and_transforms_cached(
        Models.VIT_B16,
        pretrained=Models.get_config(Models.VIT_B16)['pretrained'])
    model = model.to(device).eval()
    anchors = AnchorBank(model, open_clip.get_tokenizer(Models.VIT_B16), device)
    anchors.compute_embeddings()
    adjacency = PromptGraph(anchors, device).artifacts().adjacency.cpu().numpy()
    graph = nx.from_numpy_array(adjacency)
    positions = nx.spring_layout(graph, seed=42)
    categories = anchors.category_names
    colors = [categories.index(c) for c in anchors.prompt_categories]
    plt.figure(figsize=(11, 8))
    nx.draw_networkx(graph, positions, with_labels=False, node_color=colors, cmap=plt.get_cmap('tab10'),
                     node_size=[25 + 8 * graph.degree(i) for i in graph], edge_color='#c0c0c0', width=.6)
    for i, category in enumerate(categories):
        plt.scatter([], [], color=plt.get_cmap('tab10')(i / 6), label=category)
    plt.legend(frameon=False)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
