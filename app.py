import os
import gradio as gr
from cluster_visualize import PredictionArgs, generate_visualization

checkpoint_dir = 'checkpoints'

models = []
checkpoints = []

for root, dirs, files in os.walk(checkpoint_dir):
    for file in files:
        if file.endswith(".pth.tar"):
            models.append(file.split('.')[0])
            checkpoints.append(os.path.join(root, file))

def generate_coc_viz(model_name,
                 image,
                 stage,
                 block,
                 head,
                 alpha):
    model_index = models.index(model_name)
    checkpoint = checkpoints[model_index]
    args = PredictionArgs(
        model=model_name,
        image=image,
        checkpoint=checkpoint,
        stage=stage,
        block=block,
        head=head,
        alpha=alpha
    )
    coc_visualization, probability = generate_visualization(args)
    return probability, coc_visualization


demo = gr.Interface(
    fn=generate_coc_viz,
    inputs=[gr.components.Dropdown(models, label="Model Name"),
            gr.Image(label="Input Image"),
            gr.Slider(0, 3, step=1, label="Stage"),
            gr.Slider(-1, 4, step=1, label="Block"),
            gr.Slider(0, 7, step=1, label="Head"),
            gr.components.Number(0.5, label="Alpha")],
    outputs=[gr.Number(label="Probability"), gr.Image(label="Cluster Visualization")],
)

demo.launch(share=True)
