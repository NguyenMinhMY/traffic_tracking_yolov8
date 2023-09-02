import os
import gradio as gr

def video_identity(video):
    return video

demo = gr.Interface(video_identity,
                    gr.Video(),
                    "playable_video",
                    cache_axamples=True
                    )



if __name__ == "__main__":
    demo.launch(share= True)